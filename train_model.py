import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from train_unet.utils.my_dataset import My_dataset
from train_unet.utils.dice_score import dice_coeff, multiclass_dice_coeff,dice_loss
import random
from train_unet.utils.utils_prediction import mask_to_image
from datetime import datetime
import os
import yaml


def train_net(net, ikDataset, mapping, epochs, batch_size, learning_rate, device,
              val_percentage, img_size, output_folder, stop, log_mlflow, step, writer=None):
    
    # current date time used to name output files
    str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
    # 2. Split into train / validation partitions
    random.seed(seed)
    random.shuffle(ikDataset["images"])
    n_val = int(len(ikDataset["images"]) * val_percentage)
    n_train = len(ikDataset["images"]) - n_val
    # load class names from dataset
    class_names = ikDataset['metadata']['category_names']
    dataset = My_dataset({"metadata": ikDataset["metadata"], "images": ikDataset["images"]}, img_size, mapping)

    db_train, db_test = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    train_loader = DataLoader(db_train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(db_test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    net.train()

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(params=net.parameters(), lr=learning_rate, alpha=alpha, eps=eps,
                              weight_decay=weight_decay, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    # global iterations
    tr_global_step = 0
    valid_global_step = 0
    best_score = 0
    delta = 0.0001

    # 5. Begin training

    for epoch in range(epochs):
        if stop():
            break
        epoch_loss = 0
        epoch_dice_score = 0
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as train_bar:
            for batch in train_loader:
                if stop():
                    break
                images = batch['image']
                true_masks = batch['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                optimizer.zero_grad()
                masks_pred = net(images)

                # cross entrpy loss : input shape(batch size, num class, h, w) , targuet shape(batch size, h, w)
                # targuet must be the localisation of the classes
                if net.n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                # one hot vector of shape (batch_size, num_classes, W, H)

                loss.backward()

                optimizer.step()

                train_bar.update(images.shape[0])
                tr_global_step += 1
                step()

                epoch_loss += loss.item()

                # calculate training score
                running_dice_score = multiclass_dice_coeff(F.softmax(masks_pred.float(), dim=1).float(),
                                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1,2).float(),
                                                           reduce_batch_first=False, epsilon=1e-6)
                epoch_dice_score += running_dice_score.item()

                train_bar.set_postfix(**{'running loss': loss.item(), 'running_dice_score': running_dice_score.item()})

        # Add training epoch loss and score on tensorboard
        writer.add_scalar('Epoch_Loss/train', epoch_loss/len(train_loader), epoch)
        writer.add_scalar('Epoch_Dice_score/train', epoch_dice_score/len(train_loader), epoch)
        # Log  metrics to MLflow
        metrics = {'Epoch_Loss/train': epoch_loss/len(train_loader), 'Epoch_Dice_score/train': epoch_dice_score/len(train_loader)}
        log_mlflow(metrics, epoch)


        net.eval()

        validation_score = 0
        loss_val_ep = 0
        IOU_validation_score = 0
        # iterate over the validation set
        with tqdm(total=n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as valid_bar:
            for batch in val_loader:

                image, mask_true = batch['image'], batch['mask']
                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32)
                mask_true = mask_true.to(device=device, dtype=torch.long)
                #mask_true = torch.argmax(mask_true, dim=1)

                with torch.no_grad():
                    # predict the mask
                    mask_pred = net(image)
                    if net.n_classes == 1:
                        loss_val = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss_val += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss_val = criterion(masks_pred, true_masks)
                        loss_val += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    loss_val_ep += loss_val.item()
                    # compute the Dice score, ignoring background
                    val_score = multiclass_dice_coeff(F.softmax(mask_pred.float(), dim=1).float(),
                                                      F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                                                      reduce_batch_first=False, epsilon=1e-6)
                    validation_score += val_score.item()

                    valid_bar.update(image.shape[0])
                    valid_global_step += 1

                    valid_bar.set_postfix(
                        **{'validation loss': loss_val.item(), 'runing validation_score': val_score.item()})

                    # show validation images in tensorboard
                    writer.add_image('Validation/Image', image[0], valid_global_step, dataformats='CHW')
                    writer.add_image('Validation/Prediction', torch.argmax(torch.softmax(mask_pred[0], dim=0), dim=0), valid_global_step, dataformats='HW')
                    writer.add_image('Validation/GroundTruth', mask_true[0], valid_global_step, dataformats='HW')

        epoch_valid_loss = loss_val_ep / len(val_loader)
        epoch_valid_score = validation_score / len(val_loader)

        # Add validation epoch loss and score on tensorboard
        writer.add_scalar('Loss/Val', epoch_valid_loss, epoch)
        writer.add_scalar('Dice_score/Val', epoch_valid_score, epoch)
        # Log  metrics to MLflow
        val_metrics = {'Loss/Val': epoch_valid_loss,
                   'Dice_score/Val': epoch_valid_score}
        log_mlflow(val_metrics, epoch)

        net.train()

        if epoch_dice_score > best_score - delta:
            best_score = epoch_dice_score
            model_path = os.path.join(output_folder, 'trained_model'+str_datetime+'.pth')
            model_dict = {'state_dict': net.state_dict(),
                        'class_names': class_names}
            torch.save(model_dict, model_path)
            print("save model to {}".format(output_folder))

    writer.close()

    return "Training Finished!"


# load yaml file : extract model parameters
config_path = os.path.dirname(os.path.realpath(__file__)) + "/config.yaml"
with open(config_path) as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)

seed = params['training_params']['seed']
weight_decay = float(params['training_params']['weight_decay'])
momentum = float(params['training_params']['momentum'])
alpha = float(params['training_params']['alpha'])
eps = float(params['training_params']['eps'])

loss_function = params['training_params']['loss_function']
loss_function = eval(loss_function + "()")

# dataloader
num_workers = params['training_params']['num_workers']
pin_memory = params['training_params']['pin_memory']