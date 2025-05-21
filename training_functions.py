import os
from sklearn.metrics import r2_score
import numpy
from nets import PointNet
import utils
import time
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import random
import wandb
import csv


# Training function
def train_model(model, dataloader, dataloader_test, criterion, optimizer, scheduler, num_epochs, params, w=1):
    # Note the time
    since = time.time()
    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    trained = params['model_file']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    lr = params['Learning rate']
    task_type = params['task_type']
    save_folder = str(numpy.char.replace(trained[:-3], 'nets', 'runs'))
    gt_n = params['gt_n']
    # Prep variables for weights and accuracy of the best model
    best_mae = 1000
    best_R = -1
    best_Rt = -1
    best_acc = 0
    best_R_0 = -float('inf')
    best_R_1 = -float('inf')
    best_R_2 = -float('inf')
    best_R_3 = -float('inf')
    best_R_4 = -float('inf')

    best_Rt_0 = -float('inf')
    best_Rt_1 = -float('inf')
    best_Rt_2 = -float('inf')
    best_Rt_3 = -float('inf')
    best_Rt_4 = -float('inf')

    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_single = 0.0
        running_loss_pair = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1

        # Iterate over data.
        for data in dataloader:
            if len(data) == 2:
                # Get the inputs and labels
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                #                labels=torch.unsqueeze(labels,1)
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    if task_type == 'reg':
                        loss = criterion(outputs, labels)
                    elif task_type == 'cla':
                        labelsc = ((labels.floor() - 110) / 2).floor()
                        loss = criterion(outputs, labelsc.long().squeeze())

                        # KL divergence loss

                    if epoch != num_epochs - 1:
                        loss.backward()
                        optimizer.step()
                # For keeping statistics
                running_loss += loss.item() * inputs.size(0)
                loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            elif len(data) == 4 or len(data) == 5:  # paired inputs
                if len(data) == 4:
                    inputs1, labels1, inputs2, labels2 = data
                else:
                    inputs1, labels1, inputs2, labels2, _ = data
                inputs1 = inputs1.to(device)
                labels1 = labels1.to(device)
                inputs2 = inputs2.to(device)
                labels2 = labels2.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs1 = model(inputs1)
                    outputs2 = model(inputs2)
                    loss1 = criterion(outputs1, labels1)
                    loss2 = criterion(outputs2, labels2)
                    loss_single = (loss1 + loss2) / 2
                    diff_out = outputs1 - outputs2
                    dif_label = labels1 - labels2
                    pred_fft = torch.fft.fft(diff_out)
                    true_fft = torch.fft.fft(dif_label)
                    magnitude_pred = torch.abs(pred_fft)
                    magnitude_true = torch.abs(true_fft)

                    loss_pair= criterion(magnitude_pred, magnitude_true)

                    loss = loss_single + w * loss_pair 
                    if epoch != num_epochs - 1:
                        loss.backward()
                        optimizer.step()
                    # For keeping statistics
                    running_loss += loss.item() * inputs1.size(0)
                    running_loss_single += loss_single.item() * inputs1.size(0)
                    running_loss_pair += loss_pair.item() * inputs1.size(0)
                    loss_accum = running_loss / ((batch_num - 1) * batch + inputs1.size(0))
                    loss_accum_single = running_loss_single / ((batch_num - 1) * batch + inputs1.size(0))
                    loss_accum_pair = running_loss_pair / ((batch_num - 1) * batch + inputs1.size(0))
                    if batch_num % print_freq == 0:
                        if board:
                            niter = epoch * len(dataloader) + batch_num
                            writer.add_scalar('Training/Loss_single', loss_accum_single, niter)
                            writer.add_scalar('Training/Loss_pair', loss_accum_pair, niter)


            loss_batch = loss.item()

            if batch_num == 1 or batch_num == len(dataloader):
                utils.print_both(txt_file, 'training:\tEpoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                                             loss_batch,
                                                                             loss_accum))
            if board:
                niter = epoch * len(dataloader) + batch_num
                writer.add_scalar('Training/Loss', loss_accum, niter)

            batch_num = batch_num + 1

        epoch_loss = running_loss / dataset_size
        wandb.log({'Training/Loss': epoch_loss})
        if board:
            writer.add_scalar('Training/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Training:\t Loss: {:.4f}'.format(epoch_loss))
        utils.print_both(txt_file, '')

        if epoch % 1 == 0 or epoch == num_epochs - 1:
            # Training correlation
            predst, labelst, probst, _, id_arrayt = calculate_predictions(model, dataloader, params, criterion)
            if task_type == 'reg':
                predst = predst * gt_n[1] + gt_n[0]
                labelst = labelst * gt_n[1] + gt_n[0]

                # Separate Pearson correlation for each output dimension
                Rt_0 = pearsonr(predst.squeeze()[:, 0], labelst.squeeze()[:, 0])[0]
                Rt_1 = pearsonr(predst.squeeze()[:, 1], labelst.squeeze()[:, 1])[0]
                Rt_2 = pearsonr(predst.squeeze()[:, 2], labelst.squeeze()[:, 2])[0] 
                Rt_3 = pearsonr(predst.squeeze()[:, 3], labelst.squeeze()[:, 3])[0]
                Rt_4 = pearsonr(predst.squeeze()[:, 4], labelst.squeeze()[:, 4])[0]
                # Log each Pearson R to WandB
                wandb.log({
                    'Training/R_0': Rt_0,
                    'Training/R_1': Rt_1,
                    'Training/R_2': Rt_2, 
                    'Training/R_3': Rt_3,
                    'Training/R_4': Rt_4,
                })

                utils.print_both(txt_file,
                                 f'Training:\t R_0: {Rt_0:.4f}, R_1: {Rt_1:.4f}, R_2: {Rt_2:.4f}, R_3: {Rt_3:.4f}, R_4: {Rt_4:.4f},') 

                # Save the best Pearson correlation for each output dimension
                if epoch > 100:

                    if labelst.ndim == 3:
                        labelst = labelst.squeeze()  # Remove the singleton dimension if it exists
                    if Rt_0 > best_Rt_0:
                        best_Rt_0 = Rt_0
                        numpy.save(os.path.join(save_folder, 'preds_comt_R_0.npy'),
                                   numpy.concatenate((predst, labelst), 1))
                        numpy.save(os.path.join(save_folder, 'id_arrayt_R_0.npy'), id_arrayt)

                    if Rt_1 > best_Rt_1:
                        best_Rt_1 = Rt_1
                        numpy.save(os.path.join(save_folder, 'preds_comt_R_1.npy'),
                                   numpy.concatenate((predst, labelst), 1))
                        numpy.save(os.path.join(save_folder, 'id_arrayt_R_1.npy'), id_arrayt)

                    if Rt_2 > best_Rt_2:
                        best_Rt_2 = Rt_2
                        numpy.save(os.path.join(save_folder, 'preds_comt_R_2.npy'),
                                   numpy.concatenate((predst, labelst), 1))
                        numpy.save(os.path.join(save_folder, 'id_arrayt_R_2.npy'), id_arrayt) 
                    if Rt_3 > best_Rt_3:
                        best_Rt_3 = Rt_3
                        numpy.save(os.path.join(save_folder, 'preds_comt_R_3.npy'),
                                   numpy.concatenate((predst, labelst), 1))
                        numpy.save(os.path.join(save_folder, 'id_arrayt_R_3.npy'), id_arrayt)
                    if Rt_4 > best_Rt_4:
                        best_Rt_4 = Rt_4
                        numpy.save(os.path.join(save_folder, 'preds_comt_R_4.npy'),
                                   numpy.concatenate((predst, labelst), 1))
                        numpy.save(os.path.join(save_folder, 'id_arrayt_R_4.npy'), id_arrayt)

                # Tensorboard logging (optional)
                if board:
                    writer.add_scalar('Training/R_0/Epoch', Rt_0, epoch + 1)
                    writer.add_scalar('Training/R_1/Epoch', Rt_1, epoch + 1)
                    writer.add_scalar('Training/R_2/Epoch', Rt_2, epoch + 1) 
                    writer.add_scalar('Training/R_3/Epoch', Rt_3, epoch + 1)
                    writer.add_scalar('Training/R_4/Epoch', Rt_4, epoch + 1)

            # Validation correlation
            # Validation correlation
            preds, labels, probs, val_loss, id_array = calculate_predictions(model, dataloader_test, params, criterion)
            if task_type == 'reg':
                preds = preds * gt_n[1] + gt_n[0]
                labels = labels * gt_n[1] + gt_n[0]
                # Check and modify the shape of labels to match preds
                if labels.ndim == 3:
                    labels = labels.squeeze()  # Remove the singleton dimension if it exists

                # Separate Pearson correlation for each output dimension
                R_0 = pearsonr(preds.squeeze()[:, 0], labels.squeeze()[:, 0])[0]
                R_1 = pearsonr(preds.squeeze()[:, 1], labels.squeeze()[:, 1])[0]
                R_2 = pearsonr(preds.squeeze()[:, 2], labels.squeeze()[:, 2])[0] 
                R_3 = pearsonr(preds.squeeze()[:, 3], labels.squeeze()[:, 3])[0]
                R_4 = pearsonr(preds.squeeze()[:, 4], labels.squeeze()[:, 4])[0]

                # Log validation Pearson R for each dimension to WandB
                wandb.log({
                    'Validation/R_0': R_0,
                    'Validation/R_1': R_1,
                    'Validation/R_2': R_2, 
                    'Validation/R_3': R_3,
                    'Validation/R_4': R_4,
                    'Validation Loss': val_loss,
                })

                utils.print_both(txt_file,
                                 f'Validation:\t R_0: {R_0:.4f}, R_1: {R_1:.4f}, R_2: {R_2:.4f}, R_3: {R_3:.4f}, R_4: {R_4:.4f}') 
                utils.print_both(txt_file, 'Validation:\t Loss: {:.4f}'.format(val_loss))

                # Save the best Pearson correlation for validation
                if epoch:  # > 100:
                    if R_0 > best_R_0:
                        best_R_0 = R_0
                        numpy.save(os.path.join(save_folder, 'preds_com_R_0.npy'),
                                   numpy.concatenate((preds, labels), 1))
                        best_model_wts_R_0 = copy.deepcopy(model.state_dict())
                        trained_R_0 = trained.split('.')[0] + '_r_0' + '.' + trained.split('.')[1]
                        torch.save(best_model_wts_R_0, trained_R_0)
                        wandb.log({'Best Validation R_0': best_R_0})

                    if R_1 > best_R_1:
                        best_R_1 = R_1
                        numpy.save(os.path.join(save_folder, 'preds_com_R_1.npy'),
                                   numpy.concatenate((preds, labels), 1))
                        best_model_wts_R_1 = copy.deepcopy(model.state_dict())
                        trained_R_1 = trained.split('.')[0] + '_r_1' + '.' + trained.split('.')[1]
                        torch.save(best_model_wts_R_1, trained_R_1)
                        wandb.log({'Best Validation R_1': best_R_1})

                    if R_2 > best_R_2:
                        best_R_2 = R_2
                        numpy.save(os.path.join(save_folder, 'preds_com_R_2.npy'),
                                   numpy.concatenate((preds, labels), 1))
                        best_model_wts_R_2 = copy.deepcopy(model.state_dict())
                        trained_R_2 = trained.split('.')[0] + '_r_2' + '.' + trained.split('.')[1]
                        torch.save(best_model_wts_R_2, trained_R_2)
                        wandb.log({'Best Validation R_2': best_R_2}) 
                    if R_3 > best_R_3:
                        best_R_3 = R_3
                        numpy.save(os.path.join(save_folder, 'preds_com_R_3.npy'),
                                   numpy.concatenate((preds, labels), 1))
                        best_model_wts_R_3 = copy.deepcopy(model.state_dict())
                        trained_R_3 = trained.split('.')[0] + '_r_3' + '.' + trained.split('.')[1]
                        torch.save(best_model_wts_R_3, trained_R_3)
                        wandb.log({'Best Validation R_3': best_R_3})
                    if R_4 > best_R_4:
                        best_R_4 = R_4
                        numpy.save(os.path.join(save_folder, 'preds_com_R_4.npy'),
                                   numpy.concatenate((preds, labels), 1))
                        best_model_wts_R_4 = copy.deepcopy(model.state_dict())
                        trained_R_4 = trained.split('.')[0] + '_r_4' + '.' + trained.split('.')[1]
                        torch.save(best_model_wts_R_4, trained_R_4)
                        wandb.log({'Best Validation R_4': best_R_4})

                # Tensorboard logging (optional)
                if board:
                    writer.add_scalar('Validation/R_0/Epoch', R_0, epoch + 1)
                    writer.add_scalar('Validation/R_1/Epoch', R_1, epoch + 1)
                    writer.add_scalar('Validation/R_2/Epoch', R_2, epoch + 1) 
                    writer.add_scalar('Validation/R_3/Epoch', R_3, epoch + 1)
                    writer.add_scalar('Validation/R_4/Epoch', R_4, epoch + 1)
                writer.add_scalar('Validation/Loss/Epoch', val_loss, epoch + 1)


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params, criterion):
    device = params['device']
    task_type = params['task_type']
    output_array = None
    probs_array = None
    label_array = None
    id_array = None
    model.eval()
    running_loss = 0.0
    # Directory and file path
    log_dir = "./logs"
    log_file = os.path.join(log_dir, "prediction_times.csv")

    # Ensure the logs directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Check if the file exists, and if not, add headers
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Subject_ID', 'Time_Taken'])  # Add headers
    for data in dataloader:
        if len(data) == 2:
            inputs, labels = data
        elif len(data) == 4:
            inputs, labels, inputs2, labels2 = data
        elif len(data) == 5:
            inputs, labels, inputs2, labels2, id = data
        inputs = inputs.to(params['device'])
        # print(inputs.size())
        labels = labels.to(params['device']).unsqueeze(dim=1)
        start_time = time.time()  # IVAN
        outputs = model(inputs)
        end_time = time.time()  # IVAN
        time_taken = end_time - start_time
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if 'id' in locals():
                writer.writerow([id, time_taken])  # Assuming 'id' is the Subject ID
            else:
                writer.writerow([None, time_taken])  # In case id is not available
        # print(f"Time taken to predict a single subject: {time_taken:.4f} seconds")#IVAN

        if task_type == 'reg':
            loss = criterion(outputs, labels)
            probs = outputs
        elif task_type == 'cla':
            labelsc = ((labels.floor() - 110) / 2).floor()
            loss = criterion(outputs, labelsc.long().squeeze())
            probs = torch.exp(outputs)
            outputs = outputs.data.max(1)[1].unsqueeze(dim=1)

        running_loss += loss.item() * inputs.size(0)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            probs_array = np.concatenate((probs_array, probs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
            if len(data) == 5:
                id_array = np.concatenate((id_array, id.numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()
            probs_array = probs.cpu().detach().numpy()
            if len(data) == 5:
                id_array = id.numpy()
    dataset_size = output_array.shape[0]
    epoch_loss = running_loss / dataset_size
    return output_array, label_array, probs_array, epoch_loss, id_array


def calculate_predictions_vis(model, dataloader, params, criterion):
    task_type = params['task_type']
    output_array = None
    probs_array = None
    label_array = None
    point_array = None
    ids_array = []
    model.eval()
    running_loss = 0.0
    for data in dataloader:
        if len(data) == 2:
            inputs, labels = data
        elif len(data) == 4:
            inputs, labels, inputs2, labels2 = data
        elif len(data) == 5:
            inputs, labels, inputs2, labels2, id = data

        inputs = inputs.to(params['device'])
        labels = labels.to(params['device']).unsqueeze(dim=1)
        outputs, ids = model(inputs)
        if task_type == 'reg':
            loss = criterion(outputs, labels)
            probs = outputs
        elif task_type == 'cla':
            loss = criterion(outputs, labels.long().squeeze())
            probs = torch.exp(outputs[:, 0])
            outputs = outputs.data.max(1)[1].unsqueeze(dim=1)
        running_loss += loss.item() * inputs.size(0)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            probs_array = np.concatenate((probs_array, probs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
            point_array = np.concatenate((point_array, inputs.cpu().detach().numpy()), 0)
            ids_array = np.concatenate((ids_array, ids.cpu().detach().numpy()), 0)

        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()
            probs_array = probs.cpu().detach().numpy()
            point_array = inputs.cpu().detach().numpy()
            ids_array = ids.cpu().detach().numpy()
    dataset_size = output_array.shape[0]
    epoch_loss = running_loss / dataset_size

    # print(output_array.shape)
    return output_array, label_array, probs_array, epoch_loss, point_array, ids_array