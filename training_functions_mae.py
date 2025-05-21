import os
from sklearn.metrics import r2_score, mean_absolute_error
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


# Training function
def train_model(model, dataloader,dataloader_test, criterion, optimizer, scheduler, num_epochs, params,w=1):
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
    task_type=params['task_type']
    save_folder = str(numpy.char.replace(trained[:-3], 'nets', 'runs'))
    gt_n=params['gt_n']
    # Prep variables for weights and accuracy of the best model
    best_MAE=1000
    best_MAEt=1000
    best_Rt = -1
    best_acc = 0

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
                inputs,labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels=torch.unsqueeze(labels,1)
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    if task_type == 'reg':
                        loss = criterion(outputs, labels)
                    elif task_type == 'cla':
                        labelsc=((labels.floor()-110)/2).floor()
                        loss = criterion(outputs, labelsc.long().squeeze())

                        #KL divergence loss

                    if epoch != num_epochs - 1:
                        loss.backward()
                        optimizer.step()
                # For keeping statistics
                running_loss += loss.item() * inputs.size(0)
                loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            elif len(data) == 4 or len(data) == 5:  #paired inputs
                if len(data) == 4:
                    inputs1, labels1, inputs2, labels2 = data
                else:
                    inputs1,labels1,inputs2,labels2,_ = data
                inputs1 = inputs1.to(device)
                labels1 = labels1.to(device)
                inputs2 = inputs2.to(device)
                labels2 = labels2.to(device)
                labels1=torch.unsqueeze(labels1,1)
                labels2 = torch.unsqueeze(labels2,1)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs1 = model(inputs1)
                    outputs2 = model(inputs2)
                    loss1 = criterion(outputs1, labels1)
                    loss2 = criterion(outputs2, labels2)
                    loss_single = (loss1 + loss2) / 2
                    diff_out = outputs1 - outputs2
                    dif_label = labels1 - labels2
                    loss_pair = criterion(diff_out, dif_label)
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

            # Some current stats
            loss_batch = loss.item()
            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'training:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Training/Loss', loss_accum, niter)
            batch_num = batch_num + 1

        epoch_loss = running_loss / dataset_size
        if board:
            writer.add_scalar('Training/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Training:\t Loss: {:.4f}'.format(epoch_loss))
        utils.print_both(txt_file, '')

        if epoch%1==0 or epoch==num_epochs-1:
            predst, labelst,probst,_,id_arrayt = calculate_predictions(model, dataloader, params,criterion)
            if task_type=='reg':
                predst=predst*gt_n[1]+gt_n[0]
                labelst = labelst * gt_n[1] + gt_n[0]
                Rt = mean_absolute_error(predst.squeeze(), labelst.squeeze())


                if epoch > 100:
                    if Rt<best_MAEt:
                        preds_com_Rt = numpy.concatenate((predst, labelst), 1)
                        numpy.save(os.path.join(save_folder, 'preds_comt_MAE.npy'), preds_com_Rt)
                        numpy.save(os.path.join(save_folder, 'id_arrayt_MAE.npy'), id_arrayt)
                    best_MAEt=Rt
                utils.print_both(txt_file, 'Training:\t MAE: {:.4f}'.format(Rt))
                if board:
                    writer.add_scalar('Training/MAE' + '/Epoch', Rt, epoch + 1)

            preds,labels,probs,val_loss,id_array= calculate_predictions(model, dataloader_test, params,criterion)
            if task_type=='reg':
                preds=preds*gt_n[1]+gt_n[0]
                labels = labels * gt_n[1] + gt_n[0]
                R = mean_absolute_error(preds.squeeze(), labels.squeeze())
                utils.print_both(txt_file, ':\t MAE: {:.4f}'.format(R))

                if epoch: #>100:
                    if R<best_MAE:
                        best_MAE=R
                        preds_com_R = numpy.concatenate((preds, labels), 1)
                        numpy.save(os.path.join(save_folder, 'preds_com_MAE.npy'), preds_com_R)
                        best_model_wts_R = copy.deepcopy(model.state_dict())
                        trained_R = trained.split('.')[0] + '_r' + '.' + trained.split('.')[1]
                        torch.save(best_model_wts_R, trained_R)
                if board:
                    writer.add_scalar('Validation/MAE' + '/Epoch', R, epoch + 1)
            utils.print_both(txt_file, 'Validation:\t Loss: {:.4f}'.format(val_loss))
            if board:
                writer.add_scalar('Validation/Loss' + '/Epoch', val_loss, epoch + 1)

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load the model with the best performance and print the best performance across all epochs
    prediction = numpy.load(os.path.join(save_folder, 'preds_com_MAE.npy'))
    mae = mean_absolute_error(prediction[:, 0], prediction[:, 1])
    utils.print_both(txt_file,'Performance:')
    utils.print_both(txt_file, 'MAE: ' + str(mae))


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params,criterion):
    device=params['device']
    task_type = params['task_type']
    output_array = None
    probs_array = None
    label_array = None
    id_array=None
    model.eval()
    running_loss = 0.0
    for data in dataloader:
        if len(data)==2:
            inputs, labels = data
        elif len(data)==4:
            inputs, labels, inputs2, labels2 = data
        elif len(data)==5:
            inputs, labels, inputs2, labels2,id = data
        inputs = inputs.to(params['device'])
        #print(inputs.size())
        labels = labels.to(params['device']).unsqueeze(dim=1)
        start_time = time.time() #IVAN
        outputs = model(inputs)
        end_time = time.time()#IVAN
        time_taken = end_time - start_time
        #print(f"Time taken to predict a single subject: {time_taken:.4f} seconds")#IVAN

        if task_type == 'reg':
            loss = criterion(outputs, labels)
            probs=outputs
        elif task_type == 'cla':
            labelsc = ((labels.floor() - 110) / 2).floor()
            loss = criterion(outputs, labelsc.long().squeeze())
            probs=torch.exp(outputs)
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
                id_array=id.numpy()
    dataset_size=output_array.shape[0]
    epoch_loss = running_loss / dataset_size
    return output_array, label_array,probs_array,epoch_loss,id_array
