import copy
import time
from scipy.stats import pearsonr
import numpy
import whitematteranalysis as wma
import hcp
import nets
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import fnmatch
import training_functions
import training_functions_mae
import utils
from torch.utils.tensorboard import SummaryWriter
import pandas
import h5py
import wandb
import sys
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

sub_ids = ['{:05d}'.format(i) for i in range(1, 10 + 1)]  # Generate ['00001', '00002', ..., 'num_pd']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')







def find_corresponding_sub_ids(invalid_indices, sub_ids):
    invalid_sub_ids = []
    
    for index in invalid_indices:
        if index < len(sub_ids): 
            invalid_sub_ids.append(sub_ids[index])
        else:
            print(f"Index {index} is out of bounds for sub_ids.")

    return invalid_sub_ids

def find_invalid_samples_in_x_arrays(x_arrays):
    invalid_samples = []
    
    for index, feat_tracts in enumerate(x_arrays):
        for tract, feat_tract in feat_tracts.items():
            if feat_tract.shape[0] == 0: 
                invalid_samples.append(index)
                print(f"Invalid sample found at index {index} with no valid data.")
    
    if not invalid_samples:
        print("All samples have valid data.")
    
    return invalid_samples
    


def remove_invalid_samples(invalid_indices, sub_ids, x_arrays, gt_info):
    new_sub_ids = [sub_id for i, sub_id in enumerate(sub_ids) if i not in invalid_indices_set]
    new_x_arrays = [x_array for i, x_array in enumerate(x_arrays) if i not in invalid_indices_set]

    new_gt_info = gt_info.drop(invalid_indices, axis=0).reset_index(drop=True)

    return new_sub_ids, new_x_arrays, new_gt_info


    
    
    
    
    
    
    
def read_data(data_dir, sub_ids, tracts='AF_left'):
    num_pd = len(sub_ids)  
    feat_arrays = []  
    ff_tracts = []

    if tracts == ['all']:
        tracts = [
            'AF_left', 'AF_right', 'CB_left', 'CB_right', 'EmC_left', 'EmC_right',
            'ILF_left', 'ILF_right', 'IOFF_left', 'IOFF_right', 'MdLF_left', 'MdLF_right',
            'SLF-III_left', 'SLF-III_right', 'SLF-II_left', 'SLF-II_right',
            'SLF-I_left', 'SLF-I_right', 'UF_left', 'UF_right'
        ]

    # Open HDF5 files for all tracts
    for tract in tracts:
        ff = h5py.File(os.path.join(data_dir, f'feats_{tract}.h5'), 'r')
        ff_tracts.append(ff)

    ffeat_min = numpy.array([-100, -100, -100])
    ffeat_md = numpy.array([200, 200, 200])
    index = numpy.array([0, 1, 2])  

    for i in range(num_pd): 
        feat_tracts = {}  
        cluster_id_str = str(sub_ids[i]).zfill(5)  #
        for j, tract in enumerate(tracts):
            ff = ff_tracts[j]

            if cluster_id_str in ff:
                feat = ff[cluster_id_str][:, index] 
                feat_norm = ((feat - ffeat_min) / ffeat_md)  
                feat_tracts[tract] = feat_norm

        if feat_tracts:
            feat_arrays.append(feat_tracts)

    return feat_arrays  # Return populated feature arrays


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument(
        '-indir', action="store", dest="inputDirectory",
        default='./feat_tract',
        help='Input folder including feature files.')
    parser.add_argument(
        '-outdir', action="store", dest="outputDirectory", default="results",
        help='folder to save training records.')
    parser.add_argument('--task', default='tpvt', choices=['age', 'tpvt', 'pert', 'sex', 'read'], help='task')
    parser.add_argument('--tracts', default=['AF_left'], nargs='+', help='task')
    parser.add_argument('--CUDA_id', default='0', choices=['0', '1', '2', '3'], help='choose cuda')
    parser.add_argument('--task_type', default='reg', choices=['reg', 'cla'], help='type of task')
    parser.add_argument('--num_points', default=[2048], nargs='+', type=int, help='sampled points for each subject')
    parser.add_argument('--channels', default=3, type=int, help='number of input channels')
    parser.add_argument('--epochs', default=500, type=int, help='training epochs')
    parser.add_argument('--mode', default='train', choices=['train', 'CRL'],
                        help='train: model training and prediction; vis: localize critical points')
    parser.add_argument('--modeldir', dest='modelDirectory', default='./models', help='model directory for CRL')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--net_architecture', default='PointNet', choices=['PointNet', 'PointNetTRANS'],
                        help='network architecture used')
    parser.add_argument('--dataset', default='PointSet_pair', choices=['PointSet', 'PointSet_pair'],
                        help='PointSet: w/o Lps')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--rate', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--weight', default=0.005, type=float, help='weight decay for clustering')
    parser.add_argument('--weight_loss', default=0.1, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--printing_frequency', default=1, type=int, help='training stats printing frequency')
    parser.add_argument('--seed', default=0, type=int, help='seed for random')
    parser.add_argument('--seed_data', default=1, type=int, help='seed for random')
    parser.add_argument('--Nos_v', default=0, type=int, help='0-number of streamlines 1-density of streamlines')
    args = parser.parse_args()

    setup_seed(args.seed)
    board = args.tensorboard
    dataset = args.dataset
    batch = args.batch_size
    rate = args.rate
    weight = args.weight
    sched_step = args.sched_step
    epochs = args.epochs
    print_freq = args.printing_frequency
    sched_gamma = args.sched_gamma
    channels = args.channels
    tracts = args.tracts
    print(args.num_points)

    wandb.init(project="ShapeWMA", config=args, name="tmp")
    wandb.config.update({
        'batch_size': batch,
        'learning_rate': rate,
        'epochs': epochs,
        'scheduler_step': sched_step,
        'scheduler_gamma': sched_gamma,
        'input_channels': channels,
        'num_points': args.num_points,
        'tracts': args.tracts,
    })

    # Directories
    # Create directories structure
    sub_folder = args.outputDirectory
    txt_folder = os.path.join('reports', sub_folder)
    model_folder = os.path.join('nets', sub_folder)
    event_folder = os.path.join('runs', sub_folder)
    dirs = [event_folder, txt_folder, model_folder]
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    # Net architecture
    model_name = args.net_architecture

    data_dir = args.inputDirectory
    params = {'batch': batch}
    params['print_freq'] = print_freq
    params['Learning rate'] = rate
    params['task_type'] = args.task_type

    
    
    
    
    
    # #reading groudtruth
    gt_info = pandas.read_csv('gt_updated_file.csv')  # d:[62.3989, 16
    # ] min:[90.69, 24]
    sub_ids = gt_info.T.values[0, :].astype(int)  # or numpy.int64
    x_arrays = read_data(data_dir, sub_ids, tracts=tracts)
    invalid_indices = find_invalid_samples_in_x_arrays(x_arrays)
    invalid_sub_ids = find_corresponding_sub_ids(invalid_indices, sub_ids)
    print("Invalid subject IDs corresponding to the invalid indices:", invalid_sub_ids)
    
    invalid_indices_set = set(invalid_indices)  # Convert to a set for faster lookup
    sub_ids, x_arrays, gt_info = remove_invalid_samples(invalid_indices, sub_ids, x_arrays, gt_info)
    
    
    
    # sub_ids=sub_ids[:32]
    if args.task == 'tpvt':
        # Extract three outputs (columns 10, 11, 12)
        print("Shape Column Descriptor:", gt_info.columns[[3, 4, 8, 11, 14]])
        gt1 = gt_info.T.values[[3, 4, 8, 11, 14], :].T 
        # This should give a (15975, 3) shape array

        if args.task_type == 'reg':
            gt_norm = (gt1 - gt1.min(axis=0)) / (gt1.max(axis=0) - gt1.min(axis=0))  # Normalize each column
            params['gt_n'] = [gt1.min(axis=0), gt1.max(axis=0) - gt1.min(axis=0)]  # Store min/max for each output
            print(params['gt_n'][0], params['gt_n'][1])
        elif args.task_type == 'cla':
            gt_norm = gt1  # For classification, just keep the original values
            params['gt_n'] = [gt1.min(axis=0), gt1.max(axis=0) - gt1.min(axis=0)]

    elif args.task == 'pert':
        gt = gt_info.T.values[5, :].T
        gt_norm = gt
        params['gt_n'] = [0, 1]
    elif args.task == 'read':
        gt1 = gt_info.T.values[6, :].T
        gt = numpy.empty(gt1.shape[0])
        gt[:] = gt1[:]
        mean = gt.mean()
        std = gt.std()
        gt_norm = ((gt - gt.min()) / (gt.max() - gt.min())).T
        params['gt_n'] = [gt.min(), gt.max() - gt.min()]
        print(params['gt_n'][0], params['gt_n'][1])

    elif args.task == 'sex':
        gt1 = gt_info.T.values[3, :]
        gt = numpy.empty(gt1.shape[0])
        for i in range(len(gt1)):
            if gt1[i] == 'M':
                gt[i] = 0
            elif gt1[i] == 'F':
                gt[i] = 1
        gt_norm = gt
        params['gt_n'] = [0, 1]
    params['n_points'] = args.num_points

    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    reports_list = sorted(os.listdir(txt_folder), reverse=True)
    if reports_list:
        for file in reports_list:
            # print(file)
            if fnmatch.fnmatch(file, model_name + '_0' + '*'):
                idx = int(str(file)[-7:-4]) + 1
                break
    try:
        idx
    except NameError:
        idx = 1
    isDebug = True if sys.gettrace() else False
    if isDebug == True:
        idx = 0
    print('save_id', idx)
    # Base filename
    name = model_name + '_' + str(idx).zfill(3)
    name_net = name + '.pt'
    name_txt = name + '.txt'
    name_txt = os.path.join(txt_folder, name_txt)
    name_net = os.path.join(model_folder, name_net)
    workers = 4
    print('number of workers:', workers)

    f = open(name_txt, 'w')
    params['model_file'] = name_net
    params['txt_file'] = f

    # Delete tensorboard entry if exist (not to overlap as the charts become unreadable)
    try:
        os.system("rm -rf " + event_folder + name)
    except:
        pass
    # Initialize tensorboard writer
    if board:
        writer = SummaryWriter(event_folder + '/' + name)
        params['writer'] = writer
    else:
        params['writer'] = None

    utils.print_both(f, str(args))
    # Report for settings
    tmp = "Training the '" + model_name + "' architecture"
    utils.print_both(f, tmp)
    tmp = "\n" + "The following parameters are used:"
    utils.print_both(f, tmp)
    tmp = "Batch size:\t" + str(batch)
    utils.print_both(f, tmp)
    tmp = "Number of workers:\t" + str(workers)
    utils.print_both(f, tmp)
    tmp = "Learning rate:\t" + str(rate)
    utils.print_both(f, tmp)
    tmp = "Weight decay:\t" + str(weight)
    utils.print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(sched_step)
    utils.print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(sched_gamma)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(epochs)
    utils.print_both(f, tmp)
    tmp = "Number of input channels:\t" + str(channels)
    utils.print_both(f, tmp)
    tmp = "numner of points used: {}".format(args.num_points)
    utils.print_both(f, tmp)
    tmp = "\nData preparation\nReading data from:\t./" + data_dir
    utils.print_both(f, tmp)

    # Get the training and validation data

    print(f"Number of feature arrays (x_arrays): {len(x_arrays)}")
    print(f"Number of ground truth samples (gt_norm): {len(gt_norm)}")

    x_train, x_test, y_train, y_test = train_test_split(x_arrays, gt_norm, test_size=0.2, random_state=args.seed_data)
    id_train, id_test, _, _ = train_test_split(list(sub_ids), gt_norm, test_size=0.2, random_state=args.seed_data)
    # Save id_train to a text file in the current directory
    with open('./id_train.txt', 'w') as f:
        for item in id_train:
            f.write("%s\n" % item)

    # Save id_test to a text file in the current directory
    with open('./id_test.txt', 'w') as f:
        for item in id_test:
            f.write("%s\n" % item)

    assert len(x_test) == len(y_test)
    if len(tracts) > 3:
        dataset = hcp.Fiber_sub(x_train, y_train, transform=transforms.Compose([transforms.ToTensor()]),
                                n_points=args.num_points)
    else:
        if dataset == 'PointSet':
            dataset_train = hcp.PointSet(x_train, y_train, transform=transforms.Compose([transforms.ToTensor()]),
                                         n_points=args.num_points)
            print(f"Length of Pointset dataset_train: {len(dataset_train)}")
            print(f"x_train length: {len(x_train)}, y_train length: {len(y_train)}")

            dataset_test = hcp.PointSet(x_test, y_test, transform=transforms.Compose([transforms.ToTensor()]),
                                        n_points=args.num_points, val=True)

        elif dataset == 'PointSet_pair':
         
            dataset_train = hcp.PointSet_pair(x_train, y_train, transform=transforms.Compose([transforms.ToTensor()]),
                                              n_points=args.num_points)
            print(f"Length of dataset_train: {len(dataset_train)}")
            print(f"x_train length: {len(x_train)}, y_train length: {len(y_train)}")

            dataset_test = hcp.PointSet_pair(x_test, y_test, transform=transforms.Compose([transforms.ToTensor()]),
                                             n_points=args.num_points, val=True)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch, shuffle=True, num_workers=workers,
                                                   drop_last=False)
    print(f"Length of dataloader_train: {len(dataloader_train)}")

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch, shuffle=False, num_workers=workers,
                                                  drop_last=False)

    with open(name_txt, 'w') as f:
        # Training set size
        dataset_size = len(dataset_train)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)  # Print and log
        params['dataset_size'] = dataset_size

        # Test set size
        dataset_sizev = len(dataset_test)
        tmp = "Test set size:\t" + str(dataset_sizev)
        utils.print_both(f, tmp)

        # Device identification (CUDA or CPU)
        device = torch.device("cuda:{}".format(args.CUDA_id) if torch.cuda.is_available() else "cpu")
        tmp = "\nPerforming calculations on:\t" + str(device)
        utils.print_both(f, tmp + '\n')
        params['device'] = device

    # Evaluate the proper model
    if args.net_architecture == 'PointNet' or args.net_architecture == 'PointNetTRANS':
        to_eval = "nets." + model_name + "(input_channel=channels,task_type=args.task_type)"
        model = eval(to_eval)

    model = model.to(device)
    if args.task_type == 'reg':
        criteria = nn.MSELoss(reduction='mean')
    elif args.task_type == 'cla':
        criteria = nn.NLLLoss(size_average=True)

    if args.mode == 'train':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
        model = training_functions.train_model(model, dataloader_train, dataloader_test, criteria, optimizer, scheduler,
                                               epochs, params, w=args.weight_loss)
        # model = training_functions_mae.train_model(model,dataloader_train,dataloader_test,criteria,optimizer,scheduler,epochs,params,w=args.weight_loss)


    elif args.mode == 'CRL':
        model.load_state_dict(torch.load(args.modelDirectory))
        print(args.modelDirectory)
        model = model.to(device)
        key_points = []
        counts_all = []

        tract = tracts[0]
        save_folder = args.outputDirectory + '/' + tract
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        npoint_tract = numpy.array([x[tract].shape[0] for x in x_test])
        f1 = h5py.File(os.path.join(save_folder, 'key_points.h5'), 'w')
        f2 = h5py.File(os.path.join(save_folder, 'counts.h5'), 'w')
        for k in range(len((x_test))):
            start = time.time()
            print(k, id_test[k])
            x_test1 = x_test[k][tract]
            y_test1 = y_test[k]
            preds_sub = []
            labels_sub = []
            for m in range(num_pd):
                print('m', m)
                numpy.random.shuffle(x_test1)
                times = int(numpy.floor(x_test1.shape[0] / args.num_points[0]))
                x_test_sub = [x_test1 for s in range(times)]
                y_test_sub = [y_test1 for s in range(times)]
                datasetv = hcp.Fiber_pair_vis(x_test_sub, y_test_sub,
                                              transform=transforms.Compose([transforms.ToTensor()]),
                                              n_points=args.num_points[0])
                dataloaderv = torch.utils.data.DataLoader(datasetv, batch_size=batch, shuffle=False,
                                                          num_workers=workers)
                preds, labels, probs, _, points, ids = training_functions.calculate_predictions_vis(model, dataloaderv,
                                                                                                    params, criteria)

                ids_onehot = numpy.zeros((ids.shape[0], points.shape[2]))
                for i in range(ids.shape[0]):
                    id_sub = ids[i, :]
                    temp, counts = numpy.unique(id_sub, return_counts=True)
                    temp = temp[1:]
                    counts = counts[1:]
                    key_p = points[i, :, temp]
                    if i == 0 and m == 0:
                        points_sub = key_p
                        counts_sub = counts
                    else:
                        points_sub = numpy.concatenate((points_sub, key_p), 0)
                        counts_sub = numpy.concatenate((counts_sub, counts), 0)

            pointu, inverse = numpy.unique(points_sub, axis=0, return_inverse=True)
            countu = numpy.zeros((pointu.shape[0]), dtype=int)
            for s in range(pointu.shape[0]):
                countu[s] = int(numpy.sum(counts_sub[inverse == s]))
            f1[str(id_test[k])] = pointu[:, :3]
            f2[str(id_test[k])] = countu
            print(len(countu))

        f1.close()
        f2.close()
    f.close()



