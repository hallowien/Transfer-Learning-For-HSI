
import argparse
import collections
import math
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics, preprocessing
from sklearn.metrics import confusion_matrix
from torchsummary import summary
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import Utils

PARAM_DATASET = "IN"
PARAM_EPOCH = 1
PARAM_ITER = 5
PATCH_SIZE = 1
PARAM_VAL = 0.2
PARAM_OPTIM = 'adam'
# # Data Loading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

import numpy as np
import torch
from operator import truediv

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
         for i, data in enumerate(data_iter, 0):
                  
            test_l_sum, test_num = 0, 0
            
            inputs, targets = data
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
                #print("y 1 ",y.shape)
            targets = targets.to(device)
                
            net.eval() 
            outputs = net(inputs)
                
                # Compute loss
            loss = loss_function(outputs, targets.long())
            acc_sum += (outputs.argmax(dim=1) == targets.to(device)).float().sum().cpu().item()
            test_l_sum += loss
            test_num += 1
            net.train() 
            n +=  targets.shape[0]
    return [acc_sum / n, test_l_sum] # / test_num]


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def record_output(oa_ae, aa_ae, kappa_ae, f1_ae, element_acc_ae, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n'
    f.write(sentence2)
    sentence10 = 'F1 scores for each iteration are:' + str(f1_ae) + '\n' + '\n'
    f.write(sentence10)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' 
    f.write(sentence5)
    sentence11 = 'mean_F1 ± std_F1 is: ' + str(np.mean(f1_ae)) + ' ± ' + str(np.std(f1_ae)) + '\n' + '\n'
    f.write(sentence11)
    sentence6 = 'Total average Training time is: ' + str(np.sum(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'
    f.write(sentence9)
    f.close()
    


global Dataset  # UP,IN,SV, KSC
dataset = PARAM_DATASET  #input('Please input the name of Dataset(IN, UP, SV, KSC):')
Dataset = dataset.upper()


def load_dataset(Dataset, split=0.8):
    if Dataset == 'IN':
        mat_data = sio.loadmat('Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        K = 200
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        uPavia = sio.loadmat('PaviaU.mat')
        gt_uPavia = sio.loadmat('PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        K = 103
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = split
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
        

    shapeor = data_hsi.shape
    shapeor = np.array(shapeor)
    shapeor[-1] = K
    data_hsi = data_hsi.reshape(shapeor)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


# # Pytorch Data Loader Creation
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(Dataset, PARAM_VAL)
print(data_hsi.shape)

image_x, image_y, BAND = data_hsi.shape

data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))

gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )

CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = PARAM_ITER
PATCH_LENGTH = PATCH_SIZE
lr, num_epochs, batch_size = 0.001, 100, 32
loss = torch.nn.CrossEntropyLoss()


img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

F1 = []
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))



data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
data = preprocessing.scale(data)

data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])

whole_data = data_

padded_data = np.lib.pad(
    whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH),
                 (0, 0)),
    'constant',
    constant_values=0)


# # Model


"""nasnet in pytorch



[1] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le

    Learning Transferable Architectures for Scalable Image Recognition
    https://arxiv.org/abs/1707.07012
"""

import torch
import torch.nn as nn

class SeperableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            **kwargs
        )

        self.pointwise = nn.Conv2d(
            input_channels,
            output_channels,
            1
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class SeperableBranch(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        """Adds 2 blocks of [relu-separable conv-batchnorm]."""
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(output_channels, output_channels, kernel_size, stride=1, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(output_channels)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x

class Fit(nn.Module):
    """Make the cell outputs compatiblese

    Args:
        prev_filters: filter number of tensor prev, needs to be modified
        filters: filter number of normal cell branch output filters
    """

    def __init__(self, prev_filters, filters):
        super().__init__()
        self.relu = nn.ReLU()

        self.p1 = nn.Sequential(
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        #make sure there is no information loss
        self.p2 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConstantPad2d((-1, 0, -1, 0), 0),   #cropping
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        self.bn = nn.BatchNorm2d(filters)

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(prev_filters, filters, 1),
            nn.BatchNorm2d(filters)
        )

        self.filters = filters
    
    def forward(self, inputs):
        x, prev = inputs
        if prev is None:
            return x

        #image size does not match
        elif x.size(2) != prev.size(2):
            prev = self.relu(prev)
            p1 = self.p1(prev)
            p2 = self.p2(prev)
            prev = torch.cat([p1, p2], 1)
            prev = self.bn(prev)

        elif prev.size(1) != self.filters:
            prev = self.dim_reduce(prev)

        return prev


class NormalCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dem_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels)
        )

        self.block1_left = SeperableBranch(
            output_channels, 
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.block1_right = nn.Sequential()

        self.block2_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.block2_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False
        )

        self.block3_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block3_right = nn.Sequential()

        self.block4_left = nn.AvgPool2d(3, stride=1, padding=1)
        self.block4_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.block5_left = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=5,
            padding=2,
            bias=False
        )
        self.block5_right = SeperableBranch(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.fit = Fit(prev_in, output_channels)
    
    def forward(self, x):
        x, prev = x

        #return transformed x as new x, and original x as prev 
        #only prev tensor needs to be modified
        prev = self.fit((x, prev)) 

        h = self.dem_reduce(x)

        x1 = self.block1_left(h) + self.block1_right(h)
        x2 = self.block2_left(prev) + self.block2_right(h)
        x3 = self.block3_left(h) + self.block3_right(h)
        x4 = self.block4_left(prev) + self.block4_right(prev)
        x5 = self.block5_left(prev) + self.block5_right(prev)

        return torch.cat([prev, x1, x2, x3, x4, x5], 1), x

class ReductionCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1),
            nn.BatchNorm2d(output_channels)
        )

        #block1
        self.layer1block1_left = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)
        self.layer1block1_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        #block2
        self.layer1block2_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1block2_right = SeperableBranch(output_channels, output_channels, 7, stride=2, padding=3)

        #block3
        self.layer1block3_left = nn.AvgPool2d(3, 2, 1)
        self.layer1block3_right = SeperableBranch(output_channels, output_channels, 5, stride=2, padding=2)

        #block5
        self.layer2block1_left = nn.MaxPool2d(3, 2, 1)
        self.layer2block1_right = SeperableBranch(output_channels, output_channels, 3, stride=1, padding=1)

        #block4
        self.layer2block2_left = nn.AvgPool2d(3, 1, 1)
        self.layer2block2_right = nn.Sequential()
    
        self.fit = Fit(prev_in, output_channels)
    
    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))

        h = self.dim_reduce(x)

        layer1block1 = self.layer1block1_left(prev) + self.layer1block1_right(h)
        layer1block2 = self.layer1block2_left(h) + self.layer1block2_right(prev)
        layer1block3 = self.layer1block3_left(h) + self.layer1block3_right(prev)
        layer2block1 = self.layer2block1_left(h) + self.layer2block1_right(layer1block1)
        layer2block2 = self.layer2block2_left(layer1block1) + self.layer2block2_right(layer1block2)

        return torch.cat([
            layer1block2, #https://github.com/keras-team/keras-applications/blob/master/keras_applications/nasnet.py line 739
            layer1block3,
            layer2block1,
            layer2block2
        ], 1), x


class NasNetA(nn.Module):

    def __init__(self, repeat_cell_num, reduction_num, filters, stemfilter, class_num=16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(200, stemfilter, 3, padding=1, bias=False),
            nn.BatchNorm2d(stemfilter)
        )

        self.prev_filters = stemfilter
        self.x_filters = stemfilter
        self.filters = filters

        self.cell_layers = self._make_layers(repeat_cell_num, reduction_num)

        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.filters * 6, class_num)
    
    
    def _make_normal(self, block, repeat, output):
        """make normal cell
        Args:
            block: cell type
            repeat: number of repeated normal cell
            output: output filters for each branch in normal cell
        Returns:
            stacked normal cells
        """

        layers = [] 
        for r in range(repeat):
            layers.append(block(self.x_filters, self.prev_filters, output))
            self.prev_filters = self.x_filters
            self.x_filters = output * 6 #concatenate 6 branches
        
        return layers

    def _make_reduction(self, block, output):
        """make normal cell
        Args:
            block: cell type
            output: output filters for each branch in reduction cell
        Returns:
            reduction cell
        """

        reduction = block(self.x_filters, self.prev_filters, output)
        self.prev_filters = self.x_filters
        self.x_filters = output * 4 #stack for 4 branches

        return reduction
    
    def _make_layers(self, repeat_cell_num, reduction_num):

        layers = []
        for i in range(reduction_num):

            layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))
            self.filters *= 2
            layers.append(self._make_reduction(ReductionCell, self.filters))
        
        layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.stem(x)
        prev = None
        x, prev = self.cell_layers((x, prev))
        x = self.relu(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

        
        
def nasnet():

    #stem filters must be 44, it's a pytorch workaround, cant change to other number
    
    model =  NasNetA(4, 2, 44, 44)    
    #summary(model, input_data=(BAND, img_rows, img_cols), verbose=1)
    return model  

class FinetuneResnet(nn.Module):
    def __init__(self, modelx):
        super(FinetuneResnet,self).__init__()
        self.fc1 = nn.Conv2d(103, 200, 3, padding=1, bias=False)
        self.model = modelx
        self.fc2 = nn.Linear(16,9)
        self.dropout = nn.Dropout(0.3)#, modelx

    def forward(self, x):
        x = self.fc1(x)
        x = self.model(x)
        x = self.fc2(x)
        x = self.dropout(x)
      #  x = nn.functional.softmax((x), dim=1)

        return x


import torch
import numpy as np
import torch.utils.data as Data

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)

    return small_cubic_data

def get_indices(ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        nb_val = 0
        test[i] = indexes[nb_val:]

    test_indexes = []
    for i in range(m):
        test_indexes += test[i]
    np.random.shuffle(test_indexes)
    return test_indexes



print("Cross validation")
#kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
iteration = 1

total_indices = get_indices(gt)

gt_all = gt[total_indices] - 1


all_data = select_small_cubic(TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
print("all data shape", all_data.shape)



del padded_data, whole_data, gt, data_,data 

loss_function = nn.CrossEntropyLoss()



import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold


index_iter = 0
sfolder = StratifiedKFold(n_splits=5,random_state=None,shuffle=False)


for train, test in sfolder.split(all_data, gt_all):
    print('Train: %s | test: %s' % (train, test))
    print(train.shape)
    
    print("iteration number ", index_iter+1)
    print(index_iter)


    all_train = all_data[train]
    all_train_label = gt_all[train]
    all_test = all_data[test]
    all_test_label = gt_all[test]
    all_tensor_train = torch.from_numpy(all_train).type(torch.FloatTensor)#.unsqueeze(1)
    all_tensor_train_label = torch.from_numpy(all_train_label).type(torch.FloatTensor)
    all_tensor_test = torch.from_numpy(all_test).type(torch.FloatTensor)#.unsqueeze(1)
    all_tensor_test_label = torch.from_numpy(all_test_label).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(all_tensor_train, all_tensor_train_label)
    torch_dataset_test = Data.TensorDataset(all_tensor_test, all_tensor_test_label)
    del all_train, all_test, all_tensor_train, all_tensor_test, all_train_label, all_tensor_test_label,all_tensor_train_label, all_test_label
        
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                          torch_dataset_train, 
                          batch_size=16)#, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                          torch_dataset_test,
                          batch_size=16)# sampler=test_subsampler)
    
    
    net = nasnet()
    # for param in net.parameters():
    #     param.requires_grad = False   
    net.load_state_dict(torch.load('./yenimodeller/nasnet_9_0.999_IN_.pt'),strict=False)
    # net = FinetuneResnet(net)
    
    
    optimizer = optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0)
    time_1 = int(time.time())
    
    
    loss_list = [100]
    early_epoch = 0
    
    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    tic1 = time.time()
    for epoch in range(0, 60):
         train_acc_sum, n = 0.0, 0
         time_epoch = time.time()
         lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(
             optimizer, 15, eta_min=0.0, last_epoch=-1)
      
         train_acc_sum, n = 0.0, 0
         time_epoch = time.time()
       #  Set current loss value
         current_loss = 0.0
     
           # Iterate over the DataLoader for training data
         for i, data in enumerate(trainloader, 0):
               
             batch_count, train_l_sum = 0, 0
             
             # Get inputs
             
             inputs, targets = data
             inputs = inputs.permute(0, 3, 1, 2)
             inputs = inputs.to(device)
              #print("y 1 ",y.shape)
             targets = targets.to(device)
             
             # Zero the gradients
             optimizer.zero_grad()
             
             # Perform forward pass
             outputs = net(inputs)
             
             # Compute loss
             loss = loss_function(outputs, targets.long())
             
             # Perform backward pass
             loss.backward()
             
             # Perform optimization
             optimizer.step()
             train_l_sum += loss.cpu().item()
             train_acc_sum += (outputs.argmax(dim=1) == targets).sum().cpu().item()
             n += targets.shape[0]
             batch_count += 1
         lr_adjust.step()
         valida_acc, valida_loss = evaluate_accuracy(
             testloader, net, loss, device)
         loss_list.append(valida_loss)
     
         train_loss_list.append(train_l_sum)  # / batch_count)
         train_acc_list.append(train_acc_sum / n)
         valida_loss_list.append(valida_loss)
         valida_acc_list.append(valida_acc)
     
         print(
             'epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
               % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                 valida_loss, valida_acc, time.time() - time_epoch))
     
         PATH = "./net_DBA.pt"
         
         
         
    toc1 = time.time()         
    pred_test = []
    preddss = []
    gt_test = []
    tic2 = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
                  
            test_l_sum, test_num = 0, 0
            inputs, ttargets = data
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            ttargets = ttargets.to(device)
            net.eval() 
            outputs = net(inputs)
            pred_test.extend(np.array(net(inputs).cpu().argmax(axis=1)))
            
    
       
    toc2 = time.time()
    collections.Counter(pred_test)
    # gt_test = gt[test] - 1
     #print("targets", ttargets.shape)
     #print("pred test", pred_test.shape)
     
    gts = gt_all[test]
    
    
    overall_acc = metrics.accuracy_score(pred_test, gts)
    confusion_matrix = metrics.confusion_matrix(pred_test, gts)
    each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
    print(each_acc.shape)
    kappa = metrics.cohen_kappa_score(pred_test, gts)
    f1s = metrics.f1_score(gts, pred_test, average='macro')
    
    torch.save(net.state_dict(),
         'nasnet_'  + str(img_rows) + '_' + str(round(overall_acc, 3)) + '_' + Dataset +'_.pt')
    KAPPA.append(kappa)
    F1.append(f1s)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc
    index_iter = index_iter + 1

# # Map, Records
print("--------" + " Training Finished-----------")
record_output(
OA, AA, KAPPA, F1, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
'./report/' + 'nasnet_' + str(img_rows) + '_' + Dataset + '_split' +
str(VALIDATION_SPLIT) + '_lr' + str(lr) + PARAM_OPTIM + '.txt')



all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor)#.unsqueeze(1)
print(all_tensor_data.shape)
all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
print(all_tensor_data_label.shape)
torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)
print("all tensor label shape", all_tensor_data_label.shape)


all_iter = Data.DataLoader(
    dataset=torch_dataset_all,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=False, 
    num_workers=0, 
)

Utils.generate_png(
all_iter, net, gt_hsi, Dataset, device, total_indices,
'./classification_maps/' + 'nasnet_' + str(img_rows) + '_' + Dataset
+ '_split' + str(VALIDATION_SPLIT) + 'lr' + str(lr) + PARAM_OPTIM)