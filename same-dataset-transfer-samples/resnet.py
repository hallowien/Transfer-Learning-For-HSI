

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

PARAM_DATASET = "UP"
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


# # Model


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 in_shape=3,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            200,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def ResNet50(in_shape, num_classes):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        in_shape (tuple): Shape of input
        num_classes (tuple): No of classes
    """
    model = ResNet(
        Bottleneck, [3, 4, 6, 3],
        in_shape=in_shape[0],
        num_classes=num_classes)
    return model



class FinetuneResnet(nn.Module):
    def __init__(self, modelx):
        super(FinetuneResnet,self).__init__()
        self.fc1 = nn.Conv2d(103,200, 3, padding=1, bias=False)
        self.model = modelx
        self.fc2 = nn.Linear(16,9)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.model(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x



# # Create models and load state_dicts
# modelA = MyModelA()
# modelB = MyModelB()
# # Load state dicts
# modelA.load_state_dict(torch.load(PATH))
# modelB.load_state_dict(torch.load(PATH))

# model = MyEnsemble(modelA, modelB)
# x1, x2 = torch.randn(1, 10), torch.randn(1, 20)
# output = model(x1, x2)



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
    
    
    net =  ResNet50(in_shape=(BAND, img_rows, img_cols), num_classes=16).cuda()
    
    # for param in net.parameters():
    #       param.requires_grad = False   
         
    # net.load_state_dict(torch.load('./yenimodeller/v2/resnet_9_0.987_IN_.pt'),strict=False)
    # net = FinetuneResnet(net)
    
    summary(net)
    
    
    
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
         'resnet_'  + str(img_rows) + '_' + str(round(overall_acc, 3)) + '_' + Dataset +'_.pt')
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
'./report/' + 'resnet_' + str(img_rows) + '_' + Dataset + '_split' +
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
'./classification_maps/' + 'resnet_' + str(img_rows) + '_' + Dataset
+ '_split' + str(VALIDATION_SPLIT) + 'lr' + str(lr) + PARAM_OPTIM)