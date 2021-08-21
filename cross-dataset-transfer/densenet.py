
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



import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
#from .utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 9,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()
       # self.Upsample=nn.Upsample(scale_factor=5, mode='nearest')
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(200, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, size=(32,32))
       # x=self.Upsample(x)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)

    
class Finetune(nn.Module):
    def __init__(self, modelx):
        super(Finetune,self).__init__()
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
    
    
    net = densenet121(num_classes=16)
    
    for param in net.parameters():
         param.requires_grad = False
    net.load_state_dict(torch.load('./yenimodeller/densenet_9_0.999_IN_.pt'),strict=False)
    net = Finetune(net)

    

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
         'mobilenet_'  + str(img_rows) + '_' + str(round(overall_acc, 3)) + '_' + Dataset +'_.pt')
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
'./report/' + 'mobilenet_' + str(img_rows) + '_' + Dataset + '_split' +
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
'./classification_maps/' + 'mobilenet_' + str(img_rows) + '_' + Dataset
+ '_split' + str(VALIDATION_SPLIT) + 'lr' + str(lr) + PARAM_OPTIM)
