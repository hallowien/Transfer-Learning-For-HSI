

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
PATCH_SIZE = 2
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



from collections import namedtuple
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
#from .utils import load_state_dict_from_url
from typing import Callable, Any, Optional, Tuple, List


__all__ = ['Inception3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


def inception_v3(in_shape, num_classes) -> "Inception3":
    model = Inception3(
        in_shape=in_shape[0],
        num_classes=num_classes)
    #summary(model, input_data=(BAND, img_rows, img_cols), verbose=1)
    return model

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



class Inception3(nn.Module):

    def __init__(self,
                 inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
                 init_weights: Optional[bool] = None,
                 aux_logits: bool = False,
                 transform_input: bool = False,
                 num_classes=16,
                 in_shape=3,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        
        self.Conv2d_1a_3x3 = conv_block(103, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
    
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        
        x = nn.functional.interpolate(x, size=(75,75))
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
       # x=self.Upsample(x)
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x,aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        #x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):
    downsample = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
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
    # print(index)
    # print("row =",row)
    # print("col", col)
    # print("pad", pad_length)
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
      #  print("assign0 = ", assign_0)
        assign_1 = value % col + pad_length
      #  print("assign1 = ", assign_1)
        new_assign[counter] = [assign_0, assign_1] #matrixteki yerini bul.
    #print("\n" , new_assign)
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    #print("selecteds rows:", selected_rows)
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    #print( "selected patch", selected_patch)
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    print(data_assign[0])
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    print(small_cubic_data[0])

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



del padded_data, gt, data_,data 

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
    
    
    net = inception_v3(in_shape=(BAND, img_rows, img_cols), num_classes=16)
    
    # for param in net.parameters():
    #     param.requires_grad = False   
    # net.load_state_dict(torch.load('./yenimodeller/testli/incptionv3_9_0.994_IN_.pt'),strict=False)
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
    for epoch in range(0, 10):
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
         'inception_'  + str(img_rows) + '_' + str(round(overall_acc, 3)) + '_' + Dataset +'_.pt')
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
'./report/' + 'inception_' + str(img_rows) + '_' + Dataset + '_split' +
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
'./classification_maps/' + 'inception_' + str(img_rows) + '_' + Dataset
+ '_split' + str(VALIDATION_SPLIT) + 'lr' + str(lr) + PARAM_OPTIM)