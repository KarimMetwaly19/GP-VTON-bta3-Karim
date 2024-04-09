from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import cv2
import datetime
import functools
import time
import argparse
from torch import FloatTensor
from random import random
from PIL import ImageDraw
import cv2
import pycocotools.mask as maskUtils
import math
import json
from cv2 import sort
from importlib_metadata import files
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
# !/usr/bin/env python
import torch
import cupy
import math
import re
import torch.nn.parallel
from torchvision import models
from torch.nn.utils.spectral_norm import spectral_norm

##ResWalid++

##############################


#ResUnet++ 
# class ResidualConv(nn.Module):
#     def __init__(self, input_dim, output_dim, stride, padding):
#         super(ResidualConv, self).__init__()

#         self.conv_block = nn.Sequential(
#             nn.BatchNorm2d(input_dim),
#             nn.ReLU(),
#             nn.Conv2d(
#                 input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
#             ),
#             nn.BatchNorm2d(output_dim),
#             nn.ReLU(),
#             nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
#         )
#         self.conv_skip = nn.Sequential(
#             nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm2d(output_dim),
#         )

#     def forward(self, x):

#         return self.conv_block(x) + self.conv_skip(x)


# class Upsample(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel, stride):
#         super(Upsample, self).__init__()

#         self.upsample = nn.ConvTranspose2d(
#             input_dim, output_dim, kernel_size=kernel, stride=stride
#         )

#     def forward(self, x):
#         return self.upsample(x)


# class Squeeze_Excite_Block(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(Squeeze_Excite_Block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class ASPP(nn.Module):
#     def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
#         super(ASPP, self).__init__()

#         self.aspp_block1 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#         self.aspp_block2 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )
#         self.aspp_block3 = nn.Sequential(
#             nn.Conv2d(
#                 in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
#             ),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(out_dims),
#         )

#         self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
#         self._init_weights()

#     def forward(self, x):
#         x1 = self.aspp_block1(x)
#         x2 = self.aspp_block2(x)
#         x3 = self.aspp_block3(x)
#         out = torch.cat([x1, x2, x3], dim=1)
#         return self.output(out)

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class Upsample_(nn.Module):
#     def __init__(self, scale=2):
#         super(Upsample_, self).__init__()

#         self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

#     def forward(self, x):
#         return self.upsample(x)


# class AttentionBlock(nn.Module):
#     def __init__(self, input_encoder, input_decoder, output_dim):
#         super(AttentionBlock, self).__init__()

#         self.conv_encoder = nn.Sequential(
#             nn.BatchNorm2d(input_encoder),
#             nn.ReLU(),
#             nn.Conv2d(input_encoder, output_dim, 3, padding=1),
#             nn.MaxPool2d(2, 2),
#         )

#         self.conv_decoder = nn.Sequential(
#             nn.BatchNorm2d(input_decoder),
#             nn.ReLU(),
#             nn.Conv2d(input_decoder, output_dim, 3, padding=1),
#         )

#         self.conv_attn = nn.Sequential(
#             nn.BatchNorm2d(output_dim),
#             nn.ReLU(),
#             nn.Conv2d(output_dim, 1, 1),
#         )

#     def forward(self, x1, x2):
#         out = self.conv_encoder(x1) + self.conv_decoder(x2)
#         out = self.conv_attn(out)
#         return out * x2






# #EL ResUnet++ nafso
# #ResUnet++
# class ResUnetPlusPlus(nn.Module):
#     def __init__(self, channel, filters=[64, 128, 256, 512, 512]):
#         super(ResUnetPlusPlus, self).__init__()

#         self.input_layer = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters[0]),
#             nn.ReLU(),
#             nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
#         )
#         self.input_skip = nn.Sequential(
#             nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
#         )

#         self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

#         self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

#         self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

#         self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

#         self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

#         self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

#         self.aspp_bridge = ASPP(filters[3], filters[4])

#         self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
#         self.upsample1 = Upsample_(2)
#         self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

#         self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
#         self.upsample2 = Upsample_(2)
#         self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

#         self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
#         self.upsample3 = Upsample_(2)
#         self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

#         self.aspp_out = ASPP(filters[1], filters[0])

#       #  self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid()) # //the real one
#         #self.output_layer=nn.Conv2d(filters[0], 1, 1)
#        # self.output_layer = nn.Conv2d(filters[0], 3, kernel_size=1)  # Assuming 3 channels for RGB image


#         self.output_layer = nn.Sequential(
#             nn.Conv2d(filters[0], 3, 1),  # in,out,filtersize
#             nn.Identity(),  # Linear activation (identity function)
#         )

#     def forward(self, x):
#         x1 = self.input_layer(x) + self.input_skip(x)

#         x2 = self.squeeze_excite1(x1)
#         x2 = self.residual_conv1(x2)

#         x3 = self.squeeze_excite2(x2)
#         x3 = self.residual_conv2(x3)

#         x4 = self.squeeze_excite3(x3)
#         x4 = self.residual_conv3(x4)

#         x5 = self.aspp_bridge(x4)

#         x6 = self.attn1(x3, x5)
#         x6 = self.upsample1(x6)
#         x6 = torch.cat([x6, x3], dim=1)
#         x6 = self.up_residual_conv1(x6)

#         x7 = self.attn2(x2, x6)
#         x7 = self.upsample2(x7)
#         x7 = torch.cat([x7, x2], dim=1)
#         x7 = self.up_residual_conv2(x7)

#         x8 = self.attn3(x1, x7)
#         x8 = self.upsample3(x8)
#         x8 = torch.cat([x8, x1], dim=1)
#         x8 = self.up_residual_conv3(x8)

#         x9 = self.aspp_out(x8)
#         out = self.output_layer(x9)

#         return out


######################################### testtesttest


kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float fltValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / SIZE_3(input)) + 3*{{intStride}};
	  int intPaddedX = (intIndex % SIZE_3(input)) + 3*{{intStride}};
	  int intRearrange = ((SIZE_3(input) + 6*{{intStride}}) * intPaddedY) + intPaddedX;

	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = fltValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];

	  float *patch_data = (float *)patch_data_char;

	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = (blockIdx.x + 3) * {{intStride}};
	  int y1 = (blockIdx.y + 3) * {{intStride}};
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;

	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }

	  __syncthreads();

	  __shared__ float sum[32];

	  // Compute correlation
	  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;

	    int s2o = (top_channel % 7 - 3) * {{intStride}};
	    int s2p = (top_channel / 7 - 3) * {{intStride}};

	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;

	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;

	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }

	    __syncthreads();

	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  }
	}
'''

kernel_Correlation_updateGradFirst = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradFirst); // channels
	  int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 3*{{intStride}}; // w-pos
	  int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 3*{{intStride}}; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = {{intStride}} * round_off;

	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}}) / {{intStride}}
	  int ymin = (m - 3*{{intStride}} + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}}) / {{intStride}}

	  // Same here:
	  int xmax = (l - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor (l - 3*{{intStride}}) / {{intStride}}
	  int ymax = (m - 3*{{intStride}} + round_off_s1) / {{intStride}} - round_off; // floor (m - 3*{{intStride}}) / {{intStride}}

	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	    xmin = max(0,xmin);
	    xmax = min(SIZE_3(gradOutput)-1,xmax);

	    ymin = max(0,ymin);
	    ymax = min(SIZE_2(gradOutput)-1,ymax);

	    for (int p = -3; p <= 3; p++) {
	      for (int o = -3; o <= 3; o++) {
	        // Get rbot1 data:
	        int s2o = {{intStride}} * o;
	        int s2p = {{intStride}} * p;
	        int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradFirst);
	  const int bot0index = ((n * SIZE_2(gradFirst)) + (m-3*{{intStride}})) * SIZE_3(gradFirst) + (l-3*{{intStride}});
	  gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
	} }
'''

kernel_Correlation_updateGradSecond = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradSecond); // channels
	  int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 3*{{intStride}}; // w-pos
	  int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 3*{{intStride}}; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = {{intStride}} * round_off;

	  float sum = 0;
	  for (int p = -3; p <= 3; p++) {
	    for (int o = -3; o <= 3; o++) {
	      int s2o = {{intStride}} * o;
	      int s2p = {{intStride}} * p;

	      //Get X,Y ranges and clamp
	      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	      int xmin = (l - 3*{{intStride}} - s2o + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}} - s2o) / {{intStride}}
	      int ymin = (m - 3*{{intStride}} - s2p + round_off_s1 - 1) / {{intStride}} + 1 - round_off; // ceil (l - 3*{{intStride}} - s2o) / {{intStride}}

	      // Same here:
	      int xmax = (l - 3*{{intStride}} - s2o + round_off_s1) / {{intStride}} - round_off; // floor (l - 3*{{intStride}} - s2o) / {{intStride}}
	      int ymax = (m - 3*{{intStride}} - s2p + round_off_s1) / {{intStride}} - round_off; // floor (m - 3*{{intStride}} - s2p) / {{intStride}}

	      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	        xmin = max(0,xmin);
	        xmax = min(SIZE_3(gradOutput)-1,xmax);

	        ymin = max(0,ymin);
	        ymax = min(SIZE_2(gradOutput)-1,ymax);

	        // Get rbot0 data:
	        int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradSecond);
	  const int bot1index = ((n * SIZE_2(gradSecond)) + (m-3*{{intStride}})) * SIZE_3(gradSecond) + (l-3*{{intStride}});
	  gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
	} }
'''


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction].replace('{{intStride}}', str(objVariables['intStride']))

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel


# end

@cupy.memoize()
def cupy_launch(strFunction, strKernel):
    # Compile the raw kernel
    raw_kernel = cupy.RawKernel(strKernel, strFunction)
    return raw_kernel

# end

class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(self, first, second, intStride):
        rbot0 = first.new_zeros(
            [first.shape[0], first.shape[2] + (6 * intStride), first.shape[3] + (6 * intStride), first.shape[1]])
        rbot1 = first.new_zeros(
            [first.shape[0], first.shape[2] + (6 * intStride), first.shape[3] + (6 * intStride), first.shape[1]])

        self.save_for_backward(first, second, rbot0, rbot1)

        self.intStride = intStride

        assert (first.is_contiguous() == True)
        assert (second.is_contiguous() == True)

        output = first.new_zeros([first.shape[0], 49, int(math.ceil(first.shape[2] / intStride)),
                                  int(math.ceil(first.shape[3] / intStride))])

        if first.is_cuda == True:
            n = first.shape[2] * first.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'intStride': self.intStride,
                'input': first,
                'output': rbot0
            }))(
                grid=tuple([int((n + 16 - 1) / 16), first.shape[1], first.shape[0]]),
                block=tuple([16, 1, 1]),
                args=[n, first.data_ptr(), rbot0.data_ptr()]
            )

            n = second.shape[2] * second.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'intStride': self.intStride,
                'input': second,
                'output': rbot1
            }))(
                grid=tuple([int((n + 16 - 1) / 16), second.shape[1], second.shape[0]]),
                block=tuple([16, 1, 1]),
                args=[n, second.data_ptr(), rbot1.data_ptr()]
            )

            n = output.shape[1] * output.shape[2] * output.shape[3]
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
                'intStride': self.intStride,
                'rbot0': rbot0,
                'rbot1': rbot1,
                'top': output
            }))(
                grid=tuple([output.shape[3], output.shape[2], output.shape[0]]),
                block=tuple([32, 1, 1]),
                shared_mem=first.shape[1] * 4,
                args=[n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()]
            )

        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        return output

    # end

    @staticmethod
    def backward(self, gradOutput):
        first, second, rbot0, rbot1 = self.saved_tensors

        assert (gradOutput.is_contiguous() == True)

        gradFirst = first.new_zeros([first.shape[0], first.shape[1], first.shape[2], first.shape[3]]) if \
        self.needs_input_grad[0] == True else None
        gradSecond = first.new_zeros([first.shape[0], first.shape[1], first.shape[2], first.shape[3]]) if \
        self.needs_input_grad[1] == True else None

        if first.is_cuda == True:
            if gradFirst is not None:
                for intSample in range(first.shape[0]):
                    n = first.shape[1] * first.shape[2] * first.shape[3]
                    cupy_launch('kernel_Correlation_updateGradFirst',
                                cupy_kernel('kernel_Correlation_updateGradFirst', {
                                    'intStride': self.intStride,
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': gradFirst,
                                    'gradSecond': None
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(),
                              gradFirst.data_ptr(), None]
                    )
            # end
            # end

            if gradSecond is not None:
                for intSample in range(first.shape[0]):
                    n = first.shape[1] * first.shape[2] * first.shape[3]
                    cupy_launch('kernel_Correlation_updateGradSecond',
                                cupy_kernel('kernel_Correlation_updateGradSecond', {
                                    'intStride': self.intStride,
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': None,
                                    'gradSecond': gradSecond
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None,
                              gradSecond.data_ptr()]
                    )
            # end
        # end

        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradFirst, gradSecond, None


# end
# end

def FunctionCorrelation(tenFirst, tenSecond, intStride):
    return _FunctionCorrelation.apply(tenFirst, tenSecond, intStride)


# end

class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    # end

    def forward(self, tenFirst, tenSecond, intStride):
        return _FunctionCorrelation.apply(tenFirst, tenSecond, intStride)


# end
# end


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x) + x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)



def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid(
        [torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
                 for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
                 for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)


class AFlowNet_Vitonhd_lrarms(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet_Vitonhd_lrarms, self).__init__()
        self.netLeftMain = []
        self.netTorsoMain = []
        self.netRightMain = []

        self.netLeftRefine = []
        self.netTorsoRefine = []
        self.netRightRefine = []

        self.netAttentionRefine = []
        self.netPartFusion = []
        self.netSeg = []

        for i in range(num_pyramid):
            netLeftMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netTorsoMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRightMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )

            netRefine_left_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_torso_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_right_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )

            netAttentionRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=4 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=3,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

            netSeg_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=fpn_dim * 2, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=7,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

            partFusion_layer = torch.nn.Sequential(
                nn.Conv2d(fpn_dim * 3, fpn_dim, kernel_size=1),
                ResBlock(fpn_dim)
            )

            self.netLeftMain.append(netLeftMain_layer)
            self.netTorsoMain.append(netTorsoMain_layer)
            self.netRightMain.append(netRightMain_layer)

            self.netLeftRefine.append(netRefine_left_layer)
            self.netTorsoRefine.append(netRefine_torso_layer)
            self.netRightRefine.append(netRefine_right_layer)

            self.netAttentionRefine.append(netAttentionRefine_layer)
            self.netPartFusion.append(partFusion_layer)
            self.netSeg.append(netSeg_layer)

        self.netLeftMain = nn.ModuleList(self.netLeftMain)
        self.netTorsoMain = nn.ModuleList(self.netTorsoMain)
        self.netRightMain = nn.ModuleList(self.netRightMain)

        self.netLeftRefine = nn.ModuleList(self.netLeftRefine)
        self.netTorsoRefine = nn.ModuleList(self.netTorsoRefine)
        self.netRightRefine = nn.ModuleList(self.netRightRefine)

        self.netAttentionRefine = nn.ModuleList(self.netAttentionRefine)
        self.netPartFusion = nn.ModuleList(self.netPartFusion)
        self.netSeg = nn.ModuleList(self.netSeg)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, x_edge, x_full, x_edge_full, x_warps, x_conds, preserve_mask, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        x_full_all = []
        x_edge_full_all = []
        attention_all = []
        seg_list = []
        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        for i in range(len(x_warps)):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_cond_concate = torch.cat([x_cond, x_cond, x_cond], 0)
            x_warp_concate = torch.cat([x_warp, x_warp, x_warp], 0)

            if last_flow is not None and warp_feature:
                x_warp_after = F.grid_sample(x_warp_concate, last_flow.detach().permute(0, 2, 3, 1),
                                             mode='bilinear', padding_mode='border')
            else:
                x_warp_after = x_warp_concate

            tenCorrelation = F.leaky_relu(input=FunctionCorrelation(
                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), negative_slope=0.1, inplace=False)

            bz = x_cond.size(0)

            left_tenCorrelation = tenCorrelation[0:bz]
            torso_tenCorrelation = tenCorrelation[bz:2 * bz]
            right_tenCorrelation = tenCorrelation[2 * bz:]

            left_flow = self.netLeftMain[i](left_tenCorrelation)
            torso_flow = self.netTorsoMain[i](torso_tenCorrelation)
            right_flow = self.netRightMain[i](right_tenCorrelation)

            flow = torch.cat([left_flow, torso_flow, right_flow], 0)

            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp_concate = F.grid_sample(x_warp_concate, flow.permute(
                0, 2, 3, 1), mode='bilinear', padding_mode='border')

            left_concat = torch.cat([x_warp_concate[0:bz], x_cond_concate[0:bz]], 1)
            torso_concat = torch.cat([x_warp_concate[bz:2 * bz], x_cond_concate[bz:2 * bz]], 1)
            right_concat = torch.cat([x_warp_concate[2 * bz:], x_cond_concate[2 * bz:]], 1)

            x_attention = torch.cat([x_warp_concate[0:bz], x_warp_concate[bz:2 * bz], x_warp_concate[2 * bz:], x_cond],
                                    1)
            fused_attention = self.netAttentionRefine[i](x_attention)
            fused_attention = self.softmax(fused_attention)

            left_flow = self.netLeftRefine[i](left_concat)
            torso_flow = self.netTorsoRefine[i](torso_concat)
            right_flow = self.netRightRefine[i](right_concat)

            flow = torch.cat([left_flow, torso_flow, right_flow], 0)
            delta_list.append(flow)
            flow = apply_offset(flow)
            flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

            fused_flow = flow[0:bz] * fused_attention[:, 0:1, ...] + \
                         flow[bz:2 * bz] * fused_attention[:, 1:2, ...] + \
                         flow[2 * bz:] * fused_attention[:, 2:3, ...]
            last_fused_flow = F.interpolate(fused_flow, scale_factor=2, mode='bilinear')

            fused_attention = F.interpolate(fused_attention, scale_factor=2, mode='bilinear')
            attention_all.append(fused_attention)

            cur_x_full = F.interpolate(x_full, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear')
            cur_x_full_warp = F.grid_sample(cur_x_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear',
                                            padding_mode='zeros')
            x_full_all.append(cur_x_full_warp)
            cur_x_edge_full = F.interpolate(x_edge_full, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear')
            cur_x_edge_full_warp = F.grid_sample(cur_x_edge_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear',
                                                 padding_mode='zeros')
            x_edge_full_all.append(cur_x_edge_full_warp)

            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            cur_x = F.interpolate(x, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear')
            cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_all.append(cur_x_warp)
            cur_x_edge = F.interpolate(x_edge, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear')
            cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear',
                                            padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)

            flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y, self.weight)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

            # predict seg
            cur_preserve_mask = F.interpolate(preserve_mask, scale_factor=0.5 ** (len(x_warps) - 1 - i),
                                              mode='bilinear')
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_warp = torch.cat([x_warp, x_warp, x_warp], 0)
            x_warp = F.interpolate(x_warp, scale_factor=2, mode='bilinear')
            x_cond = F.interpolate(x_cond, scale_factor=2, mode='bilinear')

            x_warp = F.grid_sample(x_warp, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
            x_warp_left = x_warp[0:bz]
            x_warp_torso = x_warp[bz:2 * bz]
            x_warp_right = x_warp[2 * bz:]

            x_edge_left = cur_x_warp_edge[0:bz]
            x_edge_torso = cur_x_warp_edge[bz:2 * bz]
            x_edge_right = cur_x_warp_edge[2 * bz:]

            x_warp_left = x_warp_left * x_edge_left * (1 - cur_preserve_mask)
            x_warp_torso = x_warp_torso * x_edge_torso * (1 - cur_preserve_mask)
            x_warp_right = x_warp_right * x_edge_right * (1 - cur_preserve_mask)

            x_warp = torch.cat([x_warp_left, x_warp_torso, x_warp_right], 1)
            x_warp = self.netPartFusion[i](x_warp)

            concate = torch.cat([x_warp, x_cond], 1)
            seg = self.netSeg[i](concate)
            seg_list.append(seg)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, \
            x_edge_full_all, attention_all, seg_list

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features


class RefinePyramid(nn.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + \
                    F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


class AFWM_Vitonhd_lrarms(nn.Module):
    def __init__(self, opt, input_nc, clothes_input_nc=3):
        super(AFWM_Vitonhd_lrarms, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        # num_filters = [64,128,256,512,512]
        fpn_dim = 256
        self.image_features = FeatureEncoder(clothes_input_nc + 1, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        self.cond_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)

        self.aflow_net = AFlowNet_Vitonhd_lrarms(len(num_filters))
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr * 0.2

    def forward(self, cond_input, image_input, image_edge, image_label_input, image_input_left, image_input_torso, \
                image_input_right, image_edge_left, image_edge_torso, image_edge_right, preserve_mask):
        image_input_concat = torch.cat([image_input, image_label_input], 1)

        image_pyramids = self.image_FPN(self.image_features(image_input_concat))
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))  # maybe use nn.Sequential

        image_concat = torch.cat([image_input_left, image_input_torso, image_input_right], 0)
        image_edge_concat = torch.cat([image_edge_left, image_edge_torso, image_edge_right], 0)

        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = self.aflow_net(image_concat, image_edge_concat, image_input,
                                                                                  image_edge, image_pyramids,
                                                                                  cond_pyramids, preserve_mask)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))



def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        assert gan_mode in ['lsgan', 'vanilla', 'wgangp']
        if gan_mode in ['wgangp']:
            self.loss = None
        self.gan_mode = gan_mode

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, prediction, target_is_real, add_gradient=False):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()  # + 0.001*(prediction**2).mean()
                if add_gradient:
                    loss = -prediction.mean() + 0.001*(prediction**2).mean()
            else:
                loss = prediction.mean()
        return loss

class SpectralDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(SpectralDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf *
                              nf_mult, kernel_size=kw, stride=2, padding=padw)),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf *
                          nf_mult, kernel_size=kw, stride=1, padding=padw)),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult,
                                   1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)
        self.old_lr = opt.lr_D

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

    def update_learning_rate(self, optimizer, opt):
        lrd = opt.lr_D / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.local_rank == 0:
            print('update learning rate for D model: %f -> %f' %
                  (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr


def load_checkpoint_parallel(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(opt.local_rank))
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)
def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
        self.old_lr = opt.lr
        self.old_lr_gmm = 0.1*opt.lr

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


#######ResWalidbardo
class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class build_resunetplusplus(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = Stem_Block(36, 64, stride=1)
        self.c2 = ResNet_Block(64, 128, stride=2)
     #k   self.c3 = ResNet_Block(128, 256, stride=2)
        #self.c4 = ResNet_Block(256, 512, stride=2)

       #k self.b1 = ASPP(256, 512)
	self.b1 = ASPP(128, 256)    

        #self.d1 = Decoder_Block([256, 512], 512)
     #k   self.d2 = Decoder_Block([128, 512], 256)
        self.d3 = Decoder_Block([64, 256], 128)
        self.output = nn.Conv2d(128, 4, kernel_size=1)


    # def forward(self, inputs):
    #     c1 = self.c1(inputs)
    #     c2 = self.c2(c1)
    #     c3 = self.c3(c2)
    #     #c4 = self.c4(c3)

    #     b1 = self.b1(c3)

    #     #d1 = self.d1(c3, b1)
    #     d2 = self.d2(c2, b1)
    #     d3 = self.d3(c1, d2)

    #     output = self.output(d3)

    #     return output
#habbooda version
    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
     #k   c3 = self.c3(c2)
        #c4 = self.c4(c3)

        b1 = self.b1(c2)

        #d1 = self.d1(c3, b1)
     #k   d2 = self.d2(c2, b1)
        d3 = self.d3(c1, b1)

        output = self.output(d3)

        return output

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    # if normalize:
    #    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # else:
    #    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = (image_numpy + 1) / 2.0
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]

    return image_numpy

# Converts a one-hot tensor into a colorful label map


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    #label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0,  0,  0), (0,  0,  0), (0,  0,  0), (0,  0,  0), (0,  0,  0), (111, 74,  0), (81,  0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230,
                                                                           150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153,
                                                                            153, 153), (153, 153, 153), (250, 170, 30), (220, 220,  0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220,
                                                                           20, 60), (255,  0,  0), (0,  0, 142), (0,  0, 70),
                         (0, 60, 100), (0,  0, 90), (0,  0, 110), (0, 80, 100), (0,  0, 230), (119, 11, 32), (0,  0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def save_html(path_result):
    result_dir = path_result
    filenames = sorted(os.listdir(result_dir))
    fs = []
    for filename in filenames:
        # fs.append(os.path.join(result_dir.split('/')[-1], filename))
        fs.append(filename)

    # first_dirs = os.listdir(result_dir)
    # for first_dir in first_dirs:
    #     if os.path.isdir(os.path.join(result_dir, first_dir)) == False:
    #         continue
    #     sub_dirs = os.listdir(os.path.join(result_dir, first_dir))
    #     for sub_dir in sub_dirs:
    #         filenames = os.listdir(os.path.join(
    #             result_dir, first_dir, sub_dir))
    #         for filename in filenames:
    #             fs.append(os.path.join(first_dir, sub_dir, filename))

    title = 'result'
    height = 'auto'
    width = 'auto'
    outpath = os.path.join(result_dir, 'vis.html')

    str_ = "<p>%s</p>" % title
    for i in range(len(fs)):
        path = fs[i]
        str_ += "<img src='" + path + "' alt='" + path
        str_ += "'  height=%s width=%s padding: 50px 10px style='border:100px '>" % (
            height, width)
        str_ += "<br>"

    with open(outpath, 'w') as f:
        f.write(str_)


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='flow',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/kaggle/working', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32],
                                 help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=True, help='toggles verbose')
        self.parser.add_argument('--nproc_per_node', type=int, default=1, help='nproc_per_node is the number of gpus')
        self.parser.add_argument('--master_port', type=int, default=7129, help='the master port number')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=14, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='/kaggle/input/gp-vton-dataset/VITON-HD/VITON-HD')
        self.parser.add_argument('--resize_or_crop', type=str, default='none',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--warproot', type=str, default='/kaggle/input/warping-results')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=4,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the outmost local enhancer')
        self.parser.add_argument('--tv_weight', type=float, default=0.1, help='weight for TV loss')

        self.parser.add_argument('--image_pairs_txt', type=str, default='test_pairs_unpaired_1018.txt')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=[])
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        # expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        # util.mkdirs(expr_dir)
        # if save and not self.opt.continue_train:
        #     file_name = os.path.join(expr_dir, 'opt.txt')
        #     with open(file_name, 'wt') as opt_file:
        #         opt_file.write('------------ Options -------------\n')
        #         for k, v in sorted(args.items()):
        #             opt_file.write('%s: %s\n' % (str(k), str(v)))
        #         opt_file.write('-------------- End ----------------\n')
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument(
            '--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
        self.parser.add_argument('--local_rank', type=int, default=0)

        self.parser.add_argument('--write_loss_frep', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int,
                                 default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument(
            '--niter', type=int, default=1, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=1,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument(
            '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
            '--lr', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument(
            '--lr_D', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument('--pretrain_checkpoint_D', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_warp_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_gen_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_warp_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_gen_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')

        self.parser.add_argument('--CPM_checkpoint', type=str)
        self.parser.add_argument('--CPM_D_checkpoint', type=str)

        self.parser.add_argument('--write_loss_frep_eval', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_freq_eval', type=int, default=100,
                                 help='frequency of showing training results on screen')

        self.parser.add_argument('--add_mask_tvloss', action='store_true',
                                 help='if specified, use employ tv loss for the predicted composited mask')

        # for discriminators
        self.parser.add_argument(
            '--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument(
            '--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument(
            '--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument(
            '--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true',
                                 help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true',
                                 help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0,
                                 help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--debug_test', action='store_true')
        self.parser.add_argument(
            '--image_test_pairs_txt', type=str, default='')
        self.parser.add_argument(
            '--image_pairs_txt_eval', type=str, default='')
        self.parser.add_argument('--use_preserve_mask_refine', action='store_true',
                                 help='if specified, use preserve mask to refine to the warp clothes')

        self.parser.add_argument('--repeat_num', type=int, default=6)
        self.parser.add_argument('--loss_ce', type=float, default=1)
        self.parser.add_argument('--loss_gan', type=float, default=1)

        self.parser.add_argument('--debug_train', action='store_true')
        self.parser.add_argument('--test_flip', action='store_true')

        self.parser.add_argument(
            '--first_order_smooth_weight', type=float, default=0.01)
        self.parser.add_argument(
            '--squaretv_weight', type=float, default=1)

        self.parser.add_argument('--mask_epoch', type=int, default=70)
        self.parser.add_argument('--no_dynamic_mask', action='store_true')

        self.parser.add_argument('--resolution', type=int, default=512)
        self.parser.add_argument('--dataset', type=str, default='vitonhd')

        self.isTrain = True




class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    # flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform_resize(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
    osize = [256, 192]
    transform_list.append(transforms.Resize(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [256, 192]
        transform_list.append(transforms.Resize(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img




class AlignedDataset(BaseDataset):
    def initialize(self, opt, mode='train'):
        self.opt = opt
        self.root = opt.dataroot
        self.warproot = opt.warproot
        self.resolution = opt.resolution

        if self.resolution == 512:
            self.fine_height = 512
            self.fine_width = 384
            self.radius = 8
        else:
            self.fine_height = 1024
            self.fine_width = 768
            self.radius = 16

        pair_txt_path = os.path.join(self.root, opt.image_pairs_txt)
        if mode == 'train' and 'train' in opt.image_pairs_txt:
            self.mode = 'train'
        else:
            self.mode = 'test'
        with open(pair_txt_path, 'r') as f:
            lines = f.readlines()

        self.P_paths = []
        self.C_paths = []
        self.C_types = []
        for line in lines:
            p_name, c_name, c_type = line.strip().split()
            P_path = os.path.join(self.root, 'test', 'image', p_name)
            C_path = os.path.join(self.root, 'test', 'cloth', c_name)
            if self.resolution == 1024:
                P_path = P_path.replace('.png', '.jpg')
                C_path = C_path.replace('.png', '.jpg')
            self.P_paths.append(P_path)
            self.C_paths.append(C_path)
            self.C_types.append(c_type)

        ratio_dict = None
        if self.mode == 'train':
            ratio_dict = {}
            person_clothes_ratio_txt = os.path.join(self.root, 'person_clothes_ratio_train.txt')
            with open(person_clothes_ratio_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                c_name, ratio = line.strip().split()
                ratio = float(ratio)
                ratio_dict[c_name] = ratio
        self.ratio_dict = ratio_dict
        self.dataset_size = len(self.P_paths)

    ############### get palm mask ################
    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[..., np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b - d) / 4, b + (c - a) / 4
        x2, y2 = a - (b - d) / 4, b - (c - a) / 4

        x3, y3 = c + (b - d) / 4, d + (c - a) / 4
        x4, y4 = c - (b - d) / 4, d - (c - a) / 4

        kps = [x1, y1, x2, y2]

        v0_x, v0_y = c - a, d - b
        v1_x, v1_y = x3 - x1, y3 - y1
        v2_x, v2_y = x4 - x1, y4 - y1

        cos1 = (v0_x * v1_x + v0_y * v1_y) / \
               (math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v1_x * v1_x + v1_y * v1_y))
        cos2 = (v0_x * v2_x + v0_y * v2_y) / \
               (math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v2_x * v2_x + v2_y * v2_y))

        if cos1 < cos2:
            kps.extend([x3, y3, x4, y4])
        else:
            kps.extend([x4, y4, x3, y3])

        kps = np.array(kps).reshape(1, -1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints, h, w):
        # shoulder, elbow, wrist
        s_x, s_y, s_c = hand_keypoints[0]
        e_x, e_y, e_c = hand_keypoints[1]
        w_x, w_y, w_c = hand_keypoints[2]

        up_mask = np.ones((h, w, 1), dtype=np.float32)
        bottom_mask = np.ones((h, w, 1), dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((50, 50), np.uint8)
            else:
                kernel = np.ones((100, 100), np.uint8)
            up_mask = cv2.dilate(up_mask, kernel, iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[..., np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((30, 30), np.uint8)
            else:
                kernel = np.ones((60, 60), np.uint8)
            bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[..., np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask + hand_bottom_mask)
                             == 2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, parsing, keypoints):
        h, w = parsing.shape[0:2]

        left_hand_keypoints = keypoints[[5, 6, 7], :].copy()
        right_hand_keypoints = keypoints[[2, 3, 4], :].copy()

        left_hand_up_mask, left_hand_bottom_mask = self.get_hand_mask(
            left_hand_keypoints, h, w)
        right_hand_up_mask, right_hand_bottom_mask = self.get_hand_mask(
            right_hand_keypoints, h, w)

        # mask refined by parsing
        left_hand_mask = (parsing == 15).astype(np.float32)
        right_hand_mask = (parsing == 16).astype(np.float32)

        left_palm_mask = self.get_palm_mask(
            left_hand_mask, left_hand_up_mask, left_hand_bottom_mask)
        right_palm_mask = self.get_palm_mask(
            right_hand_mask, right_hand_up_mask, right_hand_bottom_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############### get palm mask ################

    def __getitem__(self, index):
        C_type = self.C_types[index]

        # person image
        P_path = self.P_paths[index]
        P = Image.open(P_path).convert('RGB')
        P_np = np.array(P)
        params = get_params(self.opt, P.size)
        transform_for_rgb = get_transform(self.opt, params)
        P_tensor = transform_for_rgb(P)

        # person 2d pose
        pose_path = P_path.replace('/image/', '/openpose_json/')[:-4] + '_keypoints.json'
        with open(pose_path, 'r') as f:
            datas = json.load(f)
        pose_data = np.array(datas['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = transform_for_rgb(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        Pose_tensor = pose_map

        # person 3d pose
        densepose_path = P_path.replace('/image/', '/dense/')[:-4] + '.png'
        dense_mask = Image.open(densepose_path).convert('L')
        transform_for_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        dense_mask_tensor = transform_for_mask(dense_mask) * 255.0
        dense_mask_tensor = dense_mask_tensor[0:1, ...]

        # person parsing
        parsing_path = P_path.replace('/image/', '/parse/')[:-4] + '.png'
        parsing = Image.open(parsing_path).convert('L')
        parsing_tensor = transform_for_mask(parsing) * 255.0

        parsing_np = (parsing_tensor.numpy().transpose(1, 2, 0)[..., 0:1]).astype(np.uint8)
        palm_mask_np = self.get_palm(parsing_np, pose_data)

        person_clothes_left_sleeve_mask_np = (parsing_np == 21).astype(int) + \
                                             (parsing_np == 24).astype(int)
        person_clothes_torso_mask_np = (parsing_np == 5).astype(int) + \
                                       (parsing_np == 6).astype(int)
        person_clothes_right_sleeve_mask_np = (parsing_np == 22).astype(int) + \
                                              (parsing_np == 25).astype(int)
        person_clothes_mask_np = person_clothes_left_sleeve_mask_np + \
                                 person_clothes_torso_mask_np + \
                                 person_clothes_right_sleeve_mask_np
        left_arm_mask_np = (parsing_np == 15).astype(int)
        right_arm_mask_np = (parsing_np == 16).astype(int)
        hand_mask_np = (parsing_np == 15).astype(int) + (parsing_np == 16).astype(int)
        neck_mask_np = (parsing_np == 11).astype(int)

        person_clothes_left_sleeve_mask_tensor = torch.tensor(
            person_clothes_left_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_torso_mask_tensor = torch.tensor(person_clothes_torso_mask_np.transpose(2, 0, 1)).float()
        person_clothes_right_sleeve_mask_tensor = torch.tensor(
            person_clothes_right_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_mask_tensor = torch.tensor(person_clothes_mask_np.transpose(2, 0, 1)).float()
        left_arm_mask_tensor = torch.tensor(left_arm_mask_np.transpose(2, 0, 1)).float()
        right_arm_mask_tensor = torch.tensor(right_arm_mask_np.transpose(2, 0, 1)).float()
        neck_mask_tensor = torch.tensor(neck_mask_np.transpose(2, 0, 1)).float()

        seg_gt_tensor = person_clothes_left_sleeve_mask_tensor * 1 + person_clothes_torso_mask_tensor * 2 + \
                        person_clothes_right_sleeve_mask_tensor * 3 + left_arm_mask_tensor * 4 + \
                        right_arm_mask_tensor * 5 + neck_mask_tensor * 6
        background_mask_tensor = 1 - (person_clothes_left_sleeve_mask_tensor + person_clothes_torso_mask_tensor + person_clothes_right_sleeve_mask_tensor + left_arm_mask_tensor + right_arm_mask_tensor + neck_mask_tensor)
        seg_gt_onehot_tensor = torch.cat([background_mask_tensor, person_clothes_left_sleeve_mask_tensor, \
                                          person_clothes_torso_mask_tensor, person_clothes_right_sleeve_mask_tensor, \
                                          left_arm_mask_tensor, right_arm_mask_tensor, neck_mask_tensor], 0)

        ### preserve region mask
        if self.opt.no_dynamic_mask or self.ratio_dict is None:
            preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                  [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26, 27,
                                                   28]])
            preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
        else:
            pc_ratio = self.ratio_dict[self.C_paths[index].split('/')[-1][:-4] + '.png']
            if pc_ratio < 0.9:
                preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                      [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26, 27,
                                                       28]])
                preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
            elif pc_ratio < 0.95:
                if random.random() < 0.5:
                    preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                          [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26,
                                                           27, 28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
                else:
                    preserve_mask_for_loss_np = np.array(
                        [(parsing_np == index).astype(int) for index in [1, 2, 3, 4, 7, 12, 14, 23, 26, 27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
            else:
                if random.random() < 0.1:
                    preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                          [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26,
                                                           27, 28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
                else:
                    preserve_mask_for_loss_np = np.array(
                        [(parsing_np == index).astype(int) for index in [1, 2, 3, 4, 7, 12, 14, 23, 26, 27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)

        preserve_mask_np = np.array([(parsing_np == index).astype(int) for index in
                                     [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26, 27, 28]])
        preserve_mask_np = np.sum(preserve_mask_np, axis=0)

        preserve_mask1_np = preserve_mask_for_loss_np + palm_mask_np
        preserve_mask2_np = preserve_mask_for_loss_np + hand_mask_np
        preserve_mask3_np = preserve_mask_np + palm_mask_np

        preserve_mask1_tensor = torch.tensor(preserve_mask1_np.transpose(2, 0, 1)).float()
        preserve_mask2_tensor = torch.tensor(preserve_mask2_np.transpose(2, 0, 1)).float()
        preserve_mask3_tensor = torch.tensor(preserve_mask3_np.transpose(2, 0, 1)).float()

        ### clothes
        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_for_rgb(C)

        CM_path = C_path.replace('/cloth/', '/cloth_mask/')[:-4] + '.png'
        CM = Image.open(CM_path).convert('L')
        CM_tensor = transform_for_mask(CM)

        cloth_parsing_path = C_path.replace('/cloth/', '/cloth_parse/')[:-4] + '.png'
        cloth_parsing = Image.open(cloth_parsing_path).convert('L')
        cloth_parsing_tensor = transform_for_mask(cloth_parsing) * 255.0
        cloth_parsing_tensor = cloth_parsing_tensor[0:1, ...]

        cloth_parsing_np = (cloth_parsing_tensor.numpy().transpose(1, 2, 0)).astype(int)
        flat_cloth_left_mask_np = (cloth_parsing_np == 21).astype(int)
        flat_cloth_middle_mask_np = (cloth_parsing_np == 5).astype(int) + \
                                    (cloth_parsing_np == 24).astype(int) + \
                                    (cloth_parsing_np == 13).astype(int)
        flat_cloth_right_mask_np = (cloth_parsing_np == 22).astype(int)
        flat_cloth_label_np = flat_cloth_left_mask_np * 1 + flat_cloth_middle_mask_np * 2 + flat_cloth_right_mask_np * 3
        flat_cloth_label_np = flat_cloth_label_np / 3

        flat_cloth_left_mask_tensor = torch.tensor(flat_cloth_left_mask_np.transpose(2, 0, 1)).float()
        flat_cloth_middle_mask_tensor = torch.tensor(flat_cloth_middle_mask_np.transpose(2, 0, 1)).float()
        flat_cloth_right_mask_tensor = torch.tensor(flat_cloth_right_mask_np.transpose(2, 0, 1)).float()

        flat_cloth_label_tensor = torch.tensor(flat_cloth_label_np.transpose(2, 0, 1)).float()

        WC_tensor = None
        WE_tensor = None
        AMC_tensor = None
        ANL_tensor = None
        if self.warproot:
            ### skin color
            face_mask_np = (parsing_np == 14).astype(np.uint8)
            skin_mask_np = (face_mask_np + hand_mask_np + neck_mask_np).astype(np.uint8)
            skin = skin_mask_np * P_np
            skin_r = skin[..., 0].reshape((-1))
            skin_g = skin[..., 1].reshape((-1))
            skin_b = skin[..., 2].reshape((-1))
            skin_r_valid_index = np.where(skin_r > 0)[0]
            skin_g_valid_index = np.where(skin_g > 0)[0]
            skin_b_valid_index = np.where(skin_b > 0)[0]

            skin_r_median = np.median(skin_r[skin_r_valid_index])
            skin_g_median = np.median(skin_g[skin_g_valid_index])
            skin_b_median = np.median(skin_b[skin_b_valid_index])

            arms_r = np.ones_like(parsing_np[..., 0:1]) * skin_r_median
            arms_g = np.ones_like(parsing_np[..., 0:1]) * skin_g_median
            arms_b = np.ones_like(parsing_np[..., 0:1]) * skin_b_median
            arms_color = np.concatenate([arms_r, arms_g, arms_b], 2).transpose(2, 0, 1)
            AMC_tensor = torch.FloatTensor(arms_color)
            AMC_tensor = AMC_tensor / 127.5 - 1.0

            # warped clothes
            warped_name = C_type + '___' + P_path.split('/')[-1] + '___' + C_path.split('/')[-1][:-4] + '.png'
            warped_path = os.path.join(self.warproot, warped_name)
            warped_result = Image.open(warped_path).convert('RGB')
            warped_result_np = np.array(warped_result)

            if self.resolution == 512:
                w = 384
            else:
                w = 768
            warped_cloth_np = warped_result_np[:, -2 * w:-w, :]
            warped_parse_np = warped_result_np[:, -w:, :]

            warped_cloth = Image.fromarray(warped_cloth_np).convert('RGB')
            WC_tensor = transform_for_rgb(warped_cloth)

            warped_edge_np = (warped_parse_np == 1).astype(np.uint8) + \
                             (warped_parse_np == 2).astype(np.uint8) + \
                             (warped_parse_np == 3).astype(np.uint8)
            warped_edge = Image.fromarray(warped_edge_np).convert('L')
            WE_tensor = transform_for_mask(warped_edge) * 255.0
            WE_tensor = WE_tensor[0:1, ...]

            arms_neck_label = (warped_parse_np == 4).astype(np.uint8) * 1 + \
                              (warped_parse_np == 5).astype(np.uint8) * 2 + \
                              (warped_parse_np == 6).astype(np.uint8) * 3

            arms_neck_label = Image.fromarray(arms_neck_label).convert('L')
            ANL_tensor = transform_for_mask(arms_neck_label) * 255.0 / 3.0
            ANL_tensor = ANL_tensor[0:1, ...]

        input_dict = {
            'image': P_tensor, 'pose': Pose_tensor, 'dense': dense_mask_tensor,
            'seg_gt': seg_gt_tensor, 'seg_gt_onehot': seg_gt_onehot_tensor,
            'person_clothes_mask': person_clothes_mask_tensor,
            'person_clothes_left_mask': person_clothes_left_sleeve_mask_tensor,
            'person_clothes_middle_mask': person_clothes_torso_mask_tensor,
            'person_clothes_right_mask': person_clothes_right_sleeve_mask_tensor,
            'preserve_mask': preserve_mask1_tensor, 'preserve_mask2': preserve_mask2_tensor,
            'preserve_mask3': preserve_mask3_tensor,
            'color': C_tensor, 'edge': CM_tensor,
            'flat_clothes_left_mask': flat_cloth_left_mask_tensor,
            'flat_clothes_middle_mask': flat_cloth_middle_mask_tensor,
            'flat_clothes_right_mask': flat_cloth_right_mask_tensor,
            'flat_clothes_label': flat_cloth_label_tensor,
            'c_type': C_type,
            'color_path': C_path,
            'img_path': P_path,
        }
        if WC_tensor is not None:
            input_dict['warped_cloth'] = WC_tensor
            input_dict['warped_edge'] = WE_tensor
            input_dict['arms_color'] = AMC_tensor
            input_dict['arms_neck_lable'] = ANL_tensor

        return input_dict

    def __len__(self):
        if self.mode == 'train':
            return (len(self.P_paths) // (self.opt.batchSize * self.opt.num_gpus) *
                    (self.opt.batchSize * self.opt.num_gpus))
        else:
            return len(self.P_paths)

    def name(self):
        return 'AlignedDataset'




def CreateDataset(opt):
    if opt.dataset == 'vitonhd':
        dataset = AlignedDataset()
        dataset.initialize(opt)
    elif opt.dataset == 'dresscode':
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='train', stage='gen')
    return dataset


def train_tryon():
    run_path = 'runs/' + opt.name
    sample_path = 'sample/' + opt.name
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group('nccl', init_method='env://')
    device = torch.device(f'cuda:{opt.local_rank}')

    train_data = CreateDataset(opt)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                              num_workers=4, pin_memory=True, sampler=train_sampler)
    dataset_size = len(train_loader)

    gen_model=build_resunetplusplus()
    #gen_model=ResUnetPlusPlus(36) 
   # gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    gen_model.train()
    gen_model.cuda()
    if opt.PBAFN_gen_checkpoint is not None:
        load_checkpoint_parallel(gen_model, opt.PBAFN_gen_checkpoint)
    gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)
    if opt.isTrain and len(opt.gpu_ids):
        model_gen = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])

    params_gen = [p for p in model_gen.parameters()]
    optimizer_gen = torch.optim.Adam(params_gen, lr=opt.lr, betas=(opt.beta1, 0.999))

    discriminator = SpectralDiscriminator(opt, input_nc=39, ndf=64, n_layers=3,
                                          norm_layer=functools.partial(nn.BatchNorm2d,
                                                                       affine=True, track_running_stats=True),
                                          use_sigmoid=False)
    discriminator.train()
    discriminator.cuda()
    if opt.pretrain_checkpoint_D is not None:
        load_checkpoint_parallel(discriminator, opt.pretrain_checkpoint_D)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator).to(device)
    if opt.isTrain and len(opt.gpu_ids):
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[opt.local_rank])

    params_D = list(filter(lambda p: p.requires_grad,
                           discriminator.parameters()))
    optimizer_D = torch.optim.Adam(
        params_D, lr=opt.lr_D, betas=(opt.beta1, 0.999))

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionLSGANloss = GANLoss().cuda()

    if opt.local_rank == 0:
        writer = SummaryWriter(run_path)
        print('#training images = %d' % dataset_size)

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    step = 0
    step_per_batch = dataset_size

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        train_sampler.set_epoch(epoch)

        for ii, data in enumerate(train_loader):
            iter_start_time = time.time()

            total_steps += 1
            epoch_iter += 1

            person_clothes_edge = data['person_clothes_mask'].cuda()
            real_image = data['image'].cuda()
            preserve_mask = data['preserve_mask3'].cuda()
            preserve_region = real_image * preserve_mask
            warped_cloth = data['warped_cloth'].cuda()
            warped_prod_edge = data['warped_edge'].cuda()
            arms_color = data['arms_color'].cuda()
            arms_neck_label = data['arms_neck_lable'].cuda()
            pose = data['pose'].cuda()

            gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose],
                                   1)

            gen_outputs = model_gen(gen_inputs)
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite1 = m_composite * warped_prod_edge
            if opt.dataset == 'vitonhd':
                m_composite = person_clothes_edge.cuda() * m_composite1
            elif opt.dataset == 'dresscode':
                m_composite = m_composite1
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            set_requires_grad(discriminator, True)
            optimizer_D.zero_grad()
            pred_seg_D = p_rendered
            D_in_fake = torch.cat([gen_inputs, pred_seg_D.detach()], 1)
            D_in_real = torch.cat([gen_inputs, real_image], 1)
            loss_gan_D = (criterionLSGANloss(discriminator(
                D_in_fake), False) + criterionLSGANloss(discriminator(D_in_real), True)) * 0.5
            loss_gan_D.backward()
            optimizer_D.step()
            set_requires_grad(discriminator, False)

            D_in_fake_G = torch.cat([gen_inputs, pred_seg_D], 1)
            loss_gan_G = criterionLSGANloss(discriminator(D_in_fake_G), True) * 0.5

            loss_mask_l1 = torch.mean(torch.abs(1 - m_composite)) * 5
            loss_l1 = criterionL1(p_tryon, real_image.cuda())
            loss_vgg = criterionVGG(p_tryon, real_image.cuda())
            bg_loss_l1 = criterionL1(p_rendered, real_image.cuda())
            bg_loss_vgg = criterionVGG(p_rendered, real_image.cuda())
            gen_loss = (loss_l1 * 5 + loss_vgg + bg_loss_l1 * 5 + bg_loss_vgg + loss_mask_l1)

            if step % opt.write_loss_frep == 0:
                if opt.local_rank == 0:
                    writer.add_scalar('gen_loss', gen_loss, step)
                    writer.add_scalar('gen_mask_l1_loss', loss_mask_l1 * 1.0, step)
                    writer.add_scalar('gen_l1_loss', loss_l1 * 5, step)
                    writer.add_scalar('gen_vgg_loss', loss_vgg, step)
                    writer.add_scalar('gen_bg_l1_loss', bg_loss_l1 * 5, step)
                    writer.add_scalar('gen_bg_vgg_loss', bg_loss_vgg, step)
                    writer.add_scalar('gen_GAN_G_loss', loss_gan_G, step)
                    writer.add_scalar('gen_GAN_D_loss', loss_gan_D, step)

            loss_all = gen_loss + loss_gan_G

            optimizer_gen.zero_grad()
            loss_all.backward()
            optimizer_gen.step()

            ############## Display results and errors ##########
            if step % opt.display_freq == 0:
                if opt.local_rank == 0:
                    a = real_image.float().cuda()
                    e = warped_cloth
                    f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
                    ff = arms_color
                    g = preserve_region.cuda()
                    vis_pose = (pose > 0).float()
                    vis_pose = torch.sum(vis_pose.cuda(), dim=1).unsqueeze(1)
                    vis_pose_mask = (vis_pose > 0).to(
                        vis_pose.device).to(vis_pose.dtype)
                    h = torch.cat([vis_pose, vis_pose, vis_pose], 1)
                    i = p_rendered
                    j = torch.cat([m_composite1, m_composite1, m_composite1], 1)
                    k = p_tryon

                    l = torch.cat([arms_neck_label, arms_neck_label, arms_neck_label], 1)

                    combine = torch.cat(
                        [a[0], h[0], g[0], f[0], l[0], ff[0], e[0], j[0], i[0], k[0]], 2).squeeze()
                    cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                    writer.add_image('combine', (combine.data + 1) / 2.0, step)
                    rgb = (cv_img * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('sample/' + opt.name + '/' + str(epoch) +
                                '_' + str(step) + '.jpg', bgr)

            step += 1
            iter_end_time = time.time()
            iter_delta_time = iter_end_time - iter_start_time
            step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (
                        opt.niter + opt.niter_decay - epoch)
            eta = iter_delta_time * step_delta
            eta = str(datetime.timedelta(seconds=int(eta)))
            time_stamp = datetime.datetime.now()
            now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')

            if step % opt.print_freq == 0:
                if opt.local_rank == 0:
                    print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(
                        now, epoch_iter, step, loss_all, eta))

            if epoch_iter >= dataset_size:
                break

        iter_end_time = time.time()
        if opt.local_rank == 0:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        # if epoch % opt.save_epoch_freq == 0:
        if opt.local_rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            save_checkpoint(model_gen.module,
                            os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_gen_epoch_%03d.pth' % (epoch + 1)))
            save_checkpoint(discriminator.module,
                            os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_D_epoch_%03d.pth' % (epoch + 1)))
        if epoch > opt.niter:
            discriminator.module.update_learning_rate_warp(optimizer_D)
            model_gen.module.update_learning_rate(optimizer_gen)


opt = TrainOptions().parse()

train_tryon()
