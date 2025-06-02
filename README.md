# weightedConvolution2.0
We propose the updated version of the weighted convolution.
The novel version offers an improvement in terms of computational cost.

## Use
Import the wConv2d from the wConv Class. 

You can use the function in substitution of nn.Conv2d, as

wConv2d(in_channels, out_channels, kernel_size, padding, den, bias)

## Density function values
We suggest to fine tune the density function values in the following range, where the first value represents the more external value of the density function.

- *3 x 3* kernel: [0.5, 1.5]
- *5 x 5* kernel: [0.05, 1], [0.5, 1.5]

## Test files
The *wConv* is the class with the weighted convolution.

The *learning* file implements a minimal learning model applying the weighted convolution.

## Requirements
Python, PyTorch

Tested with: Python 3.11.10, PyTorch 2.6.0
