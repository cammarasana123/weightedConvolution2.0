# weightedConvolution2.0
We propose the updated version of the weighted convolution.
The novel version offers an improvement in terms of computational cost.

## Use
Import the wConv2d from the wConv Class. 

You can use the function in substitution of nn.Conv2d, as

- wConv2d(in_channels, out_channels, kernel_size, den, padding, groups, bias)

where _den_ represents the density function coefficients.

Analogously, you can import and use wConv3d in substitution of nn.Conv3d, as

- wConv3d(in_channels, out_channels, kernel_size, den, padding, groups, bias)


## Density function values
We suggest to fine tune the density function values in the following range, where the first value represents the more external value of the density function.

- *3 x 3* kernel: [[0.5, 1.5]]
- *5 x 5* kernel: [[0.05, 1], [0.5, 1.5]]

For example:
- ```wConv2d(in_channels, out_channels, kernel_size = 1, den = [], padding, groups, bias) ```

- ```wConv2d(in_channels, out_channels, kernel_size = 3, den = [0.7], padding, groups, bias) ```

- ```wConv2d(in_channels, out_channels, kernel_size = 5, den= [0.2, 0.8], padding, groups, bias) ```

Analogously
- ```wConv3d(in_channels, out_channels, kernel_size = 1, den = [], padding, groups, bias) ```

- ```wConv3d(in_channels, out_channels, kernel_size = 3, den = [0.8], padding, groups, bias) ```

- ```wConv3d(in_channels, out_channels, kernel_size = 5, den= [0.1, 0.7], padding, groups, bias) ```

## Test files
The *wConv.py* is the class with the weighted convolution.

The *learning_2D.py* file implements a minimal learning model applying the weighted convolution to 2D images.

The *learning_3D.py* file implements a minimal learning model applying the weighted convolution to 3D images.

## Requirements
Python, PyTorch

Tested with: Python 3.11.10, PyTorch 2.6.0

## References
Please refer to the following articles:

Simone Cammarasana and Giuseppe Patanè. Optimal Density Functions for Weighted Convolution in Learning Models. 2025. DOI: https://arxiv.org/abs/2505.24527.

Simone Cammarasana and Giuseppe Patanè. Optimal Weighted Convolution for Classification and Denosing. 2025. DOI: https://arxiv.org/abs/2505.24558.

Also available at: https://huggingface.co/cammarasana123/weightedConvolution2.0
