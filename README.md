# Weighted Convolution2.0
The _weighted convolution_ applies a density function to scale the contribution of neighbouring pixels based on their distance from the central pixel. This choice differs from the traditional uniform convolution, which treats all neighbouring pixels equally.

The version 2.0 offers an improvement in terms of computational cost, with an execution time comparable to the nn.Conv2d

## Use
Import the wConv2d from the wConv Class:

```from wConv import wConv2d```

```from wConv import wConv3d```

You can use the function in substitution of nn.Conv2d, as

```wConv2d(in_channels, out_channels, kernel_size, den, stride, padding, groups, dilation, bias)```

where _den_ represents the density function coefficients.

Analogously, you can import and use wConv3d in substitution of nn.Conv3d, as

```wConv3d(in_channels, out_channels, kernel_size, den, stride, padding, groups, dilation, bias)```

## Info
Currently, wConv does not support anisotropic kernels (e.g., 3 x 5)


## Density function values
We suggest to fine tune the density function values in the following range, where the first value represents the more external value of the density function.

- *3 x 3* kernel: [[0.5, 1.5]]
- *5 x 5* kernel: [[0.05, 1], [0.5, 1.5]]

For example:

```wConv2d(in_channels, out_channels, kernel_size = 1, den = []) ```

```wConv2d(in_channels, out_channels, kernel_size = 3, den = [0.7]) ```

```wConv2d(in_channels, out_channels, kernel_size = 5, den= [0.2, 0.8]) ```

Analogously:

```wConv3d(in_channels, out_channels, kernel_size = 1, den = []) ```

```wConv3d(in_channels, out_channels, kernel_size = 3, den = [0.8]) ```

```wConv3d(in_channels, out_channels, kernel_size = 5, den= [0.1, 0.7]) ```

## Tuning strategy
A possible tuning strategy for a *3 x 3* kernel is to test three different density values: [0.9], [1.0], and [1.1], possibly using the same weights for the kernel initialisation and removing any randomness.
Then, the user can compare the trend of the density function, and move in that direction up to the optimal value.


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
