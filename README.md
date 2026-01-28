# Weighted Convolution2.0
The _weighted convolution_ applies a density function to scale the contribution of neighbouring pixels based on their distance from the central pixel. This choice differs from the traditional uniform convolution, which treats all neighbouring pixels equally.

The version 2.0 offers an improvement in terms of computational cost, with an execution time comparable to the nn.Conv2d

We propose two variants:
- **wConv**: non-trainable `den`, shared across all layers.  
- **wConv-Trainable**: `den` parameters are trainable independently at each layer.

## Use
###  Non-trainable variant
  
Import the wConv2d class from the wConv file:

```from wConv import wConv2d```

```from wConv import wConv3d```

You can use the function in substitution of nn.Conv2d, as

```wConv2d(in_channels, out_channels, kernel_size, den, stride, padding, groups, dilation, bias)```

where _den_ represents the density function coefficients.

Analogously, you can import and use wConv3d in substitution of nn.Conv3d, as

```wConv3d(in_channels, out_channels, kernel_size, den, stride, padding, groups, dilation, bias)```

###  Trainable variant
  
Import the wConv2d class from the wConv-Trainable file:
  ```from wConv import wConv2d```

You can use the function in substitution of nn.Conv2d, as

```wConv2d(in_channels, out_channels, kernel_size, stride, padding, groups, dilation, bias)```

## Info
Currently, wConv does not support anisotropic kernels (e.g., 3 x 5)


## Density function values
We suggest to fine tune the density function values in the following range, where the first value represents the more external value of the density function.

- *3 x 3* kernel: [[0.5, 1.5]]
- *5 x 5* kernel: [[0.05, 1], [0.5, 1.5]]
- *7 x 7* kernel: [[0.05, 0.5], [0.15, 1.0], [0.25, 1.5]]

For example:

```wConv2d(in_channels, out_channels, kernel_size = 1, den = []) ```

```wConv2d(in_channels, out_channels, kernel_size = 3, den = [0.7]) ```

```wConv2d(in_channels, out_channels, kernel_size = 5, den= [0.2, 0.8]) ```

```wConv2d(in_channels, out_channels, kernel_size = 7, den= [0.25, 0.15, 0.55]) ```

Analogously:

```wConv3d(in_channels, out_channels, kernel_size = 1, den = []) ```

```wConv3d(in_channels, out_channels, kernel_size = 3, den = [0.8]) ```

```wConv3d(in_channels, out_channels, kernel_size = 5, den= [0.1, 0.7]) ```

## Tuning strategy
As per our experimental tests, the density function provides the best results with larger kernels (e.g., *7 x 7*). For example, given a specific denoising task, we reached the best results with ```den = [0.25, 0.15, 0.55]```.

A possible tuning strategy for a *3 x 3* kernel is to test three different density values: [0.9], [1.0], and [1.1], possibly using the same weights for the kernel initialisation and removing any randomness.
Then, the user can compare the trend of the density function, and move in that direction up to the optimal value.

## Test files
The *wConv.py* is the class with the weighted convolution.

The *wConv-Trainable.py* is the class with the weighted convolution where the den parameters are trainable by the model. Each layer of the architecture has its own den parameters.

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

## Some results

Liu, Jianlei, et al. "Dual Prompts Guided Cross-Domain Transformer for Unified Day-Night Image Dehazing." _Knowledge-Based Systems_ (2026): 115362.

Cai, Hao, et al. "LFP-Mono: Lightweight Self-Supervised Network Applying Monocular Depth Estimation to Low-Altitude Environment Scenarios." _Computers 15.1_ (2026): 19.

Cui, Yuxin, Penghui Li, and Bo Xia. "CTB-YOLO: A Lightweight Framework for Robust Cotton Terminal Bud Detection in Complex Field Environments." _Smart Agricultural Technology_ (2026): 101795.

Zhang, Cheng, et al. "AMFS-Net: An Adaptive Multi-Scale Feature Fusion Framework for Efficient SAR Ship Detection." _IEEE Geoscience and Remote Sensing Letters_ (2026).

Tian, Xing, and Juan Wang. "FWN: Global-Local Feature Weighted Fusion Network for Medical Image Classification." _6th International Conference on Internet of Things, Artificial Intelligence and Mechanical Automation (IoTAIMA)_. IEEE, 2025.
