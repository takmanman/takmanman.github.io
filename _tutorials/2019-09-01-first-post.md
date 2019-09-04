---
layout: tutorial
title: "A Very Gentle Introduction to Convolutional Neural Network (CNN) in PyTorch"
date: 2019-09-01
---
In this tutorial, I am going to walk through the necessary steps of building a CNN model in PyTorch. I will also talk about how to choose the filter sizes (kernel sizes) and stack the layers such that it will not give you a size mismatch error.

In the end, I will discuss the dimensions of the filters and the intermediate output of different layers, as well as the number of parameters of a given CNN.

This tutorial is meant to be straight forward and focused only on buiding a CNN model (but not how to train it), and hopefully in the end you will be able to implement any CNN model in PyTorch with ease.

<h2>Building a CNN</h2>

In PyTorch, we build a CNN model by creating a subclass of torch.NN.Module. I am not going to discuss what exactly torch.NN.Module is. For now, I think it is sufficient to say that it is (should be) the base class of all neural network models in PyTorch.

In the definition of this subclass (let's call it my CNN), we must implement two of its method: ```__init__()``` and ```forward()```.

<h3> Example 1 </h3>

```python
import torch
import torch.nn as nn

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32*8*8,10)
        
    def forward(self, x):
        x = x.view(1, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32*8*8) #flatten the feature map
        x = self.fc(x)
        
        return x
```
In ```__init__()```, we define the layers. In this example, the network has 2 convolution layers, and one fully-connected layer. 

In ```forward()```, we specify how to stack the layers and how an input image is passed along the network. This is where we specify the activation fucntion for each convolution layer.

```python
cnn = myCNN()
x_in = torch.randn(1, 3, 32, 32)
out = cnn(x_in)
print(out)
out2 = cnn.forward(x_in)
print(out2)
```

```
torch.Size([1, 2048])
tensor([[ 0.0209,  0.0363, -0.1219,  0.0827, -0.0073,  0.0426, -0.1122, -0.0262,
         -0.0789, -0.0085]], grad_fn=<AddmmBackward>)
torch.Size([1, 2048])
tensor([[ 0.0209,  0.0363, -0.1219,  0.0827, -0.0073,  0.0426, -0.1122, -0.0262,
         -0.0789, -0.0085]], grad_fn=<AddmmBackward>)
```

And this is all you need to do to build a CNN model in PyTorch.

<h2> Kernel size and layer output </h2>
The first convolution layer take an input image with 3 channels, and output a feature of depth 16 (or 16 channels).
The second convolution layer take an input feature map of depth 16 and output a feature map of depth 32.


