---
layout: tutorial
title: "A Very Gentle Introduction to Convolutional Neural Network (CNN) in PyTorch"
date: 2019-09-01
---
In this tutorial, I am going to walk through the necessary steps of building a CNN in PyTorch. I will also talk about how to choose the filter sizes and stack the layers such that it will not give you a dimension mismatch error.

In the end, I will discuss the dimensions of the filters and the size of the intermediate output of different layers, as well as the number of parameters of a given CNN.

This tutorial is meant to be straight forward and focused, and hopefully in the end you will be able to implement any CNN model in PyTorch with ease.

<h2>Building a CNN</h2>

In PyTorch, we build a CNN model by creating a subclass of torch.NN.Module. I am not going to discuss what exactly torch.NN.Module is. For now, I think it is sufficient to say that it is (should be) the base class of all neural network models in PyTorch.

In the definition of this subclass, we must implement two of its member functions: <span>&#95;&#95;</span>init()<span>&#95;</span> and forward().

```python
def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
