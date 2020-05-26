---
layout: page
title: "Gaussian Noise and Gaussing Filter in Image Processing"
subtitle: "Imaging Concepts for Beginners"
description: "In this post, I am going to describe the Gaussian noise and Gaussian filter."
excerpt: "An introduction of Gaussian noise and Gaussing filter in image processing"
image: "/assets/posts/2020-05-23-images/combined.jpg"
shortname: "gaussian-noise-and-filter"
twitter_title: "Gaussian Noise and Gaussing Filter in Image Processing"
twitter_image: "https://takmanman.github.io/assets/posts/2020-05-23-images/combined.jpg"
date: 2020-05-23
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3.0.0/es5/tex-mml-chtml.js">
  </script>

  In image processing, two concepts are of utmost importance: Gaussian noise and Gaussian Filter.
  A solid understanding of these two concepts will pave a smooth path for further studies.

  We can conveniently think of noise as the unwanted signal in an image. Noise is random in nature. Gaussian noise is a type of noise that follows a Gaussian distribution.

  A fitler is a tool. It transforms images in various ways. A Gaussian filter is a tool for de-noising, smoothing and blurring.

  <h1> Why is Gaussian noise important in image processing?</h1>

  Noise in images arises from various sources. Under most conditions, these noises follow the Gaussian distribution and therefore are refered to as Gaussian noises. The main source of Gaussian noise includes sensor noise and electronic circuit noise.

  There are also noises that follow other probability distributions, e.g. shot noise can be described by a poisson distribution. However, according to the <a href="https://en.wikipedia.org/wiki/Central_limit_theorem" target="_blank">central limit theorem</a>, <i>when random variables that follow different distribuions are added together, the sum tends to follow a Gaussian distrbution.</i> Therefore, when we develop a single-noise model, as will be described next, we often choose to describe the noise as Gaussian noise.

  <h1> A Gaussian noise model for images </h1>
  <p>When we talk about the Gaussian noise in an image, usually we are thinking of additive Gaussian noise. In other words, if we describes the uncontaminated, free-of-noise source image as \(S(x,y)\), then the observed image \(O(x,y)\) is given by:</p>

  <p>\[O(x,y) = S(x,y) + e(x,y)\]</p>

  <p>where each \(e(x,y)\) is drawn from a Gaussian distribution. If we assume the noise is white, as we usually do, then each pair of \(e(x_1,y_1)\) and \(e(x_2,y_2)\) are independent of each other. Another assumption is every \(e(x,y)\) is drawn from the same Gaussian distrubtion. In short, we say \(e(x,y)\) is independently and identically distributed (often abbreviated as iid). </p>

  It is easy to simulate such noise in an image. Shown below are the source image and its Gaussian-noise-contaminated versions, and the python code that generated these images.

<figure>
  <div class="row">
  <div class="column">
    <img src="{{site.url}}/assets/posts/2020-05-23-images/berries.jpg" alt="Source image" style="width:100%">
  </div>
  <div class="column">
    <img src="{{site.url}}/assets/posts/2020-05-23-images/noise-20.jpg" alt="Image with Gaussian noise of standard deviation at 20" style="width:100%">
  </div>
  <div class="column">
    <img src="{{site.url}}/assets/posts/2020-05-23-images/noise-50.jpg" alt="Image with Gaussian noise of standard deviation at 50" style="width:100%">
  </div>
  </div>
  <figcaption>Fig.1 - Left: Source image. Pixel values are in the range of 0 to 255. Middle: Added Gaussian noise with standard deviation (\(\sigma\)) = 20. Right: Added Gaussian noise with \(\sigma\) = 50. </figcaption>
  </figure>

```python
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread('berries.jpg')
plt.figure(0)
imgplot = plt.imshow(img)

height = img.shape[0]
width = img.shape[1]

#Create an array of noise by drawing from a Gaussian distribution
#Add to source image
#Clip at [0, 255]
n_20 = np.random.normal(0, 20, [height, width, 3])
noisy_20_img = (img+n_20).clip(0,255)

n_50 = np.random.normal(0, 50, [height, width, 3])
noisy_50_img = (img+n_50).clip(0,255)

plt.figure(1)
imgplot = plt.imshow(noisy_20_img.astype('uint8'))

plt.figure(2)
imgplot = plt.imshow(noisy_50_img.astype('uint8'))
```

  <h1> Gaussian filter </h1>
  A filter is defined by its kernel. When we apply a filter to an image, the result is the convolution between the kernel and the orginal image.

  The kernel of a Gaussian filter is a 2d Gaussian function (Fig.2). When such a kernel is convolved with an image, it creates a blurring effect. This is because each pixel is now assigned with <b>a weighted sum of itself and its neighbouring pixels</b>. Any discontinuities in the neighbourhood are thereby <b>averaged out</b>.

  <figure>
  <img src="{{site.url}}/assets/posts/2020-05-23-images/gaussian2d_surface_plot.png"  style="display: block; margin: auto; "/>
  <figcaption>Fig.2 - A 2d Gaussian function with mean (\(\mu\)) at zero and standard deviation (\(\sigma\)) at 5.</figcaption>
  </figure>

  <p>A 2d Gaussian function is defined over the entire real plane while, surely, a Gaussian kernel must be of a finite size. Therefore we must truncate the Gaussian function at some threshold. A resonable choice would be 3 times the function's standard deviation (\(\sigma\)). This is equivalent to saying that, during convolution, pixels that are at a distance more than 3\(\sigma\) away do not contribute to the current pixel's new value.</p>

  <p>Shown below are several Gaussian kernels, each has a different \(\sigma\), but all truncated at 3\(\sigma\), and the blurred images they produced. </p>

  <figure>
    <div class="row">
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/kernel_1d_sigma_1.png" alt="Gaussian kernal at 1d view with standard deviation at 1" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/kernel_1d_sigma_3.png" alt="Gaussian kernal at 1d view with standard deviation at 3" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/kernel_1d_sigma_5.png" alt="Gaussian kernal at 1d view with standard deviation at 5" style="width:100%">
    </div>
    </div>
    <div class="row">
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/kernel_2d_sigma_1.png" alt="Gaussian kernal with standard deviation at 1" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/kernel_2d_sigma_3.png" alt="Gaussian kernal with standard deviation at 3" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/kernel_2d_sigma_5.png" alt="Gaussian kernal with standard deviation at 5" style="width:100%">
    </div>
    </div>
    <div class="row">
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/blurred-1.jpg" alt="Image similar to original" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/blurred-3.jpg" alt="Moderately blurred image" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/blurred-5.jpg" alt="Heavily blurred image" style="width:100%">
    </div>
    </div>
    <figcaption>Fig.3 - Top row: 1d view of the Gaussian kernel. Middle row: 2d view of the Gaussian kernel. Bottom row: Blurred image. Left column:  \(\sigma\) = 1. Middle column: \(\sigma\) = 3. Right column: \(\sigma\) = 5. </figcaption>
    </figure>

  It can be seen that as the kernel size grows larger, the blurring effect becomes more prominent. The code for applying the Gaussian kernel to create the blurred image is shown below:
  ```python
import math
from scipy import signal

import matplotlib.pyplot as plt

#standard deviation
sigma = 5
N = 6*sigma

# create a Gaussian function truncated at [-3*sigma, 3*sigma]
t = np.linspace(-3*sigma, 3*sigma, N)
gau = (1/(math.sqrt(2*math.pi)*sigma))*np.exp(-0.5*(t/sigma)**2)

# create a 2d Gaussian kernel from the 1d function
kernel = gau[:,np.newaxis]*gau[np.newaxis,:]

# convolve the image with the kernel
blurred = signal.fftconvolve(img, kernel[:, :, np.newaxis], mode='same')

# rescale to [0, 255]
blurred = (blurred - blurred.min())/(blurred.max()- blurred.min())*255

plt.imshow(blurred.astype('uint8'))
  ```
  <h1>Is Gaussian filter good for eliminating Gaussian noise?</h1> The short answer is no. Although they share the same name, one is not the remedy of the other, or at least not an optimal one. As shown in the pictures below, while a sufficienly large kernel can remove most of the Gaussian noise, it also takes away a lot of details of the image. To preserve the details while de-noising, one should consider the edge-preserving filters, such as the median filter and the bilateral filter.

  <figure>
    <div class="row">
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/noise-50.jpg" alt="An image contaminated by Gaussian noise" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/noisy-blurred-1.jpg" alt="Moderately de-noised image" style="width:100%">
    </div>
    <div class="column">
      <img src="{{site.url}}/assets/posts/2020-05-23-images/noisy-blurred-3.jpg" alt="Heavily de-noised and blurred image" style="width:100%">
    </div>
    </div>
    <figcaption>Fig.1 - Left: Original noisy image. Middle: Image de-noised by a Gaussian kernel with standard deviation (\(\sigma\)) = 1. Right: Image de-noised by a Gaussian kernel with \(\sigma\) = 3. </figcaption>
    </figure>
