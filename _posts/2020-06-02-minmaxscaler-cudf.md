---
layout: page
title: "Running RAPIDS"
subtitle: "Utilizing Your GPU for Non-Neural Network-Based Machine Learning Algorithms"
description: "In this post, I am going to give a brief introduction of RAPIDS, a suite of libraries for running machine learning algorithms on GPUs, as well as some codes for a MinMaxScaler."
excerpt: "Running RAPIDS - Utilizing Your GPU for Non-Neural Network-Based Machine Learning Algorithms"
image: "/assets/posts/2020-06-02-images/RAPIDS-code-snippet.png"
shortname: "RAPIDS-intro"
twitter_title: "Running RAPIDS - Utilizing Your GPU for Non-Neural Network-Based Machine Learning Algorithms"
twitter_image: "https://takmanman.github.io/assets/posts/2020-06-02-images/RAPIDS-code-snippet.png"
date: 2020-06-02
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3.0.0/es5/tex-mml-chtml.js">
  </script>

Not long ago, I could only use my GPU to run deep learning Algorithms. Libraries like TensorFlow and PyTorch allow me to build neural network models and run them on my GPU, which can drastically reduce computation time. For algorithms that are not neural network-based, such as support vector machine (SVM) and gradient boosted tree, I can use scikit-learn, but it does not run on GPUs.

Then I discovered RAPIDS through some posts on Kaggle. It is a suite of libraries that provides the GPU version of some non-neural network-based algorithms such as SVM. I gave it a try and I am rather happy with it for 3 reasons:
<ol>
<li>It took minimal effort to get it up and running.</li>
<li>It worked. It has greatly speeded up some of my algorithms.</li>
<li>It has a GPU version of DataFrame, which has a very similar API as the DataFrame in Pandas, making a lot of code transferrable and leveling the learning curve quite a bit.</li>
</ol>
I am going to talk about each of these a little bit more.

<h1>Getting RAPIDS up and running on your machine</h1>
Instead of installing RAPIDS directly on my machine, I downloaded its Docker image and run it in a Docker container. In fact, for GPU-utilizing libraries like TensorFlow and PyTorch, I always use their docker images if they are available. To run these libraries directly, you will have to set up your CUDA environment and it is not a trivial thing to do, and there is a chance that you may mess up your graphic display for other applications, such as your graphic intenisve games, like Monster Hunter.

You can get RAPIDS from <a href="https://rapids.ai/start.html#get-rapids" target="_blank">here</a>. I ran the following command and it worked for me.
```
docker pull rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04
```
It should launch JupyterLab automatically. If it does not, just launch your browser and go to <it>localhost:8888</it>.

<h1>A casual benchmark</h1>
To create a casual benchmark, I compared the computation time of <u>training a SVR (support vector machine for regression) with a 5000 by 1500 dataframe</u> using RAPIDS and that of using scikit-learn. Below is the code for creating the dataframes. cuDF is RAPIDS's GPU DataFrame library. I trained the SVR using a 7 fold cross-validation, which I think is a fairly typical step in build machine learning models.

```python
import numpy as np

import pandas as pd
import cudf #cuDF - RAPIDS's GPU DataFrame library

num_features = 1500
num_samples = 5000
data = np.sin(np.arange(num_samples*num_features)).reshape(num_samples,num_features)
noise = np.random.normal(0, 1, num_samples*num_features).reshape(num_samples,num_features)
data += noise

df_pdf = pd.DataFrame(data)

df_cdf = cudf.from_pandas(df_pdf)

from sklearn.model_selection import KFold

def metric(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred,2), axis=0)
```

Below is the code for training and validation using RAPIDS:
```python
%%time

import cupy
from cuml import SVR as cuSVR

NUM_FOLDS = 7
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

df = df_cdf #cudf dataframe
y = np.zeros(df.shape[0])

for f, (train_ind, val_ind) in enumerate(kf.split(df)):

    train_df = df.iloc[train_ind]
    val_df = df.iloc[val_ind]

    train_target = train_df.loc[:,0]#use the first column as target
    val_target = val_df.loc[:,0]

    #fit
    model = cuSVR(gamma = 'scale', cache_size=3000.0)
    model.fit(train_df.loc[:,1:], train_target)

    #predict
    pred = model.predict(val_df.loc[:,1:])
    y[val_ind] = pred   

    current_score = metric(val_target.values, pred.values)
    print(f"Fold {f} score: {current_score}")

score = metric(cupy.asnumpy(df.loc[:,0].values), y)

print(f"Average score: {score}")
```
Below is the code for training and validation using scikit-learn:
```python
%%time

from sklearn.svm import SVR as skSVR

NUM_FOLDS = 7
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)

df = df_pdf #Pandas dataframe
y = np.zeros(df.shape[0])

for f, (train_ind, val_ind) in enumerate(kf.split(df)):

    train_df = df.iloc[train_ind]
    val_df = df.iloc[val_ind]

    train_target = train_df.loc[:,0]#use the first column as target
    val_target = val_df.loc[:,0]

    #fit
    model = skSVR(gamma = 'scale', cache_size=3000.0)
    model.fit(train_df.loc[:,1:], train_target)

    #predict
    pred = model.predict(val_df.loc[:,1:])
    y[val_ind] = pred   

    current_score = metric(val_target.values, pred)
    print(f"Fold {f} score: {current_score}")

score = metric(df.loc[:,0].values, y)

print(f"Average score: {score}")
```

Below is the output using RAPIDS:
```output
Fold 0 score: 1.0565664517928794
Fold 1 score: 1.0218699584422126
Fold 2 score: 1.1321335285464693
Fold 3 score: 1.1731134341892044
Fold 4 score: 1.0188932244086784
Fold 5 score: 1.0831970748054487
Fold 6 score: 1.019768804212447
Average score: 1.072207152911587
CPU times: user 48.5 s, sys: 352 ms, total: 48.8 s
Wall time: 48.8 s
```
Below is the output using scikit-learn:
```output
Fold 0 score: 1.056573467098067
Fold 1 score: 1.021870356859294
Fold 2 score: 1.1321247007067812
Fold 3 score: 1.173098938676455
Fold 4 score: 1.018898749025693
Fold 5 score: 1.0831928625604506
Fold 6 score: 1.0197699711997656
Average score: 1.0722052365516568
CPU times: user 2min 54s, sys: 742 ms, total: 2min 55s
Wall time: 2min 55s
```

For this particular use case, RAPIDS is 3 times faster in terms of wall time.

<h1>A MinMaxScaler for cuDF</h1>
As we can see in the code snippet above, we can use a cuDF dataframe very much the same way we use a Pandas dataframe. You can find out more about the API <a href="https://docs.rapids.ai/api/cudf/stable/" target = "blank">here</a>.

However, as RAPIDS is still relatively new and is yet to be as comprehensive as scikit-learn, some of the commonly-used routines are not available. One such routine is the MinMaxScaler, which I pretty much always have to use when pre-processing the features. Luckily, it is very easy to write one. Something like the class below would work just fine:

```python
class cuMinMaxScaler():
    def __init__(self):
        self.feature_range = (0,1)

    def _reset(self):

        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_

    def fit(self, X): #X is assumed to be a cuDF dataframe, no type checking

        self._reset()        

        X = X.dropna()

        data_min = X.min(axis = 0) #cuDF series
        data_max = X.max(axis = 0) #cuDF series

        data_range = data_max - data_min #cuDF series

        data_range[data_range==0] = 1 #replaced with 1 is range is 0

        feature_range = self.feature_range

        self.scale_ = (feature_range[1] - feature_range[0]) / data_range # element-wise divison, produces #cuDF series
        self.min_ = feature_range[0] - data_min * self.scale_ # element-wise multiplication, produces #cuDF series

        return self

    def transform(self, X):

        X *= self.scale_ # element-wise divison, match dataframe column to series index
        X += self.min_ # element-wise addition, match dataframe column to series index

        return X
```
<h1>Issues I have encountered so far</h1>
Up until now I have only talked about the good things of RAPIDS, in fact, I have also ran into a couple of issues. I am going to talk about them here just as a forewarning for people that also want to try RAPIDS.

Firstly, when you merge two cuDF dataframes together, the resulting dataframe would have a totally randomized index. This can cause a lot of confusion down the road if you extract the target from one of the dataframe before the merge, as can be seen from the example below.
```python
import numpy as np
import cudf

df_1 = cudf.DataFrame({'a': np.arange(1000),'c': np.arange(1000)})
target = df_1['c']

df_2 = cudf.DataFrame({'b': np.arange(1000)})
df_3 = df_1.merge(df_2, left_index = True, right_index = True, how = "outer")
df_3 = df_3.drop('c')

print(f"df3:\n{df_3}")
print(f"target:\n{target}")
```
```output
df3:
       a    b
992  992  992
993  993  993
994  994  994
995  995  995
996  996  996
..   ...  ...
475  475  475
476  476  476
477  477  477
478  478  478
479  479  479

[1000 rows x 2 columns]
target:
0        0
1        1
2        2
3        3
4        4
      ...
995    995
996    996
997    997
998    998
999    999
Name: c, Length: 1000, dtype: int64
```
Fitting <span style="font-size:1.2rem; font-family:monospace">df_3</span> to <span style="font-size:1.2rem; font-family:monospace">target</span> is not going to give you the result you expect!

Secondly, when I tried to apply principal component analysis using RAPIDS's <span style="font-size:1.2rem; font-family:monospace">cuML.PCA</span>. Running <span style="font-size:1.2rem; font-family:monospace">fit_transform()</span> did not give me the same result as running <span style="font-size:1.2rem; font-family:monospace">fit().transform()</span>. This has already been filed as a <a href = "https://github.com/rapidsai/cuml/issues/2157" target = "blank">bug</a> and hopefully will be fixed in their next release.

<h1>Conclusions</h1>
Despite the issues I have encountered so far, I still think RAPIDS is promising. I probably will continue to use it for my non-neural network-based model as it can be a huge time-saver, especially when running a grid search on a dataset with thousands of features and tens of thousands of sample!
