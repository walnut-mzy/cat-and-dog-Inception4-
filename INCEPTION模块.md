# INCEPTION模块

## BN算法

[这里借鉴了一个博客](https://blog.csdn.net/qq_37100442/article/details/81776191)

   一、BN算法产生的背景

         做深度学习大家应该都知道，我们在数据处理部分，我们为了加速训练首先会对数据进行处理的。其中我们最常用的是零均值和PCA（白话）。首先我们进行简单介绍零均值带来的只管效果：


      ![img](https://img-blog.csdn.net/20180817134325575?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MTAwNDQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)![img](https://img-blog.csdn.net/20180817134407542?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3MTAwNDQy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)                                               

简单的划了一个草图。第一张图我们进行分析。由于我们对网络进行参数初始化，我们一般是采用零均值化。我们初始的拟合直线也就是红色部分。另外的一条绿色直线，是我们的目标直线。从图能够直观看出，我们应该需要多次迭代才能得到我们的需要的目标直线。我们再看第二张图，假设我们还是和第一张图有相同的分布，只是我们做了减均值，让数据均值为零。能够直观的发现可能只进行简单的微调就能够实现拟合（理想）。大大提高了我们的训练速度。因此，在训练开始前，对数据进行零均值是一个必要的操作。但是，随着网络层次加深参数对分布的影响不定。导致网络每层间以及不同迭代的相同相同层的输入分布发生改变，导致网络需要重新适应新的分布，迫使我们降低学习率降低影响。在这个背景下BN算法开始出现。       有些人首先提出在每层增加PCA白化(先对数据进行去相关然后再进行归一化)，这样基本满足了数据的0均值、单位方差、弱相关性。但是这样是不可取的，因为在白化过程中会计算协方差矩阵、求逆等操作，计算量会很大，另外，在反向传播时，白化的操作不一定可微。因此，在此背景下BN算法开始出现。

 

 二、BN算法的实现和优点
   1、BN算法的产生

   上面提到了PCA白化优点，能够去相关和数据均值，标准值归一化等优点。但是当数据量比较大的情况下去相关的话需要大量的计算，因此有些人提出了只对数据进行均值和标准差归一化。叫做近似白化预处理。

![img](https://img-blog.csdn.net/20160312181715397)

由于训练过程采用了batch随机梯度下降，因此![img](https://img-blog.csdn.net/20171031164204257?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)指的是一批训练数据时，![img](https://img-blog.csdn.net/20171031164350254?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3VveXVuZmVpMjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)各神经元输入值的平均值；指的是一批训练数据时各神经元输入值的标准差。

但是，这些应用到深度学习网络还远远不够，因为可能由于这种的强制转化导致数据的分布发生破话。因此需要对公式的鲁棒性进行优化，就有人提出了变换重构的概念。就是在基础公式的基础之上加上了两个参数γ、β。这样在训练过程中就可以学习这两个参数，采用适合自己网络的BN公式。公式如下：

     ![img](https://img-blog.csdn.net/20160312190113493)

每一个神经元都会有一对这样的参数γ、β。这样其实当：       

 ![img](https://img-blog.csdn.net/20160312190336072)![img](https://img-blog.csdn.net/20160312190323411)

时，是可以恢复出原始的某一层所学到的特征的。引入可学习重构参数γ、β，让网络可以学习恢复出原始网络所要学习的特征分布。

总结上面我们会得到BN的向前传导公式：

![img](https://img-blog.csdn.net/20160312190726792)

2、BN算法在网络中的作用

   BN算法像卷积层，池化层、激活层一样也输入一层。BN层添加在激活函数前，对输入激活函数的输入进行归一化。这样解决了输入数据发生偏移和增大的影响。

优点：

```python
   1、加快训练速度，能够增大学习率，及时小的学习率也能够有快速的学习速率;

   2、不用理会拟合中的droupout、L2 正则化项的参数选择，采用BN算法可以省去这两项或者只需要小的L2正则化约束。原因，BN算法后，参数进行了归一化，原本经过激活函数没有太大影响的神经元，分布变得明显，经过一个激活函数以后，神经元会自动削弱或者去除一些神经元，就不用再对其进行dropout。另外就是L2正则化，由于每次训练都进行了归一化，就很少发生由于数据分布不同导致的参数变动过大，带来的参数不断增大。

   3、 可以吧训练数据集打乱，防止训练发生偏移。
```

使用： 在卷积中，会出现每层卷积层中有（L）多个特征图。AxAxL特征矩阵。我们只需要以每个特征图为单元求取一对γ、β。

然后在对特征图进行神经元的归一化。

## INCEPTION-V4模块

[这里有一篇论文分析](https://blog.csdn.net/qq_38807688/article/details/84590291)

![table1](https://img-blog.csdnimg.cn/20181128185423184.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4ODA3Njg4,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20181128185141694.png)

![img](https://img-blog.csdnimg.cn/20181128185214507.png)

![img](https://img-blog.csdnimg.cn/20181128185244849.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4ODA3Njg4,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20181128185318403.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4ODA3Njg4,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20181128185348262.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4ODA3Njg4,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/2018112818545721.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM4ODA3Njg4,size_16,color_FFFFFF,t_70)

```python
class Inception_stem(keras.layers.Layer):
    def __init__(self):
        super(Inception_stem, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=32,kernel_size=3,padding="valid",strides=2)
        self.conv2d2=keras.layers.Conv2D(filters=32, kernel_size=3, padding="valid", strides=1)
        self.conv2d3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", strides=1)
        self.maxpool1=keras.layers.MaxPool2D(pool_size=3,strides=2,padding="valid")
        self.conv2d4=keras.layers.Conv2D(kernel_size=3,strides=2,filters=96,padding="valid")
        self.conv2d5=keras.layers.Conv2D(kernel_size=1,filters=64,padding="same",strides=1)
        self.conv2d5_1 = keras.layers.Conv2D(kernel_size=1, filters=64, padding="same", strides=1)
        self.conv2d6=keras.layers.Conv2D(kernel_size=(7,1),filters=64,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(kernel_size=(1,7),filters=64,padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(kernel_size=3,filters=96,padding="valid",strides=1)
        self.conv2d10=keras.layers.Conv2D(kernel_size=1,filters=64,padding="same",strides=1)
        self.conv2d8_1 = keras.layers.Conv2D(kernel_size=3, filters=96, padding="valid", strides=1)
        self.conv2d9=keras.layers.Conv2D(kernel_size=3,filters=192,strides=2,padding="valid")
        self.maxpool2=keras.layers.MaxPool2D(pool_size=2,strides=2,padding="valid")
    def call(self, inputs, **kwargs):
        #inputs
        x=self.conv2d1(inputs)
        x=self.conv2d2(x)
        x=self.conv2d3(x)
        x1=self.maxpool1(x)
        x2=self.conv2d4(x)
        #Filter concat 73x73x160
        x=tf.concat([x1,x2],3)
        x1=self.conv2d5(x)
        x1=self.conv2d6(x1)
        x1=self.conv2d7(x1)
        x1=self.conv2d8(x1)
        x2=self.conv2d5_1(x)
        x2=self.conv2d8_1(x2)
        #Filter concat 71x71x192
        x=tf.concat([x1,x2],axis=3)
        # print("shape:", x.shape)
        x1=self.conv2d9(x)
        x2=self.maxpool2(x)
        #Filter concat 35x35x384
        x=tf.concat([x1,x2],axis=3)
        # print(x)
        return x
class Inception_A(keras.layers.Layer):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=64,kernel_size=1,strides=1,padding="same")
        self.conv2d2=keras.layers.Conv2D(filters=64,kernel_size=1,strides=1,padding="same")
        self.conv2d3=keras.layers.Conv2D(filters=96,kernel_size=1,strides=1,padding="same")
        self.avpool=keras.layers.AveragePooling2D(padding="same",pool_size=2,strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=96,kernel_size=1,strides=1,padding="same")
        self.conv2d5=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.conv2d6=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.conv2d7=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x2=self.conv2d2(inputs)
        x3=self.conv2d3(inputs)
        x4=self.avpool(inputs)
        x4=self.conv2d4(x4)
        x2=self.conv2d5(x2)
        x1=self.conv2d6(x1)
        x1=self.conv2d7(x1)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        #print(x)
        return x
class Inception_B(keras.layers.Layer):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(kernel_size=(1,7),filters=192,padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(kernel_size=(7,1),filters=224,padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(kernel_size=(7,1),filters=256,padding="same",strides=1)
        self.conv2d5 = keras.layers.Conv2D(kernel_size=(7, 1), filters=224, padding="same", strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(filters=224,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(filters=256,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d9=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.conv2d10=keras.layers.Conv2D(filters=128,kernel_size=1,padding="same",strides=1)
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1=self.conv2d5(x1)
        x1=self.conv2d4(x1)
        x2=self.conv2d6(inputs)
        x2=self.conv2d7(x2)
        x2=self.conv2d8(x2)
        x3=self.conv2d9(inputs)
        x4=self.avgpool(inputs)
        x4=self.conv2d10(x4)
        #print(x4.shape)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        #print(x)
        return x
class Inception_C(keras.layers.Layer):
    def __init__(self):
        super(Inception_C, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(filters=448,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(filters=512,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=256,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d5=keras.layers.Conv2D(filters=256,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(filters=256,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(filters=256,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d9=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.conv2d10=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1_1=self.conv2d4(x1)
        x1_2=self.conv2d5(x1)
        x1=tf.concat([x1_1,x1_2],axis=3)
        x2=self.conv2d6(inputs)
        x2_1=self.conv2d7(x2)
        x2_2=self.conv2d8(x2)
        x2=tf.concat([x2_1,x2_2],axis=3)
        x3=self.conv2d9(inputs)
        x4=self.avgpool(inputs)
        x4=self.conv2d10(x4)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        #print(x)
        return x
class Inception_redution_A(keras.layers.Layer):
    def __init__(self):
        super(Inception_redution_A, self).__init__()
        #这里有两三种网络结构，这里我们直接使用Inception_v4模块
        self.conv2d1=keras.layers.Conv2D(kernel_size=1,filters=192,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(kernel_size=3,filters=224,padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(kernel_size=3,filters=256,padding="valid",strides=2)
        self.conv2d4=keras.layers.Conv2D(kernel_size=3,filters=384,padding="valid",strides=2)
        self.maxpool=keras.layers.MaxPool2D(pool_size=3,strides=2,padding="valid")
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x2=self.conv2d4(inputs)
        x3=self.maxpool(inputs)
        x=tf.concat([x1,x2,x3],axis=3)
        #print(x)
        return x
class Inception_redution_B(keras.layers.Layer):
    def __init__(self):
        super(Inception_redution_B, self).__init__()
        #这里有两三种网络结构，这里我们直接使用Inception_v4模块
        self.conv2d1=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(filters=256,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(filters=320,kernel_size=(7,1),padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=320,kernel_size=3,padding="valid",strides=2)
        self.conv2d5=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=192,kernel_size=3,padding="valid",strides=2)
        self.maxpool=keras.layers.MaxPool2D(strides=2,padding="valid",pool_size=3)
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1=self.conv2d4(x1)
        x2=self.conv2d5(inputs)
        x2=self.conv2d6(x2)
        x3=self.maxpool(inputs)
        x=tf.concat([x1,x2,x3],axis=3)
        #print(x)
        return x
class Inception(keras.layers.Layer):
    def __init__(self):
        super(Inception, self).__init__()
        self.stem=Inception_stem()
        self.Inception1=Inception_A()
        self.Inception2=Inception_A()
        self.Inception3=Inception_A()
        self.Inception4=Inception_A()
        self.reduction_A=Inception_redution_A()
        self.Inception_b1=Inception_B()
        self.Inception_b2=Inception_B()
        self.Inception_b3=Inception_B()
        self.Inception_b4=Inception_B()
        self.Inception_b5=Inception_B()
        self.Inception_b6=Inception_B()
        self.Inception_b7=Inception_B()
        self.reduction_B=Inception_redution_B()
        self.Inception_c1=Inception_C()
        self.Inception_c2=Inception_C()
        self.Inception_c3=Inception_C()
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.droupout=keras.layers.Dropout(0.2)
        self.fl=keras.layers.Flatten()
        self.soft=keras.layers.Dense(2,activation="softmax")
        self.bn=keras.layers.BatchNormalization()
        self.bn1=keras.layers.BatchNormalization()
        self.bn2=keras.layers.BatchNormalization()
        self.bn3=keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()
        #self.reshape=keras.layers.Reshape([])
    def call(self, inputs, **kwargs):
        x=self.stem(inputs)
        x=self.Inception1(x)
        x=self.Inception2(x)
        x=self.Inception3(x)
        x=self.Inception4(x)
        x=self.reduction_A(x)
        x=self.bn(x)
        x=self.Inception_b1(x)
        x=self.Inception_b2(x)
        x=self.Inception_b3(x)
        x=self.Inception_b4(x)
        x=self.bn1(x)
        x=self.Inception_b5(x)
        x=self.Inception_b6(x)
        x=self.Inception_b7(x)
        x=self.reduction_B(x)
        x=self.bn2(x)
        x=self.Inception_c1(x)
        x=self.Inception_c2(x)
        x=self.Inception_c3(x)
        x=self.bn3(x)
        x=self.avgpool(x)
        x=self.droupout(x)
        x=self.fl(x)
        x=self.bn4(x)
        #print(x.shape)
        x=self.soft(x)
        #x=self.reshape(x) #这个函数的作用是使其于标签属性一致
        #print(x)
        return x

```

## 这里我们做一个有关动物的数据分类

### Overview

**Dogs vs. Cats** is a competition on [Kaggle](https://www.kaggle.com/), which needs to write an algorithm to classify whether images contain either a dog or a cat. The training archive contains 25,000 images of dogs and cats.

### The Asirra data set

Web services are often protected with a challenge that's supposed to be easy for people to solve, but difficult for computers. Such a challenge is often called a [CAPTCHA](http://www.captcha.net/) (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof). HIPs are used for many purposes, such as to reduce email and blog spam and prevent brute-force attacks on web site passwords.

[Asirra](http://research.microsoft.com/en-us/um/redmond/projects/asirra/) (Animal Species Image Recognition for Restricting Access) is a HIP that works by asking users to identify photographs of cats and dogs. This task is difficult for computers, but studies have shown that people can accomplish it quickly and accurately. Many even think it's fun!

Asirra is unique because of its partnership with [Petfinder.com](http://www.petfinder.com/), the world's largest site devoted to finding homes for homeless pets. They've provided Microsoft Research with over three million images of cats and dogs, manually classified by people at thousands of animal shelters across the United States. Kaggle is fortunate to offer a subset of this data for fun and research.

### Image recognition attacks

While random guessing is the easiest form of attack, various forms of image recognition can allow an attacker to make guesses that are better than random. There is enormous diversity in the photo database (a wide variety of backgrounds, angles, poses, lighting, etc.), making accurate automatic classification difficult. In an informal poll conducted many years ago, computer vision experts posited that a classifier with better than 60% accuracy would be difficult without a major advance in the state of the art. For reference, a 60% classifier improves the guessing probability of a 12-image HIP from 1/4096 to 1/459.

### State of the art

The current literature suggests machine classifiers can score above 80% accuracy on this task [[1\]](http://xenon.stanford.edu/~pgolle/papers/dogcat.pdf). Therefore, Asirra is no longer considered safe from attack. This contest aims to benchmark the latest computer vision and deep learning approaches to this problem.

数据集如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/5abd4167534945dd982f062ff30e7cd1.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


于是我们可以有处理数据集的脚本：

```python
#C:\Users\mzy\Desktop\机器学习\data\train
import tensorflow as tf
import random
import os
def image_deals1(train_file):       # 读取原始文件
    image_string = tf.io.read_file(train_file)  # 读取原始文件
    image_decoded = tf.image.decode_png(image_string)  # 解码JPEG图片
    image_decoded=randoc(image_decoded)
    image_decoded= tf.image.resize(image_decoded, [299, 299])  #把图片转换为224*224的大小
    #image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.cast(image_decoded, dtype=tf.float32) / 255.0-0.5
    return image
def image_deals(train_file):       # 读取原始文件
    image_string = tf.io.read_file(train_file)  # 读取原始文件
    image_decoded = tf.image.decode_png(image_string)  # 解码JPEG图片
    image_decoded=randoc(image_decoded)
    image_decoded= tf.image.resize(image_decoded, [299, 299])  #把图片转换为224*224的大小
    #image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.cast(image_decoded, dtype=tf.float32) / 255.0-0.5
    return image
def randoc(train_file):
    int1=random.randint(1,10)
    if int1==1:
        train_file = tf.image.random_flip_left_right(train_file)   #左右翻折
    elif int1==2:
        train_file=tf.image.random_flip_up_down(train_file)
    return train_file

def train_test_get(train_test_inf):
    for root,dir,files in os.walk(train_test_inf, topdown=False):
        #print(root)
        #print(files)
        list=[root+"/"+i for i in files]
        #print(list)
        filename=[]
        for i in files:
            label=i[0:3]
            if label=="cat":
                #x1 = tf.constant([0, 1], shape=(1, 2))
                x1=[0,1]
                filename.append(x1)
            else:
                #x2 = tf.constant([1, 0], shape=(1, 2))
                x2=[0,1]
                filename.append(x2)

        json={
            "list":list,
            "filename":filename
        }
        print(len(list))
        print(len(filename))
        return json
def dogandcat():
    json_train=train_test_get("C:/Users/mzy/Desktop/机器学习/data/train1")
    list_file=json_train["list"]
    list_filename=json_train["filename"]
    print(list_file)
    image_list=[image_deals(i) for i in list_file]
    #image_list=tf.expand_dims(image_list,axis=1)
    # print(image_list.shape)
    dataest=tf.data.Dataset.from_tensor_slices((image_list, list_filename))
    dataest=dataest.shuffle(buffer_size=300).repeat(count=10).prefetch(tf.data.experimental.AUTOTUNE).batch(10)

    print(dataest)
    return dataest
#dogandcat()
def dogandcat1():
    json_train=train_test_get("C:/Users/mzy/Desktop/机器学习/data/test1")
    list_file=json_train["list"]
    list_filename=json_train["filename"]
    print(list_file)
    image_list=[image_deals(i) for i in list_file]
    #image_list=tf.expand_dims(image_list,axis=1)
    # print(image_list.shape)
    dataest=tf.data.Dataset.from_tensor_slices((image_list, list_filename))
    dataest=dataest.shuffle(buffer_size=300).repeat(count=10).prefetch(tf.data.experimental.AUTOTUNE).batch(10)

    #print(dataest)
    return dataest
```

#由于笔者的电脑太拉跨了，于是，我们只能训练500张图片将就一下#

## 训练代码

```python
import tensorflow as tf
from tensorflow import keras,metrics
from 动物数据集分类 import dogandcat,dogandcat1
from tensorflow.keras import losses, optimizers,initializers
import random
import os
#我本来打算做一个手写汉字的模型，但是，我下数据集的网站崩了，只能先用猫狗识别的数据集先将就一下，回来在把手写汉字的数据集用强化学习的知识学一下
class Inception_stem(keras.layers.Layer):
    def __init__(self):
        super(Inception_stem, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=32,kernel_size=3,padding="valid",strides=2)
        self.conv2d2=keras.layers.Conv2D(filters=32, kernel_size=3, padding="valid", strides=1)
        self.conv2d3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", strides=1)
        self.maxpool1=keras.layers.MaxPool2D(pool_size=3,strides=2,padding="valid")
        self.conv2d4=keras.layers.Conv2D(kernel_size=3,strides=2,filters=96,padding="valid")
        self.conv2d5=keras.layers.Conv2D(kernel_size=1,filters=64,padding="same",strides=1)
        self.conv2d5_1 = keras.layers.Conv2D(kernel_size=1, filters=64, padding="same", strides=1)
        self.conv2d6=keras.layers.Conv2D(kernel_size=(7,1),filters=64,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(kernel_size=(1,7),filters=64,padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(kernel_size=3,filters=96,padding="valid",strides=1)
        self.conv2d10=keras.layers.Conv2D(kernel_size=1,filters=64,padding="same",strides=1)
        self.conv2d8_1 = keras.layers.Conv2D(kernel_size=3, filters=96, padding="valid", strides=1)
        self.conv2d9=keras.layers.Conv2D(kernel_size=3,filters=192,strides=2,padding="valid")
        self.maxpool2=keras.layers.MaxPool2D(pool_size=2,strides=2,padding="valid")
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        #inputs
        x=self.conv2d1(inputs)
        x=self.conv2d2(x)
        x=self.conv2d3(x)
        x1=self.maxpool1(x)
        x2=self.conv2d4(x)
        #Filter concat 73x73x160
        x=tf.concat([x1,x2],3)
        x1=self.conv2d5(x)
        x1=self.conv2d6(x1)
        x1=self.conv2d7(x1)
        x1=self.conv2d8(x1)
        x2=self.conv2d5_1(x)
        x2=self.conv2d8_1(x2)
        #Filter concat 71x71x192
        x=tf.concat([x1,x2],axis=3)
        # print("shape:", x.shape)
        x1=self.conv2d9(x)
        x2=self.maxpool2(x)
        #Filter concat 35x35x384
        x=tf.concat([x1,x2],axis=3)
        x=self.bn(x)
        # print(x)
        return x
class Inception_A(keras.layers.Layer):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=64,kernel_size=1,strides=1,padding="same")
        self.conv2d2=keras.layers.Conv2D(filters=64,kernel_size=1,strides=1,padding="same")
        self.conv2d3=keras.layers.Conv2D(filters=96,kernel_size=1,strides=1,padding="same")
        self.avpool=keras.layers.AveragePooling2D(padding="same",pool_size=2,strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=96,kernel_size=1,strides=1,padding="same")
        self.conv2d5=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.conv2d6=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.conv2d7=keras.layers.Conv2D(filters=96,kernel_size=3,strides=1,padding="same")
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x2=self.conv2d2(inputs)
        x3=self.conv2d3(inputs)
        x4=self.avpool(inputs)
        x4=self.conv2d4(x4)
        x2=self.conv2d5(x2)
        x1=self.conv2d6(x1)
        x1=self.conv2d7(x1)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_B(keras.layers.Layer):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(kernel_size=(1,7),filters=192,padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(kernel_size=(7,1),filters=224,padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(kernel_size=(7,1),filters=256,padding="same",strides=1)
        self.conv2d5 = keras.layers.Conv2D(kernel_size=(7, 1), filters=224, padding="same", strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(filters=224,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(filters=256,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d9=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.conv2d10=keras.layers.Conv2D(filters=128,kernel_size=1,padding="same",strides=1)
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1=self.conv2d5(x1)
        x1=self.conv2d4(x1)
        x2=self.conv2d6(inputs)
        x2=self.conv2d7(x2)
        x2=self.conv2d8(x2)
        x3=self.conv2d9(inputs)
        x4=self.avgpool(inputs)
        x4=self.conv2d10(x4)
        #print(x4.shape)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_C(keras.layers.Layer):
    def __init__(self):
        super(Inception_C, self).__init__()
        self.conv2d1=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(filters=448,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(filters=512,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=256,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d5=keras.layers.Conv2D(filters=256,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=384,kernel_size=1,padding="same",strides=1)
        self.conv2d7=keras.layers.Conv2D(filters=256,kernel_size=(1,3),padding="same",strides=1)
        self.conv2d8=keras.layers.Conv2D(filters=256,kernel_size=(3,1),padding="same",strides=1)
        self.conv2d9=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.conv2d10=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1_1=self.conv2d4(x1)
        x1_2=self.conv2d5(x1)
        x1=tf.concat([x1_1,x1_2],axis=3)
        x2=self.conv2d6(inputs)
        x2_1=self.conv2d7(x2)
        x2_2=self.conv2d8(x2)
        x2=tf.concat([x2_1,x2_2],axis=3)
        x3=self.conv2d9(inputs)
        x4=self.avgpool(inputs)
        x4=self.conv2d10(x4)
        x=tf.concat([x1,x2,x3,x4],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_redution_A(keras.layers.Layer):
    def __init__(self):
        super(Inception_redution_A, self).__init__()
        #这里有两三种网络结构，这里我们直接使用Inception_v4模块
        self.conv2d1=keras.layers.Conv2D(kernel_size=1,filters=192,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(kernel_size=3,filters=224,padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(kernel_size=3,filters=256,padding="valid",strides=2)
        self.conv2d4=keras.layers.Conv2D(kernel_size=3,filters=384,padding="valid",strides=2)
        self.maxpool=keras.layers.MaxPool2D(pool_size=3,strides=2,padding="valid")
        self.bn=keras.layers.BatchNormalization()
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x2=self.conv2d4(inputs)
        x3=self.maxpool(inputs)
        x=tf.concat([x1,x2,x3],axis=3)
        x=self.bn(x)
        #print(x)
        return x
class Inception_redution_B(keras.layers.Layer):
    def __init__(self):
        super(Inception_redution_B, self).__init__()
        #这里有两三种网络结构，这里我们直接使用Inception_v4模块
        self.conv2d1=keras.layers.Conv2D(filters=256,kernel_size=1,padding="same",strides=1)
        self.conv2d2=keras.layers.Conv2D(filters=256,kernel_size=(1,7),padding="same",strides=1)
        self.conv2d3=keras.layers.Conv2D(filters=320,kernel_size=(7,1),padding="same",strides=1)
        self.conv2d4=keras.layers.Conv2D(filters=320,kernel_size=3,padding="valid",strides=2)
        self.conv2d5=keras.layers.Conv2D(filters=192,kernel_size=1,padding="same",strides=1)
        self.conv2d6=keras.layers.Conv2D(filters=192,kernel_size=3,padding="valid",strides=2)
        self.maxpool=keras.layers.MaxPool2D(strides=2,padding="valid",pool_size=3)
    def call(self, inputs, **kwargs):
        x1=self.conv2d1(inputs)
        x1=self.conv2d2(x1)
        x1=self.conv2d3(x1)
        x1=self.conv2d4(x1)
        x2=self.conv2d5(inputs)
        x2=self.conv2d6(x2)
        x3=self.maxpool(inputs)
        x=tf.concat([x1,x2,x3],axis=3)
        #print(x)
        return x
class Inception(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        #self.attention_dim = attention_dim
        super(Inception, self).__init__()
        self.stem=Inception_stem()
        self.Inception1=Inception_A()
        self.Inception2=Inception_A()
        self.Inception3=Inception_A()
        self.Inception4=Inception_A()
        self.reduction_A=Inception_redution_A()
        self.Inception_b1=Inception_B()
        self.Inception_b2=Inception_B()
        self.Inception_b3=Inception_B()
        self.Inception_b4=Inception_B()
        self.Inception_b5=Inception_B()
        self.Inception_b6=Inception_B()
        self.Inception_b7=Inception_B()
        self.reduction_B=Inception_redution_B()
        self.Inception_c1=Inception_C()
        self.Inception_c2=Inception_C()
        self.Inception_c3=Inception_C()
        self.avgpool=keras.layers.AveragePooling2D(padding="valid",strides=1,pool_size=1)
        self.droupout=keras.layers.Dropout(0.2)
        self.fl=keras.layers.Flatten()
        self.soft=keras.layers.Dense(2,activation="softmax")
        self.bn=keras.layers.BatchNormalization()
        self.bn1=keras.layers.BatchNormalization()
        self.bn2=keras.layers.BatchNormalization()
        self.bn3=keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()
        #self.reshape=keras.layers.Reshape([])
    def call(self, inputs, **kwargs):
        x=self.stem(inputs)
        x=self.Inception1(x)
        x=self.Inception2(x)
        x=self.Inception3(x)
        x=self.Inception4(x)
        x=self.reduction_A(x)
        x=self.bn(x)
        x=self.Inception_b1(x)
        x=self.Inception_b2(x)
        x=self.Inception_b3(x)
        x=self.Inception_b4(x)
        x=self.bn1(x)
        x=self.Inception_b5(x)
        x=self.Inception_b6(x)
        x=self.Inception_b7(x)
        x=self.reduction_B(x)
        x=self.bn2(x)
        x=self.Inception_c1(x)
        x=self.Inception_c2(x)
        x=self.Inception_c3(x)
        x=self.bn3(x)
        x=self.avgpool(x)
        x=self.droupout(x)
        x=self.fl(x)
        x=self.bn4(x)
        #print(x.shape)
        x=self.soft(x)
        #x=self.reshape(x) #这个函数的作用是使其于标签属性一致
        #print(x)
        return x

    # def get_config(self):
    #     config = {
    #         'attention_dim': self.attention_dim
    #     }
    #     base_config = super(Inception, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))


def train_step(images,labels):
    # criteon = losses.categorical_crossentropy
    # loss_object = criteon
    # optimizer = optimizers.Adam(lr=0.001)
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
if __name__ == '__main__':
    dataest=dogandcat()
    dataset1 = dogandcat1()
    loss_object = losses.categorical_crossentropy
    acc_meter = metrics.CategoricalAccuracy()
    acc_meter1 = metrics.CategoricalAccuracy()
    optimizer = optimizers.Adam(lr=0.001)
    model=tf.keras.Sequential([
        Inception(),
    ])
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        metrics=['accuracy']
    )
    model.build(input_shape=(None,299,299,3))
    model.summary()
    # for epoch in range(10):
    #     for x,y in dataest:
    #         print(x.shape)
    #         print(y.shape)
    #         with tf.GradientTape() as tape:
    #             predictions = model(x)
    #             acc_meter1.update_state(y_true=y, y_pred=predictions)
    #             loss = loss_object(y, predictions)
    #             #loss1(y_true=y, y_pred=predictions)
    #         gradients = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #         # 打印准确率
    #         print("Test Accuracy:%f" % acc_meter.result())
    #         print("epoch{} train_loss is {};train_accuracy is {};test_accuracy is {}".format(epoch + 1,
    #                                                                                      loss[0],
    #                                                                                      acc_meter1.result(),
    #                                                                                      acc_meter.result(),
    #                                                                                          ))
    #     for x1, y1 in dataset1:  # 遍历测试集
    #         pred = model(x1)  # 前向计算
    #         acc_meter.update_state(y_true=y1, y_pred=pred)  # 更新准确率统计

    model.fit(dataest,epochs=2,batch_size=10)
    tf.saved_model.save(model, 'model-savedmodel')



```

![在这里插入图片描述](https://img-blog.csdnimg.cn/dff1f2940147469f90ad738cfc48b562.png#pic_center)


[github地址:](https://github.com/hideonpython/cat-and-dog-Inception4-)