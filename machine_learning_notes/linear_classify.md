如果我们使用神经网络来分类，那么这种方式有两个重要的组件：

* Score funtion: 将原生数据(raw data)映射到一个类别的分数(class scores)
* Loss function: 量化预测的结果和事实上的结果之间的相似程度

线性分类器(Linear classifier)就是一个简单的**Score function**：

![](http://latex.codecogs.com/gif.latex?f(x_{i},W,b)=Wx_{i}+b)

其中：

* W表示**权重(weights)**
* b表示**偏置(bias)**，对结果有影响，但是和输入  x没有联系

## Linear classifier的解释
那么该如何解释这个公式呢？
主要有两种分解释。

### 高维点解释

首先，将图像类比成高维的点，每张图片变成一个点(例如CIFAR数据集中，这个点是32*32*3的空间)，对于![](http://latex.codecogs.com/gif.latex?f(x_{i},W,b)=Wx_{i}+b)
则成了K\*D(即10*3072)的一个列向量，W的每一行，都是一个**分类器**。

也就是说，$$W$$本身是一个**分类器**集合，在二维坐标系上的表示就是：对于坐标系上的所有点，存在多条直线，每条直线都可以将这些点划分为两个类别。

这些直线的集合，就代表了![](http://latex.codecogs.com/gif.latex?W)
和![](http://latex.codecogs.com/gif.latex?b)。其中，直线的**斜率**决定于$$W$$，直线的**截距**决定于![](http://latex.codecogs.com/gif.latex?b)。

### 模板匹配解释
还一种解释就是模板匹配解释。对于![](http://latex.codecogs.com/gif.latex?W)
的每一行，都看成是一个固定的**模板**，对一个输入x进行分类，就是对![](http://latex.codecogs.com/gif.latex?W)
的每一行进行比较（实际上就是**点积**操作），从而找到最符合的一个，作为分类结果。

还有一种想法就是，这其实是一种更加高效的**Nearest Neighbor**，它不需要遍历每一个训练数据来计算距离，而是用输入数据和W的模板的点积来作为距离度量（替代了L1或者L2距离）。

### 关于bias的trick
我们可以把![](http://latex.codecogs.com/gif.latex?b)
这个列向量，合并到![](http://latex.codecogs.com/gif.latex?W)
的最后一列，对每一个输入，最后加上一行全为1的向量。

式子变成了：

![](http://latex.codecogs.com/gif.latex?f(x_{i},W,b)=Wx_{i})


其中：

* W的维度是[K+1,D+1]，用CIFAR数据集举例，K=10, D=32\32\3
* ![](http://latex.codecogs.com/gif.latex?x_{i})
的维度是[D+1, 1], D=32\*32\*3


实际上，**之前的矩阵乘法然后相加的计算量和我们这种做法的计算量相当**，但是我们这种做法有一个有效的副作用：**我们只需要计算一个矩阵（![](http://latex.codecogs.com/gif.latex?W)
），而不是两个（![](http://latex.codecogs.com/gif.latex?W)
,![](http://latex.codecogs.com/gif.latex?b)
**。

### 联系我

* Email: stupidme.me.lzy@gmail.com
* WeChat: luozhouyang0528

![](http://blog.stupidme.me/wp-content/uploads/2018/08/wechat.jpg)

个人公众号，你可能会感兴趣：

![](http://blog.stupidme.me/wp-content/uploads/2018/08/wechat_stupidmedotme_15.jpg)