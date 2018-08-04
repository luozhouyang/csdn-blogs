## Logistic Regression
逻辑回归(Logistic regression)是一种**分类算法**，名字包含**regression**是由于历史原因。

Sigmod function和Logistic function基本上是一回事。特点是函数值取值范围在[0,1]之间。

函数如下：

![](http://latex.codecogs.com/gif.latex?g(z)=\frac{1}{1+e^{-z}})


联合Hypothesis函数：

![](http://latex.codecogs.com/gif.latex?h_{\\theta}(x)=g(\\Theta^Tx))

可以得到：

![](http://latex.codecogs.com/gif.latex?h_\\theta(x)=\\frac{1}{1+e^{-\\Theta^Tx}})


训练过程就是用训练集不断**拟合![](http://latex.codecogs.com/gif.latex?$$\Theta$$)参数**。

Hypothesis函数的输出，就是对于输入x的一个预测概率。
对于二分类问题：

![](http://latex.codecogs.com/gif.latex?h_\\theta(x)=P(y=1|x;\\Theta))

Hypothesis函数的输出是，对于每一个输入特征x，分类结果y属于类别1的概率，这个概率是h关于![](http://latex.codecogs.com/gif.latex?$$\Theta$$)的函数。

类似的

![](http://latex.codecogs.com/gif.latex?h_\\theta(x)=P(y=0|x;\\Theta))

表示，对于输入特征x，分类结果y属于类别0的概率。

上述两种分类的概率和应该是１。

![](http://latex.codecogs.com/gif.latex?)
## Hypothesis函数
Hypothesis函数即**假设函数**。
要判断类别属于0还是1，需要一个**决策边界(Decision boundary)**

将Hypothesis函数展开，可以得到一下公式（举个例子）：

![](http://latex.codecogs.com/gif.latex?h_\\theta(x)=g(\\Theta_0+\\Theta_1x_1+\\Theta_2x_2))

因此，实际上，使用训练集训练的过程，就是在用输入**张量x**和**张量![](http://latex.codecogs.com/gif.latex?$$\\theta$$)** 进行矩阵乘法，然后计算loss。为了使loss最小，需要不断调整**张量![](http://latex.codecogs.com/gif.latex?$$\\theta$$)** 的值，这就是拟合的过程。

**决策边界**是**Hypothesis函数的属性**，不是训练集的属性。一旦Hypothesis函数的![](http://latex.codecogs.com/gif.latex?$$\\Theta$$)参数确定，就有了确定的**决策边界**。

为了能够处理更复杂的分类，我们必须让决策边界变得更复杂，办法就是**给函数g增加多项式**。

前面 

![](http://latex.codecogs.com/gif.latex?g(\\theta_0+\\theta_1x_1+\\theta_2x_2))

的决策边界是直线，如果我们需要决策边界是一个**圆形**，那么就需要拓展多项式，如下所示：

![](http://latex.codecogs.com/gif.latex?g(\\theta_0+\\theta_1x_1+\\theta_2x_2+\\theta_3x_1^2+\\theta_4x_2^2))


以此类推，我们可以得到更加复杂的决策边界。也就是说**更加复杂的决策边界需要更加高阶的多项式**。

## 如何拟合参数
训练的过程就是拟合![](http://latex.codecogs.com/gif.latex?$$\\Theta$$)参数的过程。

首先需要定义loss function(cost function)。
损失函数是关于![](http://latex.codecogs.com/gif.latex?$$\\Theta$$)的函数，表示成![](http://latex.codecogs.com/gif.latex?$$J(\Theta)$$)。

损失函数的选择非常多样，比如可以参考**线性回归**定义一个简单的函数，用这个函数来度量损失：

![](http://latex.codecogs.com/gif.latex?Cost(h_\\theta(x),y)=\\frac{1}{2}(h_\\theta(x)-y)^2)

这是一个方差函数。

但是对于逻辑回归(Logistic regression)，这个函数很容易造成![](http://latex.codecogs.com/gif.latex?$$J(\Theta)$$)成为一个**非凸函数**。这也就意味着，**损失函数肯能存在非常多的局部最优值**，这样就无法使用**梯度下降算法**求得全局最优。

注：
非凸函数，举个例子，**笑脸**改成波浪线。

所以，**选取一个合适的损失函数是很关键的**。

定义好了损失函数，我们需要让损失函数最小，比如使用梯度下降的方法。当损失最小的时候，我们就确定了![](http://latex.codecogs.com/gif.latex?$$\Theta$$)参数，也就确定了**决策边界**。这个降低loss，调整![](http://latex.codecogs.com/gif.latex?$$\Theta$$)参数的过程，就是**拟合(fitting)**。
