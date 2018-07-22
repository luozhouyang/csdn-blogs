# 《Fluency Boost Learning and Inference for Neural Grammatical Error Correction》论文总结  

今天看到微软亚洲研究院的一篇论文，通过**Fluency boost learning**提升模型性能，论文地址为: [Fluency Boost Learning and Inference for Neural Grammatical Error Correction](http://aclweb.org/anthology/P18-1097)，有兴趣的同学可以去下载看看。在此我总结了一下这篇论文。  

## 核心思想  
这篇论文的核心思想其实很简单，就是通过有效地增加训练数据，来使模型的推断结果更加正确。具体就是使用模型推断的n-best结果来生成新的训练数据，用于训练。

增加训练数据这个步骤是很关键的。  

### 传统的做法
想到增加训练数据，一个很正常的想法就是，人为制造一些含有错误信息的训练数据对。操作步骤为：  

* 从训练数据对dataset中选取训练数据对，即(src, tgt)
* 合理修改src中的字符，变成src'
* 修改之后的src'与tgt组成一个新的数据对
* 重复上述步骤若干次，得到不少新的训练数据对dataset'
* 将dataset和dataset'一起用于模型训练。  

### 论文的做法  
但是本论文的做法不同。它的想法其实也挺正常。具体的做法是：  

* 对每一个src，使用模型推断，得到多个推断结果(n-best)  
* 对每一个推断结果，计算一个flunecy分数  
* 抽取出所有分数低于正确推断结果(认为是n-best的第一个)的推断结果  
* 对于选取出的每一个推断结果，与tgt句子组成新的训练数据对，叫做**fluency boost sentence pair**，这些数据对用于后续的训练  

上述做法就是论文的做法。这种做法与传统的增加数据的做法相比，有一个明显的优势就是：  

* 模型推断的结果，更能反映当前模型的信息，用它来反馈给模型，能够更加有效地纠正模型。  

因此，个人觉得这种做法训练出的模型性能要优于传统的增加训练数据的做法。  
并且，使用fluency boost learning可以多回合进行逐步纠错，在连续错误的情况下，能够逐步纠正词语，使得整个推断流程的词语上下文变得清晰。  

## 几个要点  
论文有几个要点，如下：

* 如何计算fluency分数？  
* fluency boost learning也有多种类型

### 计算fluency分数
fluency分数的计算很简单，公式如下：  
[![](https://i.loli.net/2018/07/22/5b540cd132999.png)](https://i.loli.net/2018/07/22/5b540cd132999.png)

其中`x`代表句子，`f(x)` 即 `fluency score`，`H(x)`即`x`的`交叉熵`。

### fluency boost 的种类  
fluency boost leanring 有三种方式：

* Back-boost learning
* Self-boost learning
* Dual-boost learning

**Back-boost**借鉴于NMT的**Back translation**，是讲一个流畅的句子转换成一个含有错误的句子。论文给了一个伪代码：  
[![](https://i.loli.net/2018/07/22/5b54109f0ac9f.png)](https://i.loli.net/2018/07/22/5b54109f0ac9f.png)

**Self-boost**允许模型自己生成候选结果。论文的伪代码如下：  
[![](https://i.loli.net/2018/07/22/5b541129855d2.png)](https://i.loli.net/2018/07/22/5b541129855d2.png)

`back-boost`和`self-boost`是从不同的层面生成不流畅的句子用于提升模型的性能。**Dual-boost**则是两者的结合。伪代码如下：  
[![](https://i.loli.net/2018/07/22/5b54122dd4d7a.png)](https://i.loli.net/2018/07/22/5b54122dd4d7a.png)

然后论文还给出了一些数据测试结果对比，有兴趣的可以通过文章开头的论文链接，下载论文查看。

目前，还不知道哪里有开源的实现。或许你可以试着自己去实现一个。嘿嘿。

注：

* 论文指出NEC不同于NMT，NEC的目标是不改变原句子的意思的前提下使句子更流畅。  

## 联系我
* Email: [stupidme.me.lzy@gmail.com](mailto:stupidme.me.lzy@gmail.com)  
* WeChat: luozhouyang0528  

[![](https://i.loli.net/2018/07/22/5b54132b35d72.jpg)](https://i.loli.net/2018/07/22/5b54132b35d72.jpg)

个人公众号，你可能会有兴趣：  
[![](https://i.loli.net/2018/07/22/5b54155aa1b1c.jpg)](https://i.loli.net/2018/07/22/5b54155aa1b1c.jpg)
