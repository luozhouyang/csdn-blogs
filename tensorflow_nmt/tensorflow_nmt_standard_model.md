# Tensorflow nmt的基本模型

tensorflow/nmt的模型一共有以下几种：  
* 标准模型
* Attention机制的模型
* GNMT模型　　

在代码上，有如下继承关系：  
```bash
GNMTModel extends AttentionModel extends Model extends BaseModel
```  
我们从基本模型（即标准模型）开始讲起。为了把握整体的代码结构，我们直接从`BaseModel`的代码开始解释。  

## BaseModel.__init__()方法  
我们看看`BaseModel`在__init__()方法里面做了什么。  
首先看看该方法的参数:
|arguments|describe|
|:--------|:-------|
|hparams|超参数|
|mode|模式，train、eval、infer|
|iterator|数据迭代器|
|source_vocab_table|源数据的单词到ID的映射表|
|target_vocab_table|目标数据的单词到ID的映射表|
|reverse_target_vocab_table|是否构造反向映射表|
|scope|变量的域名|
|extra_args|额外的参数，如控制分布式训练的参数|

由于代码比较长，所以我们分段解释。
`__init__`方法做的事情主要是以下几件：
* 获取需要的超参数
* 初始化变量
* 初始化词嵌入
* 创建出图，也就是tensorflow的graph概念
* 选择优化器，更新梯度
* 使用Summary记录变量的变化
* 使用Saver保存模型参数

我们逐个解释代码片段。

### 获取超参数  
代码如下，具体解释请看代码中的注释。
```python
    # 数据迭代器，不管是train, eval还是infer，都有一个数据迭代器
    self.iterator = iterator
    # 记录模型的模式，train, eval, infer之一
    self.mode = mode
    # 源数据的词典单词到ID的映射表
    self.src_vocab_table = source_vocab_table
    # 目标数据的单词词典到ID的映射表
    self.tgt_vocab_table = target_vocab_table
    # 词典大小
    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    # GPU数量，分布式训练有用
    self.num_gpus = hparams.num_gpus
    # 是否是时间主要
    self.time_major = hparams.time_major

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # 设置编码器和解码器的层数
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    #　设置编码器和解码器的残差网络层数
    if hasattr(hparams, "num_residual_layers"):  # compatible common_test_utils
      self.num_encoder_residual_layers = hparams.num_residual_layers
      self.num_decoder_residual_layers = hparams.num_residual_layers
    else:
      self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
      self.num_decoder_residual_layers = hparams.num_decoder_residual_layers
```  
其中iterator在train，eval和infer过程中，不断地喂数据。想了解更多的DataSet和Iterator相关的细节，请看另一篇博客[tensorflow_nmt_dataset_process](tensorflow_nmt_dataset_process.md)。  

编码器和解码器的层数非常重要，对nmt的效果有很大的影响。一般来说，不应该少于**2**层。常用的是**4**层，甚至**6**层或者更多。  

**残差网络(Residual Network)** 的原理网上有很多文章，这里简述下作用：解决了深网络的梯度消失和梯度爆炸的问题，同时改善了网络的性能。  

### 初始化词嵌入(Word Embedding)  
关于词嵌入的过程，我已经写了另一篇博客。请查看[tensorflow_nmt_word_embedding](tensorflow_nmt_word_embedding.md)。

### 创建图(Graph)  
这是最重要的一个步骤了。



