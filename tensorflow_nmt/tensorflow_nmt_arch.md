# Tensorflow nmt的整体结构　　

在解析具体的代码之前，我们需要对NMT的整个结构有个大致的印象。

其实NMT结构不复杂，它是一种典型的**sequence-to-sequence**结构，所以呢，它包含以下几个部分：

* Embedding(词嵌入)
* Encode(编码)
* Decode(解码)

整体的流程与上面是对应的：

* 通过Embedding将文字序列src、tgt分别转换成数字表示
* 使用Encoder将序列src进行编码，得到encoder的 **输出(encoder_outputs)**和 **状态(encoder_state)**
* 使用Decoder，将序列tgt，以及encoder_outputs, encoder_state作为输入，进行解码，得到decoder的logits, sample_id, final_context_state这些信息
* logits用来计算loss，然后减少loss，优化模型
* sample_id用来转换成文本，也就是得到了一个序列src对应的输出
* final_context_state是解码器的输出

以上就是NMT(其实，几乎所有的seq2seq模型)的主要流程。

下面来更加详细一点地解释以下。

### Embedding(词嵌入)
首先，**Embedding**即**词嵌入**，它的作用是什么呢？很简单，**讲文本信息转化成数字**。因为计算机无法直接处理文字，所以我需要 **将文字转化成数字**这一个技术。

我已经在另一篇单独的文章中详细介绍了 **Embedding**，请转到[TensorFlow NMT的Word Embedding](tensorflow_nmt_word_embedding.md)

### Encoder(编码器)
编码器的大致流程，上面已经说了。但是我们还是不知道这个编码器到底是啥？或者说它长啥样？

其实，很简单，**它就是多层RNN组成的网络而已**。

回顾一下，我们的超参数中有`--num_encoder_layers`这个参数，这个就是决定我们这个编码气有多少层的。对于每一层，我们的RNN单元又是什么样子的呢？

这个和我们的超参数设置有关系。但是总的来说，就是每一个神经元都是一个RNN Cell。根据超参数的设计，它可以使普通的RNN Cell，也可以是Bi-directional的RNN Cell。当然，你还可以给这些RNN Cell增加一些包装，比如加一个DropoutWrapper。整个编码器的网络，也可以有一定的Residual network(残差网络)。

就这么简单。

### Decoder(解码器)
解码器和上面的编码器几乎一样。但是有一个值得注意的是，我们的attention机制。

我们的attention机制其实是作用在decoder这部分的。通过源码我们就会发现，普通的模型和attention模型，在encode上几乎没啥区别，最大的区别就是在于decode阶段，加入了attention mechainism。

总结一下，我们NMT的整个流程，就是这么几个步骤，一点都不复杂。但是tensorflow/nmt项目的代码组织不是很好，对于新手来说，比较难区分开来。大家可以看我重构之后的代码，这个代码就非常清晰了[luozhouyang/naivenmt](https://github.com/luozhouyang/naivenmt)。
