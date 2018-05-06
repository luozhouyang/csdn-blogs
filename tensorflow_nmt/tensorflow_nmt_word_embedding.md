# TensorFlow NMT的词嵌入(Word Embeddings)  
> 声明：本文由 罗周杨 stupidme.me.lzy@gmail.com 原创，未经授权不得转载  

自然语言处理的第一步，就是要将文本表示成计算机能理解的方式。我们将长文本分词之后，得到一个词典，对于词典中的每一个词，我们用一个或者一组数字来表示它们。这样就实现了我们的目标。  
但是用数字表示文本，这其中也有很大的想象空间。你可能会想到，对词典中的每一个词，我都给一个整数表示。举个具体的例子，假设词典大小为5，我们给每个词编码分别为00001, 00010, 00100, 01000, 10000。这种想法不错，这就是我们熟悉的one-hot编码。在某些应用上，这种表示没有问题。但是普遍的问题上，这样的表示方法会有以下几个问题：  
* 如果词典数量太大，则会造成维度爆炸  
* 这样的表示没办法表示词语之间的联系  

因此，业界想了很多其他的表示方法，后来也就出现了**词嵌入(Word Embedding)** 的概念。词嵌入有很多模型，比如**word2vec**,**GloVe**等等。本片文章将带领大家完整得理解TensorFlow NMT项目中的词嵌入过程。  

TensorFlow NMT的词嵌入代码入口位于**nmt/model.py**文件，**BaseModel**有一个`init_embeddings()`方法，NMT模型就是在此处完成词嵌入的初始化的。但是具体的代码，分散在多个方法和帮助函数，为了直观，我重新整理了一下代码。两者的流程是完全保持一致的。  

## 从超参数获取需要的参数  
需要做词嵌入，则首先要获取需要的信息。比如词典文件，或者说词嵌入文件（如果已经有训练好的词嵌入文件的话）。这些信息，都是通过超参数hparams这个参数传递过来的。主要的参数获取如下：  
```python  
    def _init_embeddings(self, hparams, scope):
        # 源数据和目标数据是否使用相同的词典
        share_vocab = hparams.share_vocab
        src_vocab_size = self.src_vocab_size
        tgt_vocab_size = self.tgt_vocab_size
        # 源数据词嵌入的维度，数值上等于指定的神经单元数量
        src_embed_size = hparams.num_units
        # 目标数据词嵌入的维度，数值上等于指定的神经单元数量
        tgt_embed_size = hparams.num_units
        # 词嵌入分块数量，分布式训练的时候，需要该值大于1
        num_partitions = hparams.num_embeddings_partitions
        # 源数据的词典文件
        src_vocab_file = hparams.src_vocab_file
        # 目标数据的词典文件
        tgt_vocab_file = hparams.tgt_vocab_file
        # 源数据已经训练好的词嵌入文件
        src_embed_file = hparams.src_embed_file
        # 目标数据已经训练好的词嵌入文件
        tgt_embed_file = hparams.tgt_embed_file

        # 分块器，用于分布式训练
        if num_partitions <= 1:
            # 小于等于1，则不需要分块，不使用分布式训练
            partitioner = None
        else:
            # 分块器也是一个张量，其值大小和分块数量一样
            partitioner = tf.fixed_size_partitioner(num_partitions)

        # 如果使用分布式训练，则不能使用已经训练好的词嵌入文件
        if (src_embed_file or tgt_embed_file) and partitioner:
            raise ValueError(
                "Can't set num_partitions > 1 when using pretrained embedding")
```  
参数的意义我已经写在注释里面了。  
获取到这些参数之后，我们就可以创建或者加载词嵌入的矩阵表示了。  

## 创建或者加载词嵌入矩阵  
根据超参数，如果提供了**预训练**的词嵌入文件，则我们只需要根据词典，将词典中的词的嵌入表示，从词嵌入文件取出来即可。如果没有提供预训练的词嵌入文件，则我们自己创建一个即可。  
```python  
        # 创建词嵌入的变量域
        with tf.variable_scope(scope or "embeddings", dtype=tf.float32, partitioner=partitioner) as scope:
            # 如果共享词典
            if share_vocab:
                # 检查词典大小是否匹配
                if src_vocab_size != tgt_vocab_size:
                    raise ValueError("Share embedding but different src/tgt vocab sizes"
                                     " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
                assert src_embed_size == tgt_embed_size
                vocab_file = src_vocab_file or tgt_vocab_file
                embed_file = src_embed_file or tgt_embed_file
                # 如果有训练好的词嵌入模型，则直接加载，否则创建新的
                embedding_encoder = self._create_or_load_embed(
                    "embedding_share", vocab_file, embed_file,
                    src_vocab_size, src_embed_size, dtype=tf.float32)
                embedding_decoder = embedding_encoder
            # 不共享词典的话，需要根据不同的词典创建对应的编码器和解码器
            else:
                # 加载或者创建编码器
                with tf.variable_scope("encoder", partitioner=partitioner):
                    embedding_encoder = self._create_or_load_embed(
                        "embedding_encoder", src_vocab_file, src_embed_file,
                        src_vocab_size, src_embed_size, tf.float32)
                # 加载或创建解码器
                with tf.variable_scope("decoder", partitioner=partitioner):
                    embedding_decoder = self._create_or_load_embed(
                        "embedding_decoder", tgt_vocab_file, tgt_embed_file,
                        tgt_vocab_size, tgt_embed_size, tf.float32)
            self.embedding_encoder = embedding_encoder
            self.embedding_decoder = embedding_decoder
```  
如你所见，在获取词嵌入表示之前，有一个share_vocab的判断。这个判断也很简单，就是判断源数据和目标数据是否使用相同的词典，不管是不是share_vocab，最后都需要创建或者加载词嵌入表示。这个关键的过程在`_create_or_load_embed()`函数中完成。  
该函数的主要工作如下：  
```python  
    def _create_or_load_embed(self, embed_name, vocab_file, embed_file, vocab_size, embed_size, dtype=tf.float32):
        # 如果提供了训练好的词嵌入文件，则直接加载
        if vocab_file and embed_file:
            embedding = self._create_pretrained_emb_from_txt(vocab_file, embed_file)
        else:
            # 否则创建新的词嵌入
            with tf.device(self._get_embed_device(vocab_size)):
                embedding = tf.get_variable(
                    embed_name, [vocab_size, embed_size], dtype)
        return embedding
```  

### 加载预训练的词嵌入表示  
如果超参数提供了embed_file这个预训练好的词嵌入文件，那么我么只需要读取该文件，创建出词嵌入矩阵，返回即可。  
主要代码如下：  
```python  
    def _create_pretrained_emb_from_txt(self, vocab_file, embed_file,
                                        num_trainable_tokens=3, dtype=tf.float32, scope=None):
        """
        从文件加载词嵌入矩阵
        :param vocab_file: 词典文件
        :param embed_file: 训练好的词嵌入文件
        :param num_trainable_tokens:词典文件前3个词标记为变量，默认为"<unk>","<s>","</s>"
        :param scope: 域
        :return: 词嵌入矩阵
        """
        # 加载词典
        vocab, _ = vocab_utils.load_vocab(vocab_file)
        # 词典的前三行会加上三个特殊标记，取出三个特殊标记
        trainable_tokens = vocab[:num_trainable_tokens]

        utils.print_out("# Using pretrained embedding: %s." % embed_file)
        utils.print_out("  with trainable tokens: ")

        # 加载训练好的词嵌入
        emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file)
        for token in trainable_tokens:
            utils.print_out("    %s" % token)
            # 如果三个标记不在训练好的词嵌入中
            if token not in emb_dict:
                # 初始化三个标记为0.0，维度为词嵌入的维度
                emb_dict[token] = [0.0] * emb_size

        # 从训练好的词嵌入矩阵中，取出词典中的词语的词嵌入表示，数据类型为tf.float32
        emb_mat = np.array(
            [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
        # 常量化词嵌入矩阵
        emb_mat = tf.constant(emb_mat)
        # 从词嵌入矩阵的第4行之后的所有行和列（因为num_trainable_tokens=3)
        # 也就是说取出除了3个标记之外所有的词嵌入表示
        # 这是常量，因为已经训练好了，不需要训练了
        emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
        with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
            with tf.device(self._get_embed_device(num_trainable_tokens)):
                # 获取3个标记的词嵌入表示，这3个标记的词嵌入是可以变的，通过训练可以学习
                emb_mat_var = tf.get_variable(
                    "emb_mat_var", [num_trainable_tokens, emb_size])
        # 将3个标记的词嵌入和其余单词的词嵌入合并起来，得到完整的单词词嵌入表示
        return tf.concat([emb_mat_var, emb_mat_const], 0)
```  
处理过程，我已经在注释里面写得很清楚了。接下来看看新创建词嵌入表示的过程。  

### 重新创建词嵌入表示   
这个过程其实很简单，就是创建一个可训练的张量而已：  
```python  
with tf.device(self._get_embed_device(vocab_size)):
    embedding = tf.get_variable(embed_name, [vocab_size, embed_size], dtype)  
```  
该张量的名字就是`embed_name`，shape即[vocab_size, embed_size]，其中`vocab_size`就是词典的大小，也就是二维矩阵的行数，`embed_size`就是词嵌入的维度，每个词用多少个数字来表示，也就是二维矩阵的列数。该张量的数据类型是单精度浮点数。当然，`tf.get_variable()`方法还有很多提供默认值的参数，其中一个就是`trainable=True`，这代表这个变量是可变的，也就是我们的词嵌入表示在训练过程中，数字是会改变的。  

这样就完成了词嵌入的准备过程。  
