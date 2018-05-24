# tensorflow nmt的数据预处理过程　　
在tensorflow/nmt项目中，训练数据和推断数据的输入使用了新的**Dataset API**，应该是tensorflow 1.2之后引入的API，方便数据的操作。如果你还在使用老的Queue和Coordinator的方式，建议升级高版本的tensorflow并且使用Dataset API。    

本教程将从**训练数据**和**推断数据**两个方面，详解解析数据的具体处理过程，你将看到**文本数据**如何转化为模型所需要的**实数**，以及中间的**张量的维度**是怎么样的，**batch_size**和其他**超参数**又是如何作用的。  

## 训练数据的处理  
先来看看训练数据的处理。训练数据的处理比推断数据的处理稍微复杂一些，弄懂了训练数据的处理过程，就可以很轻松地理解推断数据的处理。  
**训练数据**的处理代码位于**nmt/utils/iterator_utils.py**文件内的`get_iterator`函数。  

### 函数的参数  

我们先来看看这个函数所需要的参数是什么意思:  

|参数|解释|  
|----|----|  
|`src_dataset`|源数据集|  
|`tgt_dataset`|目标数据集|  
|`src_vocab_table`|源数据单词查找表，就是个单词和int类型数据的对应表|  
|`tgt_vocab_table`|目标数据单词查找表，就是个单词和int类型数据的对应表|  
|`batch_size`|批大小|  
|`sos`|句子开始标记|  
|`eos`|句子结尾标记|  
|`random_seed`|随机种子，用来打乱数据集的|  
|`num_buckets`|桶数量|  
|`src_max_len`|源数据最大长度|  
|`tgt_max_len`|目标数据最大长度|  
|`num_parallel_calls`|并发处理数据的并发数|  
|`output_buffer_size`|输出缓冲区大小|  
|`skip_count`|跳过数据行数|  
|`num_shards`|将数据集分片的数量，分布式训练中有用|  
|`shard_index`|数据集分片后的id|  
|`reshuffle_each_iteration`|是否每次迭代都重新打乱顺序|  

上面的解释，如果有不清楚的，可以查看我之前一片介绍超参数的文章：  
[tensorflow_nmt的超参数详解](tensorflow_nmt的超参数.md)  

我们首先搞清楚几个重要的参数是怎么来的。  
`src_dataset`和`tgt_dataset`是我们的训练数据集，他们是逐行一一对应的。比如我们有两个文件`src_data.txt`和`tgt_data.txt`分别对应训练数据的**源数据**和**目标**数据，那么它们的**Dataset**如何创建的呢？其实利用**Dataset API**很简单:  
```python  
src_dataset=tf.data.TextLineDataset('src_data.txt')  
tgt_dataset=tf.data.TextLineDataset('tgt_data.txt')  
```  
这就是上述函数中的两个参数`src_dataset`和`tgt_dataset`的由来。  

`src_vocab_table`和`tgt_vocab_table`是什么呢？同样顾名思义，就是这两个分别代表**源数据词典的查找表**和**目标数据词典的查找表**，实际上查找表就是一个**字符串**到**数字**的映射关系。当然，如果我们的源数据和目标数据使用的是同一个词典，那么这两个查找表的内容是一模一样的。很容易想到，肯定也有一种**数字**到**字符串**的映射表，这是肯定的，因为神经网络的数据是数字，而我们需要的目标数据是字符串，因此它们之间肯定有一个转换的过程，这个时候，就需要我们的**reverse_vocab_table**来作用了。  

我们看看这两个表是怎么构建出来的呢？代码很简单，利用tensorflow库中定义的**lookup_ops**即可：  
```python  
def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  src_vocab_table = lookup_ops.index_table_from_file(
      src_vocab_file, default_value=UNK_ID)
  if share_vocab:
    tgt_vocab_table = src_vocab_table
  else:
    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)
  return src_vocab_table, tgt_vocab_table
```  
我们可以发现，创建这两个表的过程，就是将词典中的每一个词，对应一个数字，然后返回这些数字的集合，这就是所谓的词典查找表。**效果上来说，就是对词典中的每一个词，从0开始递增的分配一个数字给这个词**。  

那么到这里你有可能会有疑问，我们**词典中的词**和我们**自定义的标记`sos`等**是不是有可能被映射为同一个整数而造成冲突？这个问题该如何解决？聪明如你，这个问题是存在的。那么我们的项目是如何解决的呢？很简单，那就是将我们**自定义的标记**当成**词典的单词**，然后**加入到词典文件中**，这样一来，`lookup_ops`操作就把标记当成单词处理了，也就就解决了冲突！  

具体的过程，本文后面会有一个例子，可以为您呈现具体过程。  
如果我们指定了`share_vocab`参数，那么返回的源单词查找表和目标单词查找表是一样的。我们还可以指定一个**default_value**，在这里是`UNK_ID`，实际上就是`0`。如果不指定，那么默认值为`-1`。这就是查找表的创建过程。如果你想具体的知道其代码实现，可以跳转到tensorflow的C++核心部分查看代码（使用PyCharm或者类似的IDE）。  
  

### 数据集的处理过程  

该函数处理训练数据的主要代码如下:  
```python  
if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

```  
我们逐步来分析，这个过程到底做了什么，**数据张量**又是如何变化的。  

我们知道，对于源数据和目标数据，每一行数据，我们都可以使用一些标记来表示数据的开始和结束，在本项目中，我们可以通过`sos`和`eos`两个参数指定句子**开始标记**和**结束标记**，默认值分别为**<s>**和**</s>**。本部分代码一开始就是将这两个句子标记表示成一个整数，代码如下：  
```python
src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)
```  
过程很简单，就是通过两个**字符串到整形**的**查找表**，根据`sos`和`eos`的字符串，找到对应的整数，用改整数来表示这两个标记，并且将这两个整数转型为int32类型。  
接下来做的是一些常规操作，解释如注释：  
```python  
# 通过zip操作将源数据集和目标数据集合并在一起
# 此时的张量变化 [src_dataset] + [tgt_dataset] ---> [src_dataset, tgt_dataset]
src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
# 数据集分片，分布式训练的时候可以分片来提高训练速度
src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
if skip_count is not None:
  # 跳过数据，比如一些文件的头尾信息行
src_tgt_dataset = src_tgt_dataset.skip(skip_count)
# 随机打乱数据，切断相邻数据之间的联系
# 根据文档，该步骤要尽早完成，完成该步骤之后在进行其他的数据集操作
src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)
```   
  
接下来就是重点了，我将用注释的形式给大家解释：  
```python  
  # 将每一行数据，根据“空格”切分开来
  # 这个步骤可以并发处理，用num_parallel_calls指定并发量
  # 通过prefetch来预获取一定数据到缓冲区，提升数据吞吐能力
  # 张量变化举例 ['上海　浦东', '上海　浦东'] ---> [['上海', '浦东'], ['上海', '浦东']]
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # 过滤掉长度为0的数据
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
　# 限制源数据最大长度
  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
　# 限制目标数据的最大长度
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # 通过map操作将字符串转换为数字
  # 张量变化举例 [['上海', '浦东'], ['上海', '浦东']] ---> [[1, 2], [1, 2]]
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # 给目标数据加上 sos, eos　标记
  # 张量变化举例 [[1, 2], [1, 2]] ---> [[1, 2], [sos_id, 1, 2], [1, 2, eos_id]]
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # 增加长度信息
  # 张量变化举例 [[1, 2], [sos_id, 1, 2], [1, 2, eos_id]] ---> [[1, 2], [sos_id, 1, 2], [1, 2, eos_id], [src_size], [tgt_size]]
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
```  
其实到这里，基本上数据已经处理好了，可以拿去训练了。但是有一个问题，那就是我们的每一行数据长度大小不一。这样拿去训练其实是需要很大的运算量的，那么有没有办法优化一下呢？有的，那就是**数据对齐处理**。  

### 如何对齐数据  
数据对齐的代码如下，使用注释的方式来解释代码：  
```python  
# 参数x实际上就是我们的 dataset 对象
def batching_func(x):
    # 调用dataset的padded_batch方法，对齐的同时，也对数据集进行分批
    return x.padded_batch(
        batch_size,
        # 对齐数据的形状
        padded_shapes=(
            # 因为数据长度不定，因此设置None
            tf.TensorShape([None]),  # src
            # 因为数据长度不定，因此设置None
            tf.TensorShape([None]),  # tgt_input
            # 因为数据长度不定，因此设置None
            tf.TensorShape([None]),  # tgt_output
            # 数据长度张量，实际上不需要对齐
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # 对齐数据的值
        padding_values=(
            # 用src_eos_id填充到 src 的末尾
            src_eos_id,  # src
            # 用tgt_eos_id填充到 tgt_input 的末尾
            tgt_eos_id,  # tgt_input
            # 用tgt_eos_id填充到 tgt_output 的末尾
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused
```  
这样就完成了数据的对齐，并且将数据集按照`batch_size`完成了分批。  

### num_buckets分桶到底起什么作用  
`num_buckets`起作用的代码如下：　　
```python   
  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  else:
    batched_dataset = batching_func(src_tgt_dataset)
```  

`num_buckets`顾名思义就是**桶的数量**，那么这个桶用来干嘛呢？我们先看看上面两个函数到底做了什么。  
首先是判断我们指定的参数`num_buckets`是否大于１，如果是那么就进入到上述的作用过程。    

`key_func`是做什么的呢？通过源码和注释我们发现，它是用来将我们的数据集(由**源数据**和**目标数据**成对组成)按照一定的方式进行分类的。具体说来就是，根据我们数据集每一行的数据长度，将它放到合适的**桶**里面去，然后返回该数据所在桶的索引。 
   
这个**分桶**的过程很简单。假设我们有一批数据，他们的长度分别为`3 8 11 16 20 21`，我们规定一个**bucket_width**为**10**，那么我们的数据分配到具体的桶的情况是怎么样的呢？因为桶的宽度为10，所以第一个桶放的是小于长度10的数据，第二个桶放的是10-20之间的数据，以此类推。  

所以，要进行分桶，我们需要知道**数据**和**bucket_width**两个条件。然后根据一定的简单计算，即可确定如何分桶。上述代码首先根据`src_max_len`来计算`bucket_width`，然后分桶，然后返回数据分到的桶的索引。就是这么简单的一个过程。 

那么，你或许有疑问了，我干嘛要分桶呢？你仔细回想下刚刚的过程，是不是发现**长度差不多的数据都分到相同的桶里面去了**！没错，这就是我们分桶的目的，**相似长度的数据放在一起，能够提升计算效率**！！！  

然后要看第二个函数`reduce_func`，这个函数做了什么呢？其实就做了一件事情，就是把刚刚分桶好的数据，做一个对齐！！！  

那么通过**分桶**和**对齐**操作之后，我们的数据集就已经成为了一个对齐（也就是说有固定长度）的数据集了！  

回到一开始，如果我们的参数`num_bucktes`不满足条件呢？那就直接做对齐操作！看代码便知！  
至此，分桶的过程和作用你已经清楚了。  

---  
至此，数据处理已经结束了。接下来就可以从处理好的数据集获取一批一批的数据来训练了。  
那么如何一批一批获取数据呢？答案是使用**迭代器**。获取Dataset的迭代器很简单，tensorflow提供了API，代码如下：  
```python  
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())
```  
通过迭代器的`get_next()`方法，就可以获取之前我们处理好的批量数据啦！  

tensorflow nmt的数据预处理过程已经结束了。欢迎和我交流。