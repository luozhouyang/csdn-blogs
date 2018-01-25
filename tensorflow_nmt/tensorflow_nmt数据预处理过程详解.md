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

`src_vocab_table`和`tgt_vocab_table`是什么呢？

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



### 如何对齐数据  


### num_buckets到底起什么作用  
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

