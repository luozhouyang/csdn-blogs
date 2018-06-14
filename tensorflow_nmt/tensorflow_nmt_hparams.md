# Tensorflow nmt的超参数　　
超参数一般用来定义我们的神经网络的关键参数．　　

在tensorflow/nmt这个demo中，我们的超参数在 nmt.nmt 模块中配置．这也导致了nmt.py这个文件的代码行数比较多，我们完全可以把参数的配置放到单独的一个文件中去．**nmt.py** 这个文件也是整个项目的入口文件．如果你想了解这个demo的整体结构，请查看我的另一篇博客[tensorflow/nmt的整体结构](tensorflow_nmt的整体结构.md),　这就不展开了．　

下面我会列出nmt模型定义的超参数，并且追条解释，希望能加深你对这些参数的理解．　　

本demo的超参数使用的是argparse模块进行配置的，如果你喜欢，也可以使用tensorflow中的 **tf.app.flags.DEFINE_xxx()** 函数来配置，后者是前者的简单封装．　　

## 超参数列表　　
首先用表格的形式列出所有的超参数，对他们的解释放在下一小节．　　

|超参数(hparams)|类型(type)|默认值(default)|简介(help)|  
|---------------|----------|---------------|----------|  
|`--num_units`|`int`|32|network size|  
|`--num_layers`|`int`|2|network depth|  
|`--num_encoder_layers`|`int`|`None`|encoder depth, equal to `num_layers` if `None`|  
|`--num_decoder_layers`|`iny`|`None`|decoder depth, equal to `num_layers` if `None`|  
|`--encoder_type`|`str`|`uni`|one of `uni`, `bi`, `gnmt`|    
|`--residual`|`bool`|`False`|whether to add residual connections|  
|`--time_major`|`bool`|`True`|whether to add time-major mode for dynamic RNN|  
|`--num_embeddings_partitions`|`int`|0|number of partitions for embedding vars|  
|`--attention`|`str`|`""`|one of `""`, `luong`, `scaled_luong`, `bahdanau`, `normed_bahdanau`|  
|`--attention_architecture`|`str`|standard|one of `standard`, `gnmt`, `gnmt_v2`|  
|`--output_attention`|`bool`|`True`|only used in standard attention_architecture|  
|`--pass_hidden_state`|`bool`|`True`|whether to pass encoder's hidden state to decoder|  
|`--optimizer`|`str`|`sgd`|one of `sgd`, `adam`|  
|`--learning_rate`|`float`|1.0|adam: 0.001 or 0.0001|  
|`--warmup_steps`|`int`|0|how many steps we inverse-decay learning|  
|`--warmup_scheme`|`str`|`t2t`|how to warmup learning rates|  
|`--decay_scheme`|`str`|`""`|how we decay learning rate|  
|`--num_train_steps`|`int`|12000|num steps to train|  
|`--colocate_gradients_with_ops`|`bool`|`True`|whether try colocating gradients with corresponding op|  
|`--init_op`|`str`|`uniform`|one of `uniform`, `glorot_normal`, `glorot_uniform`|  
|`--init_weight`|`float`|0.1|for uniform init_op, initialize weights|  
|`--src`|`str`|`None`|source suffix|  
|`--tgt`|`str`|`None`|target suffix|  
|`--train_prefix`|`str`|`None`|train prefix|  
|`--dev_prefix`|`str`|`None`|dev prefix|  
|`--test_prefix`|`str`|`None`|test prefix|  
|`--out_dir`|`str`|`None`|model folder|  
|`--vocab_prefix`|`str`|`None`|vocab prefix|  
|`--emded_prefix`|`str`|`None`|Pretrained embedding prefix, should be Glove formated txt files|  
|`--sos`|`str`|`<s>`|Start-of-sentence symbol|  
|`--eos`|`str`|`</s>`|End-of-sentence symbol|  
|`--share_vocab`|`str`|`False`|whether use the same vocab between source and target|  
|`--check_special_token`|`bool`|`True`|whether check special sos, eos, unk tokens exist in the vocab files|  
|`--src_max_len`|`int`|50|max length of source sequence during training|  
|`--tgt_max_len`|`int`|50|max length of target sequence during training|   
|`--src_max_len_infer`|`int`|`None`|max length of source sequence during inference|  
|`--tgt_max_len_infer`|`int`|`None`|max length of target sequence during inference|  
|`--unit_type`|`str`|`lstm`|one of `lstm`, `gru`, `layer_norm_lstm`, `nas`|  
|`--forget_bias`|`float`|1.0|forget bias for BasicLSTMCell|  
|`--dropout`|`float`|0.2|dropout rate|  
|`--max_gradient_norm`|`float`|5.0|clip gradients to this norm|  
|`--batch_size`|`int`|128|batch size|  
|`--steps_per_stats`|`int`|100|how many training steps to do per stats logging|  
|`--max_train`|`int`|0|limit on the size of training data(0: no limit)|  
|`--num_buckets`|`int`|5|put data into similar-length buckets|  
|`--subword_option`|`str`|`""`|one of `""`, `bpe`, `spm`|  
|`--num_gpus`|`int`|1|number of gpus in each worker|  
|`--log_device_placement`|`bool`|`False`|debug gpu allocation|  
|`--metrics`|`str`|`bleu`|comma-separated list of evaluations|  
|`--steps_per_external_eval`|`int`|`None`|how many training steps to do per external evaluation|  
|`--scope`|`str`|`None`|scope to put variables under|  
|`--hparams_path`|`str`|`None`|path to hparams json file|  
|`--random_seed`|`int`|`None`|random seed|  
|`--override_loadded_hparams`|`bool`|`Flase`|override loaded hparams with values specified|  
|`--num_keep_ckpts`|`int`|5|max number of checkpoints to keep|  
|`--avg_ckpts`|`bool`|`False`|average the last N checkpoints for external evaluation|  
|`--ckpt`|`str`|`""`|checkpoint file to load a model for inference|  
|`--inference_input_file`|`str`|`None`|set to the text decode|  
|`--inference_list`|`str`|`None`|a comma-separated list of sentence indices|  
|`--infer_batch_size`|`int`|32|batch size for inference mode|  
|`--inference_ouput_file`|`str`|`None`|output file to store decoding results|  
|`--inference_ref_file`|`str`|`None`|reference file to compute evaluation scores|  
|`--beam_width`|`int`|0|beam width when using beam search decoder|  
|`--length_penalty_weight`|`float`|0.0|length penalty for beam search|  
|`--sampling_temperature`|`float`|0.0|softmax sampling temperature for inference decoding|  
|`--num_translations_per_input`|`int`|1|number of translations generated for each sentence|  
|`--jobid`|`int`|0|task if of the worker|  
|`--num_workers`|`int`|1|number of workers(inference only)|  
|`--num_inter_threads`|`int`|0|number of inter_op_parallelism_threads|  
|`--num_train_threads`|`int`|0|number of intra_op_parallelism_threads|   

## 逐条详解　　
上一小节列出了所有的超参数，接下来我将分组进行更加详细的解释。　　

### 数据相关参数　　
本小节介绍数据相关的参数：　　
* `--num_units`　　
网络节点数量  
* `--num_layers`  
网络的层数，即网络深度  
* `--num_encoder_layers`  
编码器的网络层数  
* `--num_decoder_layers`  
解码器的网络层数  
* `--encoder_type`  
编码器的类型，`uni`, `bi`, `gnmt`三者之一，编码器的类型会对结果有较大影响。  
* `--residual`  
是否采残差网络  
* `--time_major`  
是否是时间主要模式，如果是，运算过程中会有一个矩阵转置运算  
* `--num_embeddings_partitions`  
词嵌入的分片数量  
* `--attention`  
attention机制的类型，可选项。`luong|scaled_luong|bahdanau|normed_bahdanau|`  
* `--attention_architecture`  
attention架构，可选`standard|gnmt|gnmt_v2`  
* `--output_attention`  
是否在输出单元使用attention，只有`standard`架构的attention能够使用  
* `--pass_hidden_state`  
是否将编码器的隐藏状态传递给解码器，只有在attention机制模型可用  
* `--optimizer`  
优化器，可选`sgd|adam`，默认是`sgd`，即**随机梯度下降**  
* `--learning_rate`  
学习率，默认值`1.0`，如果使用`adam`优化器，可选值为`0.001|0.0001`  
* `warmup_steps`  
预热学习率的步数  
* `warmup_shceme`  
预热学习率的方式，默认是`t2t`即tensor2tensor的方式  
* `--decay_scheme`  
学习率衰减方式，可选`luong234|luong5|luong10`，具体过程请看注释，位于`nmt.py`文件  
* `--num_train_steps`  
训练的轮数  
* `--src`  
该参数指定训练数据中，源数据的文件后缀名。举个例子，我们的训练数据是一对逐行一一对应的文本文件，分别为**address_train.ocr**和**address_train.std**，那么此时我们需要指定该参数为：　`--src=ocr` 　
* `--tgt`  
该参数指定训练数据中，目标数据的文件后缀名，按照上面的举例，我们需要指定该参数为：　`--tgt=std`  
* `--train_prefix`  
该参数是train数据文件的前缀，注意 **需要包含完整路径** ，路径可以是相对路径，也可以是绝对路径。举个例子，上述例子的两个文件我们放在 **/tmp/nmt_model** 目录下面，那么该参数需要设置为:`--train_prefix=/tmp/nmt_model/address_train`，那么train数据的完整路径就是: **/tmp/nmt_model/address_train.ocr** 和 **/tmp/nmt_model/address_train.std**    
* `--dev_prefix`  
该参数指定dev数据文件的前缀，同`--train_prefix`类似。举个例子，在 **/tmp/nmt_model** 目录下面存放我们的dev数据文件 **address_dev.ocr**　和　**address_dev.std**，那么该参数应该指定为：　`--dev_prefix=/tmp/nmt_model/address_dev`  
* `--test_prefix`  
该参数是test数据文件的前缀，其他和上述`--train_prefix`，`--dev_prefix`类似。　　
* `--vocab_prefix`  
该参数指定的是词典文件的前缀，注意　**需要包含完整路径**　，可以是相对路径也可以是绝对路径。举个例子，我们的词典文件为 **vocab.ocr**　和　**vocab.std** ，位于 **/tmp/nmt_model/** 那么该参数应该指定为:`--vocab_prefix=/tmp/nmt_model/vocab`，最终的词典路径为　**/tmp/nmt_model/vocab.ocr** 和　**/tmp/nmt_model/vocab.std**　。　　
* `--embed_prefix`  
该参数指定已经训练好的embedding文件，必须是Glove文件格式。如果没有，使用默认值`None`。　　
* `--out_dir`  
该参数指定模型的保存路径。比如你想保存在　**/tmp/** 目录下，那你这样指定:`--out_dir=/tmp` 。  
* `--sos`  
句子开始的标记，默认是`<s>`  
* `--eos`  
句子结束的标记，默认是`</s>`  
* `--share_vocab`  
训练的源文件和目标文件是否使用一样的词典  
* `--check_special_token`  
是否检查特殊标记  
* `--src_max_len`  
源句子的最大词语数量  
* `--tgt_max_len`  
目标句子的最大词语数量  
* `--src_max_len_infer`  
推断的源句子最大词语数量  
* `--tgt_max_len_infer`  
推断的目标句子最大词语数量  
* `--unit_type`  
编码器和解码器的神经网络单元类型，可选`lstm|gru|layer_norm_lstm|nas`  
* `--forget_bias`  
遗忘门的偏置，默认`1.0`  
* `--dropout`  
丢弃率，有效防止过拟合  
* `--max_gradient_norm`  
将梯度剪裁到指定的标准  
* `--batch_size`  
批大小，全部计算梯度耗时耗力，使用小批量数据计算梯度能有效提升速率  
* `--steps_per_stats`  
多少步输出一次状态  
* `--max_train`  
限制训练的数量，一般不需要设置  
* `--num_buckets`  
分桶数量，分桶策略请见后面的文章，会有分析  
* `--num_gpus`  
GPU数量，用于分布式训练  
* `--log_device_placement`  
是否输出设备信息   
* `--metrics`  
评分方式，默认`BLEU`  
* `--scope`  
变量的域，默认`translate`  
* `--random_seed`  
随机种子，在对数据集乱序的时候有用，也可以不指定  
* `--num_keep_ckpts`  
保存最近的checkpoints的数量，默认`5`  
* `--avg_ckpts`  
是否均值保存点。可以提高性能  
* `--ckpt`  
用于推断的时候，指定某个保存点来推断数据。默认采用评分最高的  
* `--inference_input_file`  
推断的输入文件  
* `--inference_list`  
指定输入文件的某些行，用来推断  
* `--infer_batch_size`  
推断的批大小  
* `--inference_output_file`  
推断的输出结果文件  
* `--inference_ref_file`  
如果提供，用来计算推断结果的得分  
* `--beam_width`  
beam search的宽度  
* `--num_translations_per_input`  
每个句子输出推断结果的数量，即可以输出多个结果  
* `--jobid`  
当前任务的id，用于分布式训练  
* `--num_works`  
workers数量  


## 注意事项　　
训练该模型，对机器的要求比较高。本人尝试过使用公司配的开发机器配置如下：　　
* Platform Windows 7 x64  
* Memory 8G  
* CPU intel core i5-6500  
* GPU GTX950 2G x1  

此配置在我改小了batch_size到32之后，还是报错　Out of memoey.  
在服务器配置如下：　　
* Platform Ubuntu16.04 amd64  
* Memory 32G  
* CPU intel core i7-7700k  
* GPU GTX1080ti 11G x1  

上训练普通模型，并且将batch_size设置成默认的128，可以正常训练，但是此配置训练GNMT模型报错OOM。  　　
  
这里也就说明一个小技巧：　改小batch_size可以降低显存使用。  
  
当然其他维度的降低也可以降低显存使用　　



