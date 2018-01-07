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


## 超参数的使用　　


## 注意事项　　


## 扩展－－分布式训练　　

