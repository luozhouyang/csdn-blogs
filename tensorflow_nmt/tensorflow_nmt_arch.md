# Tensorflow nmt的整体结构　　
tensorflow/nmt项目的入口文件是nmt/nmt.py，通过指定不同的参数，可以从该入口进入到**训练**或者**推断**流程。首先来看一看，进入不同流程的时候，做了什么。　　

## 程序入口　　
首先我们可以看到这两个函数：　　

```python
def main(unused_argv):
  default_hparams = create_hparams(FLAGS)
  train_fn = train.train
  inference_fn = inference.inference
  run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```  

通过　`python -m nmt.nmt --flags`命令启动程序，开始进入到上面的 `if __name__ == "__main__"`流程。首先创建出命令行的**参数解析器**来解析参数设置，然后启动**tensorflow**的主函数。　　

也就是执行 `def main(unused_argv)`　函数，该函数首先根据解析出来的命令行参数，创建出**超参数**，超参数用于神经网络的各项设置中。然后创建出**训练**和**推断**函数，进入到这两个流程的判断逻辑。也就是开始执行 `def run_main(flags, hparams, train_fn, inference_fn)`　函数。　　

`run_main()`函数主要做了这么几件事情：　　
* 提取jobid和num_workers这几个参数，推断的时候使用　　
* 设置随机种子　　
* 创建模型输出路径　　
* 合并超参数，在命令行指定和文件指定两者都有的情况，需要合并超参数　　
* 判断进入训练还是推断流程，根据是否指定了`inference_input_file`这个flag，如果有则进入**推断流程**，否则进入**训练**流程　　

详细代码就不贴了。你可以到**nmt/nmt.py**文件查看。　　

## 训练流程　　
训练部分的代码位于**nmt/train.py**，入口为**train()**函数。  
首先是获取超参数，然后创建训练所需要的模型，如果模型之前训练过，则从最新的保存点回复出训练模型，否则创建新的模型。  

创建出模型之后，就开始进入训练循环，一直到训练的轮数达到超参数指定的数量。
在循环体内就做了以下几件事情：  
* 从训练集获取一批数据进行训练，获取每一步训练得到的结果  
* 如果训练集遍历完成，则开始下一轮训练，训练集数据重新开始分批训练  
* 根据获取的结果，更新状态  
* 如果到了一定的轮数，则用推断模型进行一些外部估算，输出统计数据，并且保存当前的模型  
* 如果到了一定的轮数，则用推断模型进行一些内部估算，保存当前模型  
* 不断循环，直至训练完成  

循环结束后，进行一次全面的估算，然后根据指定的方法，进行模型效果的评估，同事选择出评分最高的模型。  
我们接下来更加详细的看看上述步骤做了什么处理。  

### 步骤一，获取超参数  
获取超参数很简单，大部分是根据传进来的参数，直接获取即可，少部分需要根据其他超参数进行简单的计算得出，主要代码如下：  
```python
"""Train a translation model."""
    log_device_placement = hparams.log_device_placement
    # 模型输出目录
    out_dir = hparams.out_dir
    # 模型训练的轮数
    num_train_steps = hparams.num_train_steps
    # 多少轮计算一次状态
    steps_per_stats = hparams.steps_per_stats
    # 多少轮计算一次内部估算
    steps_per_external_eval = hparams.steps_per_external_eval
    # 多少轮计算一次估算，为计算状态的轮数*10
    steps_per_eval = 10 * steps_per_stats
    avg_ckpts = hparams.avg_ckpts

    if not steps_per_external_eval:
        steps_per_external_eval = 5 * steps_per_eval

    # 区分是否启用attention机制
    if not hparams.attention:
        model_creator = nmt_model.Model
    else:  # Attention
        # 区分具体是哪种attention机制
        if (hparams.encoder_type == "gnmt" or
                hparams.attention_architecture in ["gnmt", "gnmt_v2"]):
            model_creator = gnmt_model.GNMTModel
        elif hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        else:
            raise ValueError("Unknown attention architecture %s" %
                             hparams.attention_architecture)
```  
上述`model_creator`可以根据指定的不同机制，获取不同的模型。  

### 步骤二，创建训练所需要的模型  
训练的过程，需要三种模型，分别是**训练模型、估算模型（验证模型）和推断模型**。后两种模型在进行内部估算的时候有用。  
创建三种模型的代码比较相似，都是根据对应的超参数和数据集创建。我会单独写一篇文章来说明，模型的的创建过程。在此不详细说明。获取三种模型的代码也很简单：  
```python

# 创建出训练模型
    train_model = model_helper.create_train_model(model_creator, hparams, scope)
    # 创建出估算模型
    eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
    # 创建出推断模型
    infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

    # Preload data for sample decoding.
    # 验证集的完整文件路径
    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    # 获取样本数据
    sample_src_data = inference.load_data(dev_src_file)
    sample_tgt_data = inference.load_data(dev_tgt_file)
```  

### 步骤三，获取每一批训练的结果  
进入训练循环，首先就是根据每一批训练集，进行训练，然后或许训练的结果。  
具体的过程如下：  


## 推断流程　　

