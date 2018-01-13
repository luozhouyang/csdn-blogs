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


## 推断流程　　

