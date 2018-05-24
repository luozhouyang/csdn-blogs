# Tensorflow nmt源码解析  
> 声明
> 本系列博客由 **罗周杨** [stupidme.me.lzy@gmail.com](mailto:stupidme.me.lzy@gmail.com) 原创，同步更新在多个平台，包括：  
> [csdn/stupid_3](http://blog.csdn.net/stupid_3/article/details/78956470)  
> [github/luozhouyang](https://github.com/luozhouyang/csdn-blogs/blob/master/tensorflow_nmt/tensorflow_nmt_index.md)  
> 分享或转载请注明作者和原出处。　　


NMT即Neural Machine Translation，神经网络机器翻译。Google开源的tensorflow机器学习框架中，提供了一个NMT的demo。NMT使用seq2seq模型。将一个序列转化为另一个序列。凡是符合此特征的实际问题，均可以使用seq2seq模型。因此常见的机器翻译，文本摘要和对话机器人等常常使用seq2seq模型。  

Google提供的nmt demo对seq2seq模型也有介绍，请参考项目地址的首页README文档。

Google提供的nmt代码开源在GitHub,请访问 [tensorflow/nmt](https://github.com/tensorflow/nmt)。  

本博客将带领大家从源码上入手NMT模型，在讲解代码的同时，介绍一些相关的数学原理。  

本博客是一个系列文章, 分成以下几个部分:  

* [tensorflow/nmt 的开发环境搭建](tensorflow_nmt_env_with_docker.md)  
* [tensorflow/nmt 的整体结构](tensorflow_nmt_arch.md)  
* [tensorflow/nmt 的超参数](tensorflow_nmt_hparams.md)    
* [tensorflow/nmt 的数据处理过程](tensorflow_nmt_dataset_process.md)  
* tensorflow/nmt 的基本模型  
* tensorflow/nmt 的attention模型  
* tensorflow/nmt 的gnmt模型  
* tensorflow/nmt 的训练模型  
* tensorflow/nmt 的推断模型    
* tensorflow/nmt 的训练示例    
* tensorflow/nmt 的推断示例    
* tensorflow/nmt 的模型部署到tensorflow serving    

## 联系我  
如果你发现博客内容有不对或者说的不清楚的地方，请联系我，我将第一时间改正，尽我的最大能力将问题讲清楚。  
 
我的邮箱: [stupidme.me.lzy@gmail.com](mailto:stupidme.me.lzy@gmail.com)  

以下是我的公众号，不定期和大家分享技术文章。如果你觉得我的文章对你有帮助，麻烦关注一下哟：

![stupidmedotme](wechat_gzh_code_8.jpg)  
