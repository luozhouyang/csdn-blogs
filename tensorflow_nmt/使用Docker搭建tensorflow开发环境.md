# 使用Docker搭建tensorflow nmt开发环境  
Docker是目前最流行的容器技术. 可以将Docker看成轻量级的虚拟机,但是它非常轻量. 它像是一艘运输船,将开发或者运维等等过程中所需要的依赖全部打包到一个镜像(image)中, 任何人只需要将该镜像拉取下来, 就可以开箱即用, 不需要自己去安装一大堆依赖. 同时因为这些依赖都是运行在容器里面的, 不会污染宿主机的环境, 因此非常适合用来开发运维等等.  

Docker的官方网站有更加详细的介绍和非常棒的技术文档: [Docker](http://www.docker.com)  

本教程将带领大家使用Docker搭建tensorflow的开发环境.  
tensorflow官方也有安装的文档，建议首先按照官方文档来安装，如果遇到问题，可以回来参考本文．　　

[tensorflow官方安装文档](https://www.tensorflow.org/install/)  

需要说明的是, tensorflow有CPU和GPU两种版本, 本教程使用的是GPU版本,如果你需要安装CPU版本，请参考tensorflow的文档.  

## 本教程所使用的环境　　
本教程使用的环境如下：　　
* 系统是Ubuntu 16.04 amd64  
* 显卡是GTX 1080ti　　

使用GPU版本的tensorflow需要使用CUDA toolkit，本文后续会有讲解.  
接下来，可以正式进入到开发环境的搭建过程．　　

## Docker的安装　　
Docker有CE和EE两个版本,对于个人用户,建议安装Docker-CE.  
Docker官方有非常详细的安装文档: [Docker installation](https://docs.docker.com/engine/installation/)  

本教程将主要步骤罗列如下:  
* 卸载旧版本的Docker  
* 使用apt安装Docker  

### 使用apt安装Docker  
在Ubuntu平台上使用apt安装Docker非常方便.按照如下步骤,一般不会出现问题.  

step 1.设置软件仓库  
```bash  
$ sudo apt-get update  
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common  
```  

step2.添加官方的GPG key  
```bash  
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -  
```  
如果你想验证指纹,则:  
```bash  
sudo apt-key fingerprint 0EBFCD88
```  

你会看到如下输出:  
```bash  
pub   4096R/0EBFCD88 2017-02-22
      Key fingerprint = 9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid                  Docker Release (CE deb) <docker@docker.com>
sub   4096R/F273FCD8 2017-02-22  
```  


step 3.添加软件源到apt  
本步骤需要选择合适的软件源, 本教程使用的是Ubuntu 16.04 amd64,并且想使用 稳定版 的docker,因此需要这样做:  
```bash  
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"  
```  
如果你的平台不是这样的,请参考官方文档的说明:  [Docker安装](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#set-up-the-repository)  

step 4.安装Docker  
接下来就可以安装Docker了.  
```bash  
$ sudo apt-get update  
$ sudo apt-get install docker-ce  
```  
你可以使用一定的方式确保Docker已经正确安装,比如输入如下命令:  
```bash  
$ sudo docker run hello-world  
```  
该命令会运行一个新的容器,如果成功,你会看到类似于打印信息.  
至此,Docker已经安装完毕.  

### Docker的常用设置  
Docker安装好之后,还需要一定的设置,这些设置可以让你使用的更加舒适.   

#### 使用DaoClould加速  
输入如下命令，即可使用DaoCloud给你的Docker加速：　　
```bash  
curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://85f32c34.m.daocloud.io  
```  

#### 更改Docker镜像存放的路径  
Docker的镜像默认放在　**/var/lib/docker** 目录下，当镜像越来越多，磁盘空间越发紧张，为此，可以讲镜像存放路径改到大容量的磁盘中去．　　

用你喜欢的编辑器打开　**/etc/default/docker** 文件，在 **DOCKER_OPTS**选项写上你的自定义路径：　　

```bash  
DOCKER_OPTS="-g $YOUR_PATH"
```  

修改之后，重启docker:  
```bash  
$ systemctl restart docker  
```  

#### 将用户添加到docker组，免输sudo　　
此时，Docker已经安装完毕并且可以正常使用了，但是你每次都需要输入sudo．　　

为了免输sudo，Docker提供了相关的方法，你可以查看官方文档[用户添加到docker组](https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user)  

简单来说就以下几个步骤：　　

```bash  
$ sudo groupadd docker  
$ sudo usermod -aG docker $USER  
```  

完成之后，你可以试验以下：　　

```bash  
docker run hello-world  
```  
不出意外，可以运行成功．　　

至此，Docker的安装和配置已经完成．　　

接下来就是Tensorflow GPU版本的安装了．　　


## Tensorflow GPU版本的安装　　　　
然后，为了能在docker里面使用tensorflow gpu版本，你需要安装nvidia-docker．  

安装好之后，就可以拉取tensorflow-gpu镜像，然后启动容易开始开发了．　　

### nvidia-docker的安装　　
nvidia-docker的安装官方也提供了文档，建议先查看官方文档: [nvidia-docker installation](https://github.com/NVIDIA/nvidia-docker)  

在此给出Ubuntu 16.04 版本的apt安装方式：　　
```bash
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker

# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```  
至此，nvidia-docker安装好了．　　

### 拉取镜像并运行　　
根据tensorflow的官方文档，可以按照以下命令启动容器：　

```bash  
$ nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu bash
```  

但是本人经过实验并不能成功，忘记是什么原因导致的了，估计是网络原因．　  

解决这个问题也很简单，tensorflow会上传镜像到docker hub，我们用docker hub里面拉取镜像即可．  

这里可以查看所有的tensorflow 标签: [tensorflow tags](https://hub.docker.com/r/tensorflow/tensorflow/tags/)  

我们修改命令如下：　　

```bash  
$ nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3 bash
```  
这将启动一个新的容器，你可以在里面开始进行开发啦．　  

该容器的python版本是3.ｘ的，你可以测试一下，如果你需要安装py2.x的，将上面命令最后的 **latest-gpu-py3** 改成　**latest-gpu** 即可．　　

值得注意的是，使用tensorflow gpu版本的容器，**必须要用nvidia-docker而不是docker命令** 启动容器，**否则会提示找不到 libcuda8.0.so libcudnn6.0.so** 等问题！！！

至此，tensorflow gpu版本的开发环境已经配置好了！可以进行开发了！

## 联系我  
如果你发现博客内容有不对或者说的不清楚的地方,请联系我,我将第一时间改正,尽我的最大能力将问题讲清楚.  
我的邮箱: [stupidme.me.lzy@gmail.com](mailto:stupidme.me.lzy@gmail.com)  

以下是我的公众号，不定期和大家分享技术文章．如果你觉得我的文章对你有帮助，麻烦关注一下哟：

![stupidmedotme](wechat_gzh_code_8.jpg)  

