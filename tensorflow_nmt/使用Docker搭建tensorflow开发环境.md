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



