## 从设计者的角度解读ThreadLocal  

>版权声明：本文为 *罗周杨 stupidme.me.lzy@gmail.com* 原创，未经授权不得转载。  


本文将从设计者的角度着手解析ThreadLocal。如果你是设计者，你会怎么设计？  

### 需求  
我们看到的代码都是某种需求的产物。假设有这样一个需求:　我有一些数据，我希望每一个线程都可以访问它，并且更新之后，不影响其他线程的值，也就是说，每一个线程对该数据都可以自由读写，线程之间互不干扰。  
如果是你，你会怎么设计呢？  

大致有两个思路。第一种，**将这些线程和数据组成的关系对，存放在一个公共的区域，各个线程都可以自由获取，并且通过某些手段保证数据在线程之间互不影响**。第二种，**每个线程创建一个成员变量来保存这个关系对**。　　

### 首先想到的HashMap  
我们首先来看看第一种方式。  

你一定知道HashMap，它对每一个键，都可以存储对应的一个值，并且根据不同的键，可以获取到对应的值。HashMap是不是符合我们的需求呢？  

是的，看起来完全符合。但是，这里面有一点小小的区别。因为我们要区分的是不同的线程，而不是不同的键。那我把线程当成键不就行了？那么我们试试看。　　

给我们实现需求的代码起一个名字吧，就叫做ThreadLocal。按照目前的逻辑，代码大概长成这样：  

```java  
public class ThreadLocal {
    private Map<Thread, Object> mMap = new HashMap<>();
    public void set(Object o) {
        mMap.put(Thread.currentThread(), o);
    }
    public Object get() {
        return mMap.get(Thread.currentThread());
    }
    public void remove() {
        mMap.remove(Thread.currentThread());
    }
}
```  
额，干脆改成泛型吧：  

```java  
public class ThreadLocal<T> {
    private Map<Thread, T> mMap = new HashMap<>();
    public void set(T o) {
        mMap.put(Thread.currentThread(), o);
    }
    public T get() {
        return mMap.get(Thread.currentThread());
    }
    public void remove() {
        mMap.remove(Thread.currentThread());
    }
}
```  
看起来不错。但是似乎有了一个新的问题。  

### 一个新的问题  
问题是，这样的话，我每一个线程只能存储一个数据（尽管这个数据是泛型）？如果我要存储多个数据是不是就没办法了？  

按照目前的设计，似乎是这样的。那么我们可以改进吗？看起来没有办法。因为当前是把线程本身当做键，也就是说这个键对于每一个线程来说都是唯一的，那通过这个键获取的值，也就只能是一个。问题出现在这个键上面。　　

那我们是不是可以换一种方式，不要把线程当做键不就行了？事实上，不仅仅是线程本身不能作为键，所有能够唯一标志该线程的东西，都不能作为键。　　

如果不能使用标志线程的东西作为键，那我们能用什么呢？一时半会儿好像没有什么注意。我们不妨整理一下目前的处境。　　

首先，**使用一个ThreadLocal存储线程的一个值是没问题的**。但是，**线程可能需要存储多个值**。另外，一种想法是**类似于HashMap存储多个Entry，我们使用ThreadLocalMap来存储多个Entry，把ThreadLocal作为Entry**，但是**ThreadLocalMap的键需要好好设计一下**。　　

你肯定已经发现了，**使用ThreadLocalMap**这个想法其实包含了以下要求：  
* **这个ThreadLocalMap不能放在公共的区域，因为放在公共的区域，每个线程去获取的话，只能通过标志线程的东西作为键，这个问题我们之前已经讨论过了**。

这样的话，我们只能**把ThreadLocalMap当做线程（Thread）的一个成员变量**，让线程自己持有这些数据关系。这其实也就是我们当初的**第二种**想法：  
> 每个线程创建一个成员变量来保存这个关系对  

### ThreadLocalMap作为Thread的成员变量  
终于，我们走到这条道路上面来了。实际上JDK里面的Thread类确实包含**ThreadLocalMap**成员变量，但是**数量上不是一个，而是两个**。你可以去看看源代码，然后想想为什么。  

现在我们已经明确了使用**ThreadLocalMap**，那么我们该如何设计**ThreadLocalMap**呢？　　

### 几个既有的事实  

我们先理清楚以下几个事实：　　
* **ThreadLocal的get()方法是通过Thread.currentThread()作为键来获取值的**。　　
* **不管在哪里的代码块，我们总是能够通过 Thread.currentThread()获取当前执行代码的线程**。  
* **每个线程都可以获取自己的成员变量——ThreadLocalMap**。　　

这也就是说，即使是任何一个地方的**ThreadLocal**，我们也可以通过**ThreadLocal.getMap()类似的方法**来获取当前线程的**ThreadLocalMap**。逻辑很简单：　　

```java  
code block{ 
	ThreadLocal threadLocal//ThreadLocal变量  
	Thread t = Thread.currentThread() //当前执行代码的线程
	ThreadLocalMap map = t.getMap() //当前线程的所有thread local数据
}
```  

如果是这样的话，那我们干脆给**ThreadLocal**封装一个获取对应线程的ThreadLocalMap数据的方法，就叫做**getMap()**吧，它的内容大概应该:  

```java  
public ThreadLocalMap getMap() {
	Thread t = Thread.currentThread();
	ThreadLocalMap map = t.getMap();
	return map;
}
```  

聪明的你肯定发现了，**ThreadLocalMap存放的所有该线程的数据对，那我为什么不进一步将ThreadLocalMap里面的数据取出来，直接返回该数据，而不是返回ThreadLocalMap呢**？  

没错！！！那么的代码应该是这个样子了：  

```java  
public T get(Key k) {
	Thread t = Thread.currentThread();
	ThreadLocalMap map = t.getMap();
	T value = map.get(k);
	return value;
}
```  

看起来不错，也就是说我们可以直接**根据Key**从**ThreadLocal类而不是ThreadLocalMap类**获取**线程独立的数据**。那么这个**Key**应该是什么呢？　　

### 问题还是在于如何设计Key　　
我们终于看到了一点希望。刚刚我们说的**如何设计键**的问题再次出现了。  

等等，好像有点不太对，我们一开始的**ThreadLocal**的get方法没有参数呀，我们想要的是**把当前线程作为一种区分约束，直接获取线程独立的数据值**。但是我们现在的get方法为何出现了**Key**这个参数？　　

看起来我们已经偏离需求了。那怎么办呢？按照需求来吧，先把参数**Key**去掉，然后我们再想办法解决其他问题。　　

现在的问题是，我们已经失去了**Key**这个参数，但是**ThreadLocalMap**又需要一个Key来获取对应的值。那该怎么办呢？这个Key总不能无中生有吧？　　

无中生有是不可能了。但是根据你的经验，应该知道，有些东西其实是真是存在的，但是一般情况下似乎隐藏起来了。比如代表当前对象的**this**。　　

### 神奇的this  
说到这里，你可能焕然大悟：**ThreadLocalMap的键用ThreadLocal**不就行了！！！  
好像信息量有点大，我们一步一步理清楚。第一个问题，**this**和**ThreadLocalMap**的键是如何联系起来的？第二个问题，用**ThreadLocal**作为键真的可以吗？　　

第一个问题其实很简单，看代码吧：　　

```java  
public T get() {
	Thread t = Thread.currentThread();
	ThreadLocalMap map = t.getMap();
	T value = map.get(this); //其实就是把当前ThreadLocal对象作为键
	return value;
}
```  

第二个问题，这样可行吗？假设就是把ThreadLocal作为Key，我们先理一理现在的ThreadLocal、ThreadLocalMap、Thread大概是什么样的。　　
Thread包含ThreadLocalMap成员变量：
```java  
public class Thread {
	...
	ThreadLocalMap map;
	...
}
```  
ThreadLocalMap包含多个Entry，就类似于HashMap，每个Entry的键为ThreadLocal，值类型和ThreadLocal的值类型保持一致：
```java  
public class ThreadLocalMap {
	Entry[] table;
	private static class Entry<ThreadLocal, T> {
	...
	}
	...
}
```  
ThreadLocal相对简单，主要还是那几个方法。根据前面的讨论，数据其实都存放在ThreadLocalMap里面了，因此ThreadLocal的几个方法都是对ThreadLocalMap对应方法的封装：　　
```java
public class ThreadLocal<T> {
	public void set(T o) {
		Thread t = Thread.currentThread();
		ThreadLocalMap map = t.getMap();
		map.set(this, o);
	}
	public T get() {
		Thread t = Thread.currentThread();
		ThreadLocalMap map = t.getMap();
		T value = map.get(this);
		return value;
	}
	public void remove() {
		Thread t = Thread.currentThread();
		ThreadLocalMap map = t.getMap();
		map.remove(this);
	}
}
```  

仔细看看代码，好像完全满足了需求！！！**首先，当前线程想要存放在ThreadLocal里面的数据，全部保存在了Thread的成员变量ThreadLocalMap里面，这当然是线程无关的。其次，当前线程想要获取值的时候，直接从Thread的成员变量ThreadLocalMap里面取出来即可。再者，如果当前线程需要存放多个数据，只需要构造出多个ThreadLocal对象，存到自己的ThreadLocalMap即可**。

### 结语
至此，其实我们已经设计出了一个符合需求的ThreadLocal。尽管目前的设计是很粗糙的，我们的思路其实是完全正确的。　　

好好感受一下我们从零开始到现在设计出一个粗糙的ThreadLocal的历程，再结合JDK的源代码，感受一下大师们的成品是怎么样的，你会收获更多。　　

希望本文这种从设计者的角度理解ThreadLocal的方式对你有帮助。  

#### 联系我：  
Email: <stupidme.me.lzy@gmail.com>  
WeChat: luozhouyang0528 




