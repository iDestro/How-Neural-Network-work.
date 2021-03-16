### 循环神经网络

#### 基本的循环神经网络

下图是一个简单的循环神经网络如，它由输入层、一个隐藏层和一个输出层组成：

![点击查看大图](./images/2256672-479f2a7488b91671.png)

如果我们把上面的图展开，**循环神经网络**也可以画成下面这个样子：

![点击查看大图](./images/2256672-cf18bb1f06e750a4.png)

现在看上去就比较清楚了，这个网络在$t$时刻接收到输入$x_t$之后，隐藏层的值是$s_t$，输出值是$o_t$。关键一点是，$s_t$的值不仅仅取决于$x_t$，还取决于$s_{t-1}$。我们可以用下面的公式来表示**循环神经网络**的计算方法：
$$
\begin{aligned}
\mathrm{o}_{t} &=g\left(V \mathrm{~s}_{t}\right) \\
\end{aligned} \tag{1}
$$

$$
\mathrm{s}_{t} =f\left(U \mathrm{x}_{t}+W \mathrm{~s}_{t-1}\right) \tag{2}
$$

**式1**是**输出层**的计算公式，输出层是一个**全连接层**，也就是它的每个节点都和隐藏层的每个节点相连。$V$是输出层的**权重矩阵**，$g$是**激活函数**。式2是隐藏层的计算公式，它是**循环层**。$U$是输入$x$的权重矩阵，$W$是上一次的值作为这一次的输入的**权重矩阵**，$f$是**激活函数**。

从上面的公式我们可以看出，**循环层**和**全连接层**的区别就是**循环层**多了一个**权重矩阵** W。

如果反复把**式2**带入到**式1**，我们将得到：
$$
\begin{aligned}
\mathrm{o}_{t} &=g\left(V \mathrm{~s}_{t}\right) \\
&=V f\left(U \mathrm{x}_{t}+W \mathrm{~s}_{t-1}\right) \\
&=V f\left(U \mathrm{x}_{t}+W f\left(U \mathrm{x}_{t-1}+W \mathrm{~s}_{t-2}\right)\right) \\
&=V f\left(U \mathrm{x}_{t}+W f\left(U \mathrm{x}_{t-1}+W f\left(U \mathrm{x}_{t-2}+W \mathrm{~s}_{t-3}\right)\right)\right) \\
&=V f\left(U \mathrm{x}_{t}+W f\left(U \mathrm{x}_{t-1}+W f\left(U \mathrm{x}_{t-2}+W f\left(U \mathrm{x}_{t-3}+\ldots\right)\right)\right)\right)
\end{aligned}
$$
从上面可以看出，**循环神经网络**的输出值$o_t$，是受前面历次输入值$x_t$、$x_{t-1}$、$x_{t-2}$、...影响的，这就是为什么**循环神经网络**可以往前看任意多个**输入值**的原因。

### 双向循环神经网络

![点击查看大图](/Users/idestro/PycharmProjects/Neural-Network/rnn/images/2256672-039a45251aa5d220.png)

从上图可以看出，双向卷积神经网络的隐藏层要保存两个值，一个$A$参与正向计算，另一个值参与反向计算。最终的输出值$y_2$取决于$A_2$和$A'_2$。其计算方法为：
$$
\mathrm{y}_{2}=g\left(V A_{2}+V^{\prime} A_{2}^{\prime}\right)
$$
$A_2$和$A'_2$则分别计算：
$$
\begin{array}{l}
A_{2}=f\left(W A_{1}+U_{\mathrm{x}_{2}}\right) \\
A_{2}^{\prime}=f\left(W^{\prime} A_{3}^{\prime}+U^{\prime} \mathrm{x}_{2}\right)
\end{array}
$$
现在，我们已经可以看出一般的规律：正向计算时，隐藏层的值$s_t$与$s_{t-1}$有关；反向计算时，隐藏层的值$s'_t$与$s'_{t+1}$有关；最终的输出取决于正向计算的加和。现在，我们仿照式1和式2，写出双向循环神经网络的计算方法：
$$
\begin{array}{l}
\mathrm{o}_{t}=g\left(V \mathrm{~s}_{t}+V^{\prime} \mathrm{s}_{t}^{\prime}\right) \\
\mathrm{s}_{t}=f\left(U \mathrm{x}_{t}+W \mathrm{~s}_{t-1}\right) \\
\mathrm{s}_{t}^{\prime}=f\left(U^{\prime} \mathrm{x}_{t}+W^{\prime} \mathrm{s}_{t+1}^{\prime}\right)
\end{array}
$$
从上面三个公式我们可以看到，正向计算和反向计算**不共享权重**，也就是说U和U'、W和W'、V和V'都是不同的**权重矩阵**。

### 深度循环神经网络

前面我们介绍的**循环神经网络**只有一个隐藏层，我们当然也可以堆叠两个以上的隐藏层，这样就得到了**深度循环神经网络**。如下图所示：

![点击查看大图](/Users/idestro/PycharmProjects/Neural-Network/rnn/images/480.png)

我们把第$i$各隐藏层的值表示为$s^{(i)}_t$、${s^{'(i)}}_t$，则深度循环神经网络的计算方式可以表示为：
$$
\begin{aligned}
\mathrm{o}_{t} &=g\left(V^{(i)} \mathrm{s}_{t}^{(i)}+V^{\prime(i)} \mathrm{s}_{t}^{\prime(i)}\right) \\
\mathrm{s}_{t}^{(i)} &=f\left(U^{(i)} \mathrm{s}_{t}^{(i-1)}+W^{(i)} \mathrm{s}_{t-1}\right) \\
\mathrm{s}_{t}^{\prime(i)} &=f\left(U^{\prime(i)} \mathrm{s}_{t}^{\prime(i-1)}+W^{\prime(i)} \mathrm{s}_{t+1}^{\prime}\right) \\
\ldots & \\
\mathrm{s}_{t}^{(1)} &=f\left(U^{(1)} \mathrm{x}_{t}+W^{(1)} \mathrm{s}_{t-1}\right) \\
\mathrm{s}_{t}^{\prime(1)} &=f\left(U^{\prime(1)} \mathrm{x}_{t}+W^{\prime(1)} \mathrm{s}_{t+1}^{\prime}\right)
\end{aligned}
$$

### 循环神经网络的训练

循环神经网络的训练方法：BPTT

BPTT算法是针对**循环层**的训练算法，它的基本原理和BP算法是一样的，也包含同样的三个步骤：

- 前向计算每个神经元的输出值
- 反向计算每个神经元的误差项$\delta_j$值，它是误差函数$E$对神经元$j$的加权输入$net_j$的偏导数
- 计算每个权重的梯度
- 最后使用梯度下降算法更新权重

循环层如下图所示：

![点击查看大图](/Users/idestro/PycharmProjects/Neural-Network/rnn/images/480-20210315204848030.png)

### 前向计算

使用前面的式2对循环层进行前向计算：
$$
s_t=f(Ux_t+Ws_{t-1})
$$
注意，上面的$s_t$、$x_t$、$s_{t-1}$都是向量，用黑体字母表示；而$U$、$V$是矩阵，用大写字母表示。向量的下标表示时刻。

我们假设输入向量$x$的维度是$m$，输出向量$s$的维度是$n$，则矩阵$U$的维度是$n \times m$，矩阵$W$的维度是$n \times n$。下面是上式展开成矩阵的样子，看起来更直观一些：
$$
\left[\begin{array}{c}
s_{1}^{t} \\
s_{2}^{t} \\
\cdot \\
\cdot \\
s_{n}^{t}
\end{array}\right]=f\left(\left[\begin{array}{c}
u_{11} u_{12} \ldots u_{1 m} \\
u_{21} u_{22} \ldots u_{2 m} \\
\cdot \\
\cdot \\
u_{n 1} u_{n 2} \ldots u_{n m}
\end{array}\right]\left[\begin{array}{c}
x_{1} \\
x_{2} \\
\cdot \\
\cdot \\
x_{m}
\end{array}\right]+\left[\begin{array}{c}
w_{11} w_{12} \ldots w_{1 n} \\
w_{21} w_{22} \ldots w_{2 n} \\
\cdot \\
\cdot \\
w_{n 1} w_{n 2} \ldots w_{n n}
\end{array}\right]\left[\begin{array}{c}
s_{1}^{t-1} \\
s_{2}^{t-1} \\
\cdot \\
s_{n}^{t-1}
\end{array}\right]\right)
$$
在这里我们用**手写体字母**表示向量的一个**元素**，它的下标表示它是这个向量的第几个元素，它的上标表示第几个**时刻**。例如，$s_j^t$表示向量$s$的第$j$个元素在$t$时刻的值。$u_{ji}$表示输入层第$i$各神经元到循环层第$j$各神经元的权重。$w_{ji}$表示循环层第$t-1$时刻的第$i$各神经元到循环层第$t$个时刻的第$j$个神经元的权重。

误差项的计算

BTPP算法将第$l$层$t$时刻的误差项$\delta^l_t$值沿两个方向传播，一个方向是其传递到上一层网络，得到$\delta_t^{l-1}$，这部分只和权重矩阵$W$有关。

我们用向量$net_t$表示神经元在$t$时刻的加权输入，因为：
$$
\begin{array}{l}
\operatorname{net}_{t}=U \mathrm{x}_{t}+W \mathrm{~s}_{t-1} \\
\mathrm{~s}_{t-1}=f\left(\operatorname{net}_{t-1}\right)
\end{array}
$$
因此：
$$
\frac{\partial \text { net }_{t}}{\partial \text { net }_{t-1}}=\frac{\partial \text { net }_{t}}{\partial \mathrm{s}_{t-1}} \frac{\partial s_{t-1}}{\partial \mathrm{net}_{t-1}}
$$
我们用$a$表示列向量，用$a^{T}$表示行向量。上式的第一项是向量函数对向量求导，其结果为Jacobian矩阵:
$$
\begin{aligned}
\frac{\partial \operatorname{net}_{t}}{\partial \mathrm{s}_{t-1}}=&\left[\begin{array}{llll}
\frac{\partial n e t_{1}^{t}}{\partial s_{1}^{t-1}} & \frac{\partial n e t_{1}^{t}}{\partial s_{2}^{t-1}} & \cdots & \frac{\partial n e t_{1}^{t}}{\partial s_{n}^{t-1}} \\
\frac{\partial n e t_{2}^{t}}{\partial s_{1}^{t-1}} & \frac{\partial n e t_{2}^{t}}{\partial s_{2}^{t-1}} & \cdots & \frac{\partial n e t_{2}^{t}}{\partial s_{n}^{t-1}} \\
\frac{\partial n e t_{n}^{t}}{\partial s_{1}^{t-1}} & \frac{\partial n e t_{n}^{t}}{\partial s_{2}^{t-1}} & \cdots & \frac{\partial n e t_{n}^{t}}{\partial s_{n}^{t-1}}
\end{array}\right] \\
&=\left[\begin{array}{cccc}
w_{11} & w_{12} & \ldots & w_{1 n} \\
w_{21} & w_{22} & \ldots & w_{2 n} \\
& \cdot & & \\
& \cdot & & \\
w_{n 1} & w_{n 2} & \ldots & w_{n n}
\end{array}\right] \\
&=W
\end{aligned}
$$
同理，上式第二项也是一个Jacobian矩阵：
$$
\begin{aligned}
\frac{\partial \mathrm{s}_{t-1}}{\partial \mathrm{net}_{t-1}} &=\left[\begin{array}{cccc}
\frac{\partial s_{1}^{t-1}}{\partial n e t_{1}^{t-1}} & \frac{\partial s_{1}^{t-1}}{\partial n e t_{2}^{t-1}} & \cdots & \frac{\partial s_{1}^{t-1}}{\partial n e t_{n}^{t-1}} \\
\frac{\partial s_{2}^{t-1}}{\partial n e t_{1}^{t-1}} & \frac{\partial s_{2}^{t-1}}{\partial n e t_{2}^{t-1}} & \cdots & \frac{\partial s_{2}^{t-1}}{\partial n e t_{n}^{t-1}} \\
\frac{\partial s_{n}^{t-1}}{\partial n e t_{1}^{t-1}} & \frac{\partial s_{n}^{t-1}}{\partial n e t_{2}^{t-1}} & \cdots & \frac{\partial s_{n}^{t-1}}{\partial n e t_{n}^{t-1}}
\end{array}\right] \\
&=\left[\begin{array}{cccc}
f^{\prime}\left(n e t_{1}^{t-1}\right) & 0 & & \ldots & 0 \\
0 & f^{\prime}\left(n e t_{2}^{t-1}\right) & \ldots & 0 \\
& \cdot & & & \\
& & & \cdots & \\
0 & 0 & & \ldots & f^{\prime}\left(n e t_{n}^{t-1}\right)
\end{array}\right] \\
&=\operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{t-1}\right)\right]
\end{aligned}
$$
其中，$diag[a]$表示根据向量a创建一个对角矩阵，即
$$
\operatorname{diag}(\mathrm{a})=\left[\begin{array}{cccc}
a_{1} & 0 & \ldots & 0 \\
0 & a_{2} & \ldots & 0 \\
& . & & \\
0 & 0 & \ldots & a_{n}
\end{array}\right]
$$
最后，将两项合在一起，可得：
$$
\begin{aligned}
\frac{\partial \text { net }_{t}}{\partial \text { net }_{t-1}} &=\frac{\partial \text { net }_{t}}{\partial_{\mathrm{s} t-1}} \frac{\partial \mathrm{s}_{t-1}}{\partial \mathrm{net}_{t-1}} \\
&=\operatorname{Wdiag}\left[f^{\prime}\left(\operatorname{net}_{t-1}\right)\right] \\
&=\left[\begin{array}{llll}
w_{11} f^{\prime}\left(n e t_{1}^{t-1}\right) & w_{12} f^{\prime}\left(n e t_{2}^{t-1}\right) & \ldots & w_{1 n} f\left(n e t_{n}^{t-1}\right) \\
w_{21} f^{\prime}\left(n e t_{1}^{t-1}\right) & w_{22} f^{\prime}\left(n e t_{2}^{t-1}\right) & \ldots & w_{2 n} f\left(n e t_{n}^{t-1}\right) \\
& & & & \\
& & & \\
w_{n 1} f^{\prime}\left(n e t_{1}^{t-1}\right) & w_{n 2} f^{\prime}\left(n e t_{2}^{t-1}\right) & \ldots & w_{n n} f^{\prime}\left(n e t_{n}^{t-1}\right)
\end{array}\right]
\end{aligned}
$$
上式描述了将$\delta$沿时间往前传递一个时刻的规律，有了这个规律，我们就可以求得任意时刻$k$的误差项$\delta_k$：
$$
\begin{aligned}
\delta_{k}^{T} &=\frac{\partial E}{\partial \text { net }_{k}} \\
&=\frac{\partial E}{\partial \text { net }_{t}} \frac{\partial \mathrm{net}_{t}}{\partial \operatorname{net}_{k}} \\
&=\frac{\partial E}{\partial \mathrm{net}_{t}} \frac{\partial \mathrm{net}_{t}}{\partial \mathrm{net}_{t-1}} \frac{\partial \mathrm{net}_{t-1}}{\partial \mathrm{net}_{t-2}} \ldots \frac{\partial \mathrm{net}_{k+1}}{\partial \mathrm{net}_{k}} \\
&=W \operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{t-1}\right)\right] W \operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{t-2}\right)\right] \ldots W \operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{k}\right)\right] \delta_{t}^{l} \\
&=\delta_{t}^{T} \prod_{i=k}^{t-1} W \operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{i}\right)\right]
\end{aligned}
$$
上式就是将误差项沿时间反向传的算法。

循环层将误差项反向传递到上一层网络，与普通的全连接层是完全一样的。

循环层的加权输入$net^l$与上一层的加权输入$net^{l-1}$关系如下：
$$
\begin{array}{l}
\operatorname{net}_{t}^{l}=U \mathbf{a}_{t}^{l-1}+W \mathbf{s}_{t-1} \\
\mathbf{a}_{t}^{l-1}=f^{l-1}\left(\operatorname{net}_{t}^{l-1}\right)
\end{array}
$$
上式中,

$net_t^l$是第$l$层神经元的加权输入（假设第$l$层是循环层）

$net_t^{t-1}$是第$l-1$层神经元的加权输入

$a_t^{t-1}$是第$l-1$层神经元的输出

$f^{l-1}$是第$l-1$层的激活函数
$$
\begin{aligned}
\frac{\partial \operatorname{net}_{t}^{l}}{\partial \mathrm{net}_{t}^{l-1}} &=\frac{\partial \mathrm{net}^{l}}{\partial \mathrm{a}_{t}^{l-1}} \frac{\partial \mathrm{a}_{t}^{l-1}}{\partial \mathrm{net}_{t}^{l-1}} \\
&=U \operatorname{diag}\left[f^{\prime l-1}\left(\operatorname{net}_{t}^{l-1}\right)\right]
\end{aligned}
$$
所以，
$$
\begin{aligned}
\left(\delta_{t}^{l-1}\right)^{T} &=\frac{\partial E}{\partial \operatorname{net}_{t}^{l-1}} \\
&=\frac{\partial E}{\partial \operatorname{net}_{t}^{l}} \frac{\partial{\mathrm{n} e t}_{t}^{l}}{\partial \operatorname{net}_{t}^{l-1}} \\
&=\left(\delta_{t}^{l}\right)^{T} U \operatorname{diag}\left[f^{\prime l-1}\left(\operatorname{net}_{t}^{l-1}\right)\right]
\end{aligned}
$$
上式就是将误差项传递到上一层算法。

#### 权重梯度的计算

首先，我们到目前为止，在前两步中已经计算得到的量，包括每个时刻$t$循环层的输出值$s_t$，以及误差项$\delta_t$。

![点击查看大图](/Users/idestro/PycharmProjects/Neural-Network/rnn/images/2256672-f7d034c8f05812f7.png)

只要知道了任意一个时刻的误差项$\delta_t$，以及上一个时刻循环层的输出值$s_{t-1}$，就可以按照下面的公式求出权重矩阵在$t$时刻的梯度$\nabla_{W t} E$:
$$
\nabla_{W_{t}} E=\left[\begin{array}{cccc}
\delta_{1}^{t} s_{1}^{t-1} & \delta_{1}^{t} s_{2}^{t-1} & \ldots & \delta_{1}^{t} s_{n}^{t-1} \\
\delta_{2}^{t} s_{1}^{t-1} & \delta_{2}^{t} s_{2}^{t-1} & \ldots & \delta_{2}^{t} s_{n}^{t-1} \\
\cdot & & & \\
\delta_{n}^{t} s_{1}^{t-1} & \delta_{n}^{t} s_{2}^{t-1} & \ldots & \delta_{n}^{t} s_{n}^{t-1}
\end{array}\right]
$$
在上式中，$\delta_i^t$表示$t$时刻误差项向量的第$i$个分量；$\delta_i^{t-1}$表示$t-1$时刻循环层第$i$个神经元的输出值。
$$
\begin{aligned}
\text { net }_{t} &=U_{\mathrm{x}_{t}}+W \mathrm{~s}_{t-1} \\
\left[\begin{array}{c}
n e t_{1}^{t} \\
n e t_{2}^{t} \\
\cdot \\
\cdot \\
n e t_{n}^{t}
\end{array}\right] &=U \mathrm{x}_{t}+\left[\begin{array}{cccc}
w_{11} & w_{12} & \ldots & w_{1 n} \\
w_{21} & w_{22} & \ldots & w_{2 n} \\
\cdot & & & \\
\cdot & & & \\
w_{n 1} & w_{n 2} & \ldots & w_{n n}
\end{array}\right]\left[\begin{array}{c}
s_{1}^{t-1} \\
s_{2}^{t-1} \\
\cdot \\
s_{n}^{t-1}
\end{array}\right] \\
&=U \mathrm{x}_{t}+\left[\begin{array}{c}
w_{11} s_{1}^{t-1}+w_{12} s_{2}^{t-1} \ldots w_{1 n} s_{n}^{t-1} \\
w_{21} s_{1}^{t-1}+w_{22} s_{2}^{t-1} \ldots w_{2 n} s_{n}^{t-1} \\
w_{n 1} s_{1}^{t-1}+w_{n 2} s_{2}^{t-1} \ldots w_{n n} s_{n}^{t-1}
\end{array}\right]
\end{aligned}
$$
因为对$W$求导与$U_{X_t}$无关，我们不考虑。现在，我们考虑对权重项$w_{ji}$求导。通过观察上式我们可以看到$w_{ji}$只与$net_j^t$有关，所以：
$$
\begin{aligned}
\frac{\partial E}{\partial w_{ji}}&=\frac{\partial E}{\partial net_j^t}\frac{\partial net_j^t}
{\partial w_{ji}} \\ 
&= \delta_j^t s_i^{t-1}
\end{aligned}
$$
按照上面的规律就可以生成式5里面的矩阵。我们已经求得了权重矩阵$W$在$t$时刻的梯度$\nabla_{W_t}E $，最终的梯度$\nabla_{W}E$是各个时刻的梯度之和：
$$
\begin{aligned}
\nabla_{W} E &=\sum_{i=1}^{t} \nabla_{W_{i}} E \\
&=\left[\begin{array}{cccccc}
\delta_{1}^{t} s_{1}^{t-1} & \delta_{1}^{t} s_{2}^{t-1} & \ldots & \delta_{1}^{t} s_{n}^{t-1} \\
\delta_{2}^{t} s_{1}^{t-1} & \delta_{2}^{t} s_{2}^{t-1} & \ldots & \delta_{2}^{t} s_{n}^{t-1} \\
\cdot & & & \\
\cdot & & & \\
\delta_{n}^{t} s_{1}^{t-1} & \delta_{n}^{t} s_{2}^{t-1} & \ldots & \delta_{n}^{t} s_{n}^{t-1}
\end{array}\right]+\ldots+\left[\begin{array}{cccc}
\delta_{1}^{1} s_{1}^{0} & \delta_{1}^{1} s_{2}^{0} & \ldots & \delta_{1}^{1} s_{n}^{0} \\
\delta_{2}^{1} s_{1}^{0} & \delta_{2}^{1} s_{2}^{0} & \ldots & \delta_{2}^{1} s_{n}^{0} \\
\cdot & & & \\
\cdot & & & \\
\delta_{n}^{1} s_{1}^{0} & \delta_{n}^{1} s_{2}^{0} & \ldots & \delta_{n}^{1} s_{n}^{0}
\end{array}\right]
\end{aligned}
$$
上式就是计算循环层权重矩阵$W$的梯度公式。

为什么最终的梯度是各个时刻的梯度之和呢？

我们还是从这个式子开始：
$$
net_t=Ux_t+Wf(net_{t-1})
$$
因为$U_{X_t}$与$W$完全无关，我们把它看做常量。右边的$W$与$f(net_{t-1})$都是$W$的函数，根据导数乘法原则：
$$
\frac{\partial \mathrm{net}_{t}}{\partial W}=\frac{\partial W}{\partial W} f\left(\mathrm{net}_{t-1}\right)+W \frac{\partial f\left(\mathrm{net}_{t-1}\right)}{\partial W}
$$
我们最终需要计算的是$\nabla_WE$：
$$
\begin{aligned}
\nabla_{W} E &=\frac{\partial E}{\partial W} \\
&=\frac{\partial E}{\partial \operatorname{net}_{t}} \frac{\partial \mathrm{net}_{t}}{\partial W} \\
&=\delta_{t}^{T} \frac{\partial W}{\partial W} f\left(\mathrm{net}_{t-1}\right)+\delta_{t}^{T} W \frac{\partial f\left(\mathrm{net}_{t-1}\right)}{\partial W}
\end{aligned}
$$
先计算式7加号左边的部分。$\frac{\partial W}{\partial W}$式矩阵对矩阵求导，其结果是一个四维张量(tensor)，如下所示：

![image-20210316111236544](/Users/idestro/PycharmProjects/Neural-Network/rnn/images/image-20210316111236544.png)

接下来，我们知道$s_{t-1}=f(net_{t-1})$，它是一个列向量。我们让上面的四维张量与这个向量相乘，得到了一个三维张量，再左乘行向量$\delta_t^T$，最终得到一个矩阵：

![image-20210316111534645](/Users/idestro/PycharmProjects/Neural-Network/rnn/images/image-20210316111534645.png)

接下来，我们计算右边的部分：
$$
\begin{aligned}
\delta_{t}^{T} W \frac{\partial f\left(\operatorname{net}_{t-1}\right)}{\partial W} &=\delta_{t}^{T} W \frac{\partial f\left(\operatorname{net}_{t-1}\right)}{\partial \operatorname{net}_{t-1}} \frac{\partial \operatorname{net}_{t-1}}{\partial W} \\
&=\delta_{t}^{T} W f^{\prime}\left(\operatorname{net}_{t-1}\right) \frac{\partial{\operatorname{net}}_{t-1}}{\partial W} \\
&=\delta_{t}^{T} \frac{\partial \mathrm{net}_{t}}{\partial \operatorname{net}_{t-1}} \frac{\partial \operatorname{net}_{t-1}}{\partial W} \\
&=\delta_{t-1}^{T} \frac{\partial \mathrm{net}_{t-1}}{\partial W}
\end{aligned}
$$
于是，我们得到递推公式：
$$
\begin{aligned}
\nabla_{W} E &=\frac{\partial E}{\partial W} \\
&=\frac{\partial E}{\partial \operatorname{net}_{t}} \frac{\partial \mathrm{net}_{t}}{\partial W} \\
&=\nabla_{W t} E+\delta_{t-1}^{T} \frac{\partial \operatorname{net}_{t-1}}{\partial W} \\
&=\nabla_{W t} E+\nabla_{W t-1} E+\delta_{t-2}^{T} \frac{\partial \mathrm{net}_{t-2}}{\partial W} \\
&=\nabla_{W t} E+\nabla_{W t-1} E+\ldots+\nabla_{W 1} E \\
&=\sum_{k=1}^{t} \nabla_{W k} E
\end{aligned}
$$
证毕。

同权重矩阵W类似，我们可以得到权重矩阵$U$的计算方法。

#### RNN的梯度爆炸和消失问题

不幸的是，实践中前面介绍的几种RNNs并不能很好的处理较长的序列。一个主要的原因是，RNN在训练中很容易发生**梯度爆炸**和**梯度消失**，这导致训练时梯度不能在较长序列中一直传递下去，从而使RNN无法捕捉到长距离的影响。

为什么RNN会产生梯度爆炸和消失问题呢？我们接下来将详细分析一下原因。我们根据**式3**可得：
$$
\begin{aligned}
\delta_{k}^{T} &=\delta_{t}^{T} \prod_{i=k}^{t-1} W \operatorname{diag}\left[f^{\prime}\left(\text { net }_{i}\right)\right] \\
\left\|\delta_{k}^{T}\right\| & \leqslant\left\|\delta_{t}^{T}\right\| \prod_{i=k}^{t-1}\|W\|\left\|\operatorname{diag}\left[f^{\prime}\left(\operatorname{net}_{i}\right)\right]\right\| \\
& \leqslant\left\|\delta_{t}^{T}\right\|\left(\beta_{W} \beta_{f}\right)^{t-k}
\end{aligned}
$$
上式的$\beta$定义为矩阵的模的上界。因为上式是一个指数函数，如果$t-k$很大的话（也就是向前看很远的时候），会导致对应的**误差项**的值增长或缩小的非常快，这样就会导致相应的**梯度爆炸**和**梯度消失**问题（取决于$\beta$大于1还是小于1）。

通常来说，**梯度爆炸**更容易处理一些。因为梯度爆炸的时候，我们的程序会收到NaN错误。我们也可以设置一个梯度阈值，当梯度超过这个阈值的时候可以直接截取。

**梯度消失**更难检测，而且也更难处理一些。总的来说，我们有三种方法应对梯度消失问题：

1. 合理的初始化权重值。初始化权重，使每个神经元尽可能不要取极大或极小值，以躲开梯度消失的区域。
2. 使用relu代替sigmoid和tanh作为激活函数。
3. 使用其他结构的RNNs，比如长短时记忆网络（LTSM）和Gated Recurrent Unit（GRU），这是最流行的做法。我们将在以后的文章中介绍这两种网络。

