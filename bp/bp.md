## 神经网络，反向传播算法，手写数字识别

### 神经网络

### 反向传播算法

反向传播算法为何“反向”？为何“传播”呢？接下来我会慢慢揭开这层面纱。首先定义两组向量，分别为$\vec{y}$与$\vec{t}$，其中$\vec{y}$指的是神经网络最后一层的输出值，$\vec{t}$代表目标值。

我们取网络所有输出层节点的误差平方和作为损失函数，则样本$d$对应的损失为：
$$
E_d=\frac{1}{2}\sum_{i \in outputs}(t_i-y_i)^2
$$
使用梯度下降算法更新参数，有：
$$
w_{ji}=w_{ji}-\mu\frac{\partial E_d}{w_{ji}}
$$
设$net_j$为节点$j$的输入，$\alpha_j$为节点$j$的输出，有：
$$
net_j=\vec{w_j}\vec{x_j}=\sum_iw_{ji}x_{ji}
$$
$E_d$是$net_j$的函数，$net_j$是$w_{ji}$的函数，根据链式求导规则，有：
$$
\begin{aligned}
\frac{\partial E_d}{\partial w_{ji}}&=\frac{\partial E_d}{\partial net_j}\frac{\partial net_j}{\partial w_{ji}} \\
&=\frac{\partial E_d}{\partial net_j}\frac{\partial \sum_iw_{ji}x_{ji}}{\partial w_{ji}} \\
&=\frac{\partial E_d}{\partial net_j}x_{ji}
\end{aligned}
$$
上式中，$x_{ji}$是节点$i$传输给节点$j$的输入值，也是节点$i$的输出值。

对于$\frac{\partial E_d}{\partial net_j}$要分输出层与隐藏层两层情况。

#### 输出层权值训练

对于输出层来说，$net_j$仅能通过节点$j$的输出值$y_j$来影响网络其他部分，也就是说$E_d$是$y_j$的函数，而$y_j$是$net_j$的函数，其中设$y_j=sigmoid(net_j)$。所以我们可以再次使用链式求导法则：
$$
\begin{aligned}
\frac{\partial E_d}{\partial net_j}&=\frac{\partial E_d}{\partial y_j}\frac{\partial y_j}{\partial net_j}\\
&=-(t_j-y_j)\frac{\partial y_j}{\partial net_j}\\
&=-(t_j-y_j)y_j(1-y_j)
\end{aligned}
$$
令$\delta_j=-\frac{\partial E_d}{\partial net_j}$，也就是一个节点的误差项$\delta$是网络误差对这个节点输入的偏导数的。带入上式，得到：
$$
\delta_j=(t_j-y_j)y_j(1-y_j)
$$
带入梯度下降公式得到：
$$
\begin{aligned}
w_{ji}&=w_{ji}-\mu\frac{\partial E_d}{\partial w_{ji}} \\ 
&=w_{ji}+\mu\delta_jx_{ji}
\end{aligned}
$$

#### 隐藏层训练

首先，我们需要定义节点$j$的所有下游节点的集合$Downstream(j)$。可以看到$net_j$只能通过影响$Downstream(j)$再影响$E_d$。设$net_k$是节点$j$的下游节点的输入，则$E_d$是$net_k$的函数，而$net_k$是$net_j$的函数。因为$net_k$有多个，我们应用全导数公式，有如下推导：
$$
\begin{aligned}
\frac{\partial E_d}{\partial net_j}&=\sum_{k \in Downstream}\frac{\partial E_d}{\partial net_k}\frac{\partial net_k}{\partial net_j}\\
&=\sum_{k \in Downstream}\frac{\partial E_d}{\partial net_k}\frac{\partial net_k}{\partial \alpha_j}\frac{\partial \alpha_j}{\partial net_j}\\
&=\sum_{k \in Downstream}-\delta_kw_{kj}\alpha_j(1-\alpha_j)\\
&=-\alpha_j(1-\alpha_j)\sum_{k \in Downstream}\delta_kw_{kj}\\
\end{aligned}
$$
因为$\delta_j=-\frac{\partial E_d}{\partial net_j}$ ，故有
$$
\delta_j=\alpha_j(1-\alpha_j)\sum_{k \in Downstream}\delta_kw_{kj}
$$

#### 总结

反向针对的是梯度更新的“方向”，从神经网络的右边更新至左边，那“传播”的是什么呢？经过观察，前一层需要用到上一层的误差项$\delta$，所以“反向传播”的下一层所有的误差项$\delta$。

### 向量化编程

首先，我们需要把所有的计算都表达为向量的形式。对于全连接神经网络来说，主要有三个计算公式。

前向计算，
$$
\vec{a}=\sigma(W \cdot \vec{x})
$$
上式中的$\sigma$函数。

反向计算，
$$
\vec{\delta}=\vec{y}(1-\vec{y})(\vec{t}-\vec{y})
$$

$$
\vec{\delta}^{(l)}= \vec{a}^{(l)} (1-\vec{a}^{(l)}){W}^{T} \delta^{l+1}
$$

其中，$\delta^{l}$表示第$l$层的误差项。

梯度下降，
$$
W = W+\mu \vec{\delta}\vec{x}^{T}
$$

$$
b = b + \mu \vec{\delta}
$$

