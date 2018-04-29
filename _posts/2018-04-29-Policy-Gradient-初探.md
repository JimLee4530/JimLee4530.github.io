---
layout: post
excerpt: Policy Gradient 是 RL 算法中一个基础的模型。本文会结合gym中的雅达利游戏Pong-v0做简要的说明。
permalink: /Policy Gradient 初探
published: true
images:
  - url: http://karpathy.github.io/assets/rl/pong.gif
---

# Policy Gradient 初探
## 简介
Policy Gradient 是 RL 算法中一个基础的模型。我们将结合游戏例子进行说明，以Pong游戏为例，Pong游戏界面中有左右两个白条，分别代表游戏双方，他们打网球一样，当一方就不到球时，另一方得分，先获得21分的一方获胜。

![](http://karpathy.github.io/assets/rl/pong.gif)

我们在玩pong游戏的时候会根据球飞来的位置控制自己的球拍作出向上或者向下的操作。正确的操作会使得我们接到球，然后打回去，对方接不到的话就赢了游戏。相反，错误的动作会使得我们接不到球，输掉游戏。在这里我们为了易于表示做出以下定义：
- 球飞来的位置，以及自己球拍的位置等当前游戏环境定义为：**state**， 用**s**表示。
- 根据**state**进行的操作定义为：**action**, 用**a**表示。
- 将每次**action**得到的结果定义为：**reward**，用**r**表示。

## Let's Play The Game
现在开始玩游戏了！
假设我们玩了4个回合，每一回合的具体操作如下：

![](http://karpathy.github.io/assets/rl/episodes.png)

我们将游戏中一个回合的过程定义为：**trajectory**， 用 $\tau$ 表示。

- $$\tau = \{s_1, a_1, r_1, s_2, a_2, r_2, ...,s_T, a_T, r_T\}$$
- 该回合游戏结果 为每一次操作的 **reward** 加权和(如果简单相加可以将$\gamma$看做1)：$$R(\tau) = \sum_{t=1}^{T}(\gamma^{t-1}r_t)$$
- 将模型根据当前state生成action的函数，也就是我们的 **policy** 定义*$\pi_\theta$*为：$p(a_t|s_t, \theta)$


当我们设计一个模型来让他自己玩游戏时，我们将模型的参数定义为 **$\theta$** , 而每个回合过程的概率为：
$$
P(\tau|\theta) = p(s_1)p(a_1|s_1, \theta)p(r_1, s_2,|s_1, a_1)p(a_2|s_2,\theta)p(r_2, s_3|a_2, s_2)...
=p(s_1)\prod_{t=1}^{T}{p(a_t|s_t, \theta)p(r_t, s_{t+1}|s_t, a_t)}
$$
根据上式，我们可以清楚，黄色部分是由模型参数控制的，而绿色部分与模型参数无关。所以黄色部分就可以作为需要去学习的模型，也就我们的 **policy** ($\pi_\theta$)。

![](..\images\2018-04-29\TIM20180429170210.png)

那么当前模型，整局游戏(包含所有可能回合)中的游戏结果 $R(\theta)$ 为：

$$R(\theta) = \sum_{\tau}{R(\tau)P(\tau|\theta)}\approx\frac{1}{N}\sum_{n=1}^{N}R(\tau^n)$$

- 由于不能枚举所有的情况，所以随机sample出N个回合，对 $R(\theta)$ 进行估计。

## How Can We Play Better?

那，我们如何玩的更好呢？

最直接的做法就是最大化 $R(\theta)$ 的值， 用常见的SGD来求的话，公式表达如下：

$$R(\theta) = \mathbb{E}[r_1+\gamma r_2+\gamma^2 r_3+...|\pi(·, \theta)] \\ \theta^* = \arg \max_\theta(R(\theta)) \\ \theta{}' \leftarrow \theta + \eta \nabla R(\theta)$$

接下来就是计算梯度 $\nabla R(\theta)$ :

$$\nabla R(\theta) = \sum_{\tau}R(\tau)\nabla P(\tau|\theta) = \sum_{\tau}R(\tau)P(\tau|\theta) \frac{\nabla P(\tau|\theta)}{P(\tau|\theta)} \\ =  \sum_{\tau}R(\tau)P(\tau|\theta) \nabla \log P(\tau|\theta) \\ \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau^n)\nabla \log P(\tau|\theta) $$

$$\because  P(\tau|\theta) =p(s_1)\prod_{t=1}^{T}{p(a_t|s_t, \theta)p(r_t, s_{t+1}|s_t, a_t)} \\ \log P(\tau|\theta) = \log(p(s_1)) + \sum_{t=1}^{T} \log p(a_t | s_t, \theta) \\ \nabla \log P(\tau|\theta) = \sum_{t=1}^{T} \nabla \log p(a_t | s_t, \theta)$$

$$\therefore \nabla R(\theta) \approx \frac{1}{N}\sum_{n=1}^{N}R(\tau^n)\nabla \log P(\tau|\theta) \\ = \frac{1}{N}\sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n)\nabla \log P(a_t | s_t, \theta)$$

其物理意义是，

- 当 $R(\tau^n) > 0$ 时， 为使$R(\theta)$ 加大，调整 $\theta$ 加大 $P(a_t | s_t, \theta)$ ,从而达到效果。
- 当 $R(\tau^n) < 0$ 时， 为使$R(\theta)$ 加大，调整 $\theta$ 减小 $P(a_t | s_t, \theta)$ ,从而达到效果。

引用一张[Andrej Karpathy blog](http://karpathy.github.io/)博客的图来加以说明：

![](http://karpathy.github.io/assets/rl/pg.png)

- 左图，高斯分布的选取点求梯度的图，箭头指向的方向都是函数上升的方向。
- 中图，加入reward之后，会提示选取点哪些方向是不好的，比如红色方向可能会导致函数值下降，所以需要走反方向，而绿色点的方向是函数值上升的方向。
- 右图，经过参数更新后，所有点都向绿色的区域靠近，因为那里可以使函数值上升。

知道了学习的过程，我们将玩游戏过程结合在一起就是下图：

![](..\images\2018-04-29\TIM20180429213834.png)

## Code

根据原理，我们就可以尝试去写一个基于Pytorch的Policy Gradient模型去玩Pong游戏了。可以在[gym](https://gym.openai.com/)上找到对应环境。

[Policy Gradient](https://github.com/JimLee4530/Policy-Gradient.pytorch)

## 参考

[Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)

[莫烦Python](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-A-PG/)

[Applied Deep Learning /Machine Learning and Having It Deep and Structured](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/syllabus.html)