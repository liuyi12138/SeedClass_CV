# PPT丢失说明

## 缘由

王老师，十分抱歉，**因为我在处理git conflict时做出了不慎的操作**，导致我们组的最新版本PPT丢失且**无法恢复**！当日我也已经通过QQ向您说明了此原因。

## 尝试恢复

为了恢复文件，我们尝试了以下工具和方法：

- **通过系统级文件恢复工具恢复：**
  - [Disk Drill](https://www.cleverfiles.com/cn/disk-drill-windows.html)，能找到PPT文件，但是文件受到损坏，且无法修复
  - [Restoration](http://download.cnet.com/Restoration/3000-2094_4-10322950.html)，无法找到PPT文件
  - [EaseUS数据恢复精灵](https://www.easeus.com/datarecoverywizard/free-data-recovery-software.htm)，能找到PPT文件，但是文件受到损坏，且无法修复
  - [Recuva](https://en.wikipedia.org/wiki/Recuva)，无法找到PPT文件
- **通过.git文件夹尝试恢复文件**（因为我在本地有commit记录）
  - 通过[EaseUS数据恢复精灵](https://www.easeus.com/datarecoverywizard/free-data-recovery-software.htm)恢复了.git文件夹，但是在恢复时出现了git bad signature的报错

## 我们PPT的大致内容

- **神经网络的构建：**
  - 网络的结构图，符号声明
  - forward，backward网络的数学推导
  - 相应的代码实现展示
- **神经网络的训练：**
  - 最佳超参数搜索，搜索了网络shape，learning_rate和learning_decay
  - 参数初始化的数学讨论，根据其数学分布列完成了代码层次的实现
  - 激活函数的讨论，讨论了激活函数对于梯度传递的重要影响
  - Optimizer的讨论，给出了三种optimizer的数学表达式，并解释了其作用
- **实验结果和总结**
  - 给出了half moon的可视化结果
  - 给出了Xavier和He初始化的可视化结果比较
  - 给出了tanh，relu，leaky_relu和elu的可视化结果比较
  - 给出了BGD, Momentum和Adam的可视化结果比较
  - 最后总结了我们在本次实验中的收获和感触
- **王老师您的提问：**
  - 关于数学推导
  - 关于“分布列”的概念
  - 关于如何实现两种参数初始化