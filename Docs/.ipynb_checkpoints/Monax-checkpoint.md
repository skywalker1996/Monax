#### **轻量级逻辑仿真平台-Monax**

##### 一、Monax概述

Monax是一个基于Mininet的轻量级逻辑仿真平台，采用模拟数据流的方式进行拥塞控制方面的仿真，用户也可以选择运行自定义的应用程序。主要特性有：

* 自动化应用部署，实现算法的快速验证

* 使用流控组件模拟动态带宽，可读取真实网络环境的带宽Trace

* Middle-ware模拟单跳路由功能，可定制数据包调度算法

  

##### 二、Monax使用

2.1 整体流程

* 在 `sim.py` 文件中，自定义网络拓扑结构和链路相关参数
* 在 `sim.py` 文件中，定义在各个节点上执行的命令（运行特定程序）
* `sudo python sim.py` 开始仿真实验
* 用户需要在自己的程序中使用数据持久化方案，才能提取出相关的实验数据（例如本地文件或数据库等）

