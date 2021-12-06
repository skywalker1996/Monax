# Monax-轻量级逻辑仿真平台

#### 代码结构概述

1. 文件夹分布：

* --configs：存放配置信息，global.conf为主要配置文件，需要手动配置各个组件的信息。

* --lib：主要存放功能函数，其中rate_control为率控制算法的代码 

* --traces: 存放带宽的trace文件

* --utils：存放其他工具函数

* --results：存放实验结果



2. 主要组件

* monax_server：负责发送数据流，其中率控制算法就集成在server端中。
* monax_client: 客户端，接收数据并返回ACK。
* run_experiment.py: 自动化实验脚本


#### 直接使用

`python run_experiment.py`

注：其中使用的数据库是influxdb2.0，具体的配置信息在global.conf中，目前使用的是在阿里云上搭建的数据库服务器。
