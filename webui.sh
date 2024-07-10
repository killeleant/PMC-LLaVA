#!/bin/bash

# 启动第一个Python脚本并在后台运行
python utils.py &

# 启动第二个Python脚本并在后台运行
python demo.py &

# 使用`wait`命令确保脚本在所有后台进程结束后才退出
wait
