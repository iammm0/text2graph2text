@echo off
:: 激活 Miniconda（请确保路径无误）
CALL "C:\Users\Mingjun Zhao\miniconda3\Scripts\activate.bat"

:: 激活 grownet 环境
CALL conda activate grownet

:: 进入你的项目根目录
cd /d "C:\Users\Mingjun Zhao\PycharmProjects\GrowNet"

:: 运行入口脚本（替换成你实际要跑的，比如 train.py、main.py、app.py）
python main.py

:: 保持窗口打开，方便你看输出
pause
