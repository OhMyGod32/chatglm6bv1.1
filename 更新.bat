@echo off
git pull
.\venv\scripts\pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
exit