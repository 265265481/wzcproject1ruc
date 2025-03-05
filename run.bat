@echo off
chcp 65001

type kk.txt

set /p id=请选择合适的模式序号：
if /i %id%==1 set mode="INT4"
if /i %id%==2 set mode="INT8"
if /i %id%==3 set mode="FP16"
if /i %id%==4 set mode="CPU32"
if /i %id%==5 set mode=""

.\python310\python.exe ./web_demo.py %mode%

:end
pause
exit /b