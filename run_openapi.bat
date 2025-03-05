@echo off
chcp 65001


type 1.txt

.\python310\python.exe .\openai_api.py

:end
pause
exit /b