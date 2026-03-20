@echo off
cd /d C:\PROJETO\dataError_IC-master

echo.
set /p msg=Digite a mensagem do commit: 

git status
git add .
git commit -m "%msg%"
git push origin main

echo.
pause