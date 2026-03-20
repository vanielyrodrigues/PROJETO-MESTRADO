@echo off
cd /d C:\PROJETO\dataError_IC-master

echo.
set /p msg=Digite a mensagem do commit: 

:: Captura data no formato YYYY-MM-DD
for /f "tokens=1-3 delims=/" %%a in ("%date%") do set DATA=%%c-%%b-%%a

:: Captura hora no formato HH-MM
for /f "tokens=1-2 delims=:" %%a in ("%time%") do set HORA=%%a-%%b

echo.
echo Enviando com mensagem:
echo %msg% - %DATA% %HORA%
echo.

git status
git add .
git commit -m "%msg% - %DATA% %HORA%"
git push origin main

echo.
pause