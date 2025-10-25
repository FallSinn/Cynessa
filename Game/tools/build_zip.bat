@echo off
setlocal
set ROOT=%~dp0..
pushd %ROOT%
if exist ..\dist mkdir ..\dist >nul 2>&1
powershell -Command "Compress-Archive -Path 'Game' -DestinationPath '..\\dist\\Game_Portable.zip' -Force"
popd
endlocal
