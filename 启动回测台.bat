@echo off
chcp 65001 >nul
title A股回测台 - 缩量企稳反转策略
cd /d "%~dp0"

echo ============================================================
echo   A股「缩量企稳反转」策略 交互式回测台
echo ============================================================
echo.
echo   正在加载 1,146 万行日线数据，首次启动约需 30-60 秒...
echo   加载完成后浏览器会自动打开。
echo.
echo   关闭本窗口即可停止服务。
echo ============================================================
echo.

start "" http://127.0.0.1:8760
"C:\Users\50651\.workbuddy\binaries\python\envs\default\Scripts\python.exe" -u src\server.py 8760

echo.
echo 服务已停止。按任意键关闭窗口。
pause >nul
