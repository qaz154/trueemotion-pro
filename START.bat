@echo off
REM TrueEmotion Pro v1.15 启动脚本
cd /d "%~dp0"
echo ================================================
echo TrueEmotion Pro v1.15
echo ================================================
echo.
echo 使用方法:
echo.
echo   cd src
echo   python -c "from trueemotion.api import analyze; print(analyze('工作好累').reply)"
echo.
echo 或运行测试:
echo   python tests	est_integration.py
echo.
pause
