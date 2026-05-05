@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================
echo   DanbooruSearch 本地版启动器
echo ============================================
echo.

:: ── 检测 GPU 与 CUDA 版本 ──────────────────────────────────────────────
set "TORCH_CUDA="

where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 未检测到 nvidia-smi，将安装 CPU 版 PyTorch。
    echo        如有独立显卡，请确认 NVIDIA 驱动已正确安装。
    goto :install_torch
)

echo [检测] 正在读取 GPU 信息...
for /f "delims=" %%i in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do (
    echo [检测] GPU: %%i
)

:: 从 nvidia-smi 输出中提取 CUDA 版本（格式如 "CUDA Version: 12.8"）
for /f "tokens=6 delims= " %%v in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do (
    set "CUDA_VER=%%v"
)

if not defined CUDA_VER (
    echo [提示] 无法解析 CUDA 版本，将安装 CPU 版 PyTorch。
    goto :install_torch
)

echo [检测] CUDA 版本: %CUDA_VER%

:: 取主版本号（如 12.8 → 12, 13.0 → 13）
for /f "tokens=1 delims=." %%m in ("%CUDA_VER%") do set "CUDA_MAJOR=%%m"

if "%CUDA_MAJOR%" geq "12" (
    set "TORCH_CUDA=cu128"
    echo [匹配] PyTorch → cu128
    goto :install_torch
)

if "%CUDA_MAJOR%"=="11" (
    set "TORCH_CUDA=cu118"
    echo [匹配] PyTorch → cu118
    goto :install_torch
)

echo [提示] CUDA %CUDA_VER% 版本较旧，将安装 CPU 版 PyTorch。

:install_torch
echo.
if defined TORCH_CUDA (
    echo [安装] 正在安装 PyTorch (GPU 版, %TORCH_CUDA%)...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/%TORCH_CUDA% --quiet
) else (
    echo [安装] 正在安装 PyTorch (CPU 版)...
    python -m pip install torch torchvision --quiet
)

:: ── 安装其余依赖 ──────────────────────────────────────────────────────
echo.
echo [安装] 正在安装其余依赖...
python -m pip install -r requirements.txt --quiet

:: ── 启动 ──────────────────────────────────────────────────────────────
echo.
echo [启动] 正在启动 DanbooruSearch...
echo.
python ui_nicegui.py

pause
