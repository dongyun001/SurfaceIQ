@echo off
title SurfaceIQ Rect Measure
cd /d "%~dp0"
if "%~1"=="" (
    start "" venv\Scripts\pythonw.exe src\gui_ring.py --shape rect
) else (
    if "%~x1"=="" (
        start "" venv\Scripts\pythonw.exe src\gui_ring.py --shape rect %*
    ) else (
        start "" venv\Scripts\pythonw.exe src\gui_ring.py --shape rect --image "%~1"
    )
)
