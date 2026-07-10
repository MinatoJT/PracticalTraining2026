@echo off
setlocal

rem Portable WinUI 3 launcher. Requires Visual Studio/Build Tools MSBuild.
set "SCRIPT_DIR=%~dp0"
set "PROJECT_FILE=%SCRIPT_DIR%CRAGMM.WinUI.csproj"
set "APP_EXE=%SCRIPT_DIR%bin\x64\Release\net8.0-windows10.0.19041.0\win-x64\CRAGMM.WinUI.exe"

set "MSBUILD_EXE="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.ComponentGroup.UWP.BuildTools -find MSBuild\Current\Bin\MSBuild.exe`) do (
        set "MSBUILD_EXE=%%i"
    )
)

if not defined MSBUILD_EXE if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe" (
    set "MSBUILD_EXE=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
)

if not defined MSBUILD_EXE if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set "MSBUILD_EXE=%ProgramFiles%\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
)

if not defined MSBUILD_EXE (
    echo MSBuild with UWP/WinUI build tools was not found.
    echo Install Visual Studio Build Tools 2022 with Microsoft.VisualStudio.ComponentGroup.UWP.BuildTools.
    pause
    exit /b 1
)

"%MSBUILD_EXE%" "%PROJECT_FILE%" /restore /p:Configuration=Release /p:Platform=x64
if not "%errorlevel%"=="0" (
    pause
    exit /b %errorlevel%
)

start "" "%APP_EXE%"
exit /b 0
