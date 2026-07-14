@echo off
setlocal

rem Locate the project from this bat file. No machine-specific absolute path.
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PANDAS_USE_NUMEXPR=0"
set "PANDAS_USE_BOTTLENECK=0"
set "HF_HOME=%PROJECT_DIR%\Dataset\hf_home"
set "HF_DATASETS_CACHE=%PROJECT_DIR%\Dataset\hf_datasets"
set "HUGGINGFACE_HUB_CACHE=%PROJECT_DIR%\Dataset\hf_hub"
set "HF_XET_CACHE=%PROJECT_DIR%\Dataset\hf_xet"
set "TRANSFORMERS_CACHE=%PROJECT_DIR%\Dataset\transformers"
set "SENTENCE_TRANSFORMERS_HOME=%PROJECT_DIR%\Dataset\sentence_transformers"
set "CRAG_CACHE_DIR=%PROJECT_DIR%\Dataset\crag_images"
set "CRAG_WEBSEARCH_CACHE_DIR=%PROJECT_DIR%\Dataset\crag_web_search"

rem Select a Python that actually contains both UI and evaluation dependencies.
set "PYTHON_EXE="
set "PYTHON_PREFIX="
if defined CRAGMM_PYTHON call :try_python "%CRAGMM_PYTHON%"
if not defined PYTHON_EXE if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" call :try_python "%PROJECT_DIR%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" call :try_python "%CONDA_PREFIX%\python.exe"
if not defined PYTHON_EXE if exist "C:\anaconda\python.exe" call :try_python "C:\anaconda\python.exe"
if not defined PYTHON_EXE for /f "delims=" %%i in ('where python 2^>nul') do if not defined PYTHON_EXE call :try_python "%%i"
if not defined PYTHON_EXE call :try_py

if not defined PYTHON_EXE (
    echo No Python environment with PySide6 and datasets was found.
    echo Run: python -m pip install -r "%PROJECT_DIR%\requirements.txt"
    echo Or set CRAGMM_PYTHON to the correct python.exe.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" %PYTHON_PREFIX% -B "%SCRIPT_DIR%app.py"
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:try_python
"%~1" -c "import PySide6, datasets" >nul 2>nul
if "%errorlevel%"=="0" set "PYTHON_EXE=%~1"
exit /b 0

:try_py
where py >nul 2>nul
if not "%errorlevel%"=="0" exit /b 0
py -3 -c "import PySide6, datasets" >nul 2>nul
if "%errorlevel%"=="0" (
    set "PYTHON_EXE=py"
    set "PYTHON_PREFIX=-3"
)
exit /b 0
