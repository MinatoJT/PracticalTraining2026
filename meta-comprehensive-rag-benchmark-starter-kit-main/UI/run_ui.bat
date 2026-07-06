@echo off
set PYTHONDONTWRITEBYTECODE=1
set PANDAS_USE_NUMEXPR=0
set PANDAS_USE_BOTTLENECK=0
set HF_HOME=%~dp0..\Dataset\hf_home
set HF_DATASETS_CACHE=%~dp0..\Dataset\hf_datasets
set HUGGINGFACE_HUB_CACHE=%~dp0..\Dataset\hf_hub
set HF_XET_CACHE=%~dp0..\Dataset\hf_xet
set TRANSFORMERS_CACHE=%~dp0..\Dataset\transformers
set SENTENCE_TRANSFORMERS_HOME=%~dp0..\Dataset\sentence_transformers
set CRAG_CACHE_DIR=%~dp0..\Dataset\crag_images
set CRAG_WEBSEARCH_CACHE_DIR=%~dp0..\Dataset\crag_web_search
cd /d "%~dp0.."
C:\anaconda\python.exe -B UI\app.py
pause

