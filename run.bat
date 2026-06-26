@echo off
REM =====================================================================
REM  Chatbot RAG pedagogique - lancement Windows
REM =====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === [1/5] Verification de Python ===
where python >nul 2>&1
if errorlevel 1 (
    echo ERREUR : Python introuvable dans le PATH. Installez Python 3.11+.
    pause & exit /b 1
)
python --version

echo.
echo === [2/5] Environnement virtuel ===
if not exist "venv\Scripts\python.exe" goto :make_venv
venv\Scripts\python.exe -c "import fastapi, faiss, sentence_transformers, huggingface_hub, multipart" >nul 2>&1
if errorlevel 1 goto :install_deps
echo venv OK.
goto :check_env

:make_venv
echo Creation du venv (quelques minutes)...
if exist "venv" rmdir /s /q "venv"
python -m venv venv
if errorlevel 1 ( echo ERREUR creation venv. & pause & exit /b 1 )

:install_deps
echo Installation / mise a jour des dependances...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 ( echo ERREUR installation dependances. & pause & exit /b 1 )

:check_env
echo.
echo === [3/5] Verification du token Hugging Face ===
findstr /c:"hf_xxxxxxxx" "app\.env" >nul 2>&1
if not errorlevel 1 (
    echo  /!\ Renseignez HUGGINGFACEHUB_API_TOKEN dans app\.env avant de lancer.
    pause & exit /b 1
)

echo.
echo === [4/5] Construction de l'index (ingestion des documents) ===
cd app
if not exist "index_store\faiss.index" (
    "..\venv\Scripts\python.exe" ingest.py
    if errorlevel 1 ( echo ERREUR ingestion. & pause & exit /b 1 )
) else (
    echo Index existant detecte. Pour le reconstruire : supprimez app\index_store ou appelez POST /reindex.
)

echo.
echo === [5/5] Demarrage du serveur : http://localhost:8002 ===
"..\venv\Scripts\python.exe" main.py
pause
