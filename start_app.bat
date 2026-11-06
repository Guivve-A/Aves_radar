@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [INFO] Entorno virtual no encontrado. Ejecutando setup_env.bat...
    call setup_env.bat
    if errorlevel 1 (
        echo [ERROR] Fallo la configuracion del entorno.
        exit /b 1
    )
)

call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual.
    exit /b 1
)

echo [INFO] Verificando entorno...
python test_install.py
if errorlevel 1 (
    echo [ERROR] El smoke test fallo. Revisa dependencias.
    exit /b 1
)

echo [INFO] Iniciando EcoAcoustic Sentinel...
python main.py

endlocal
