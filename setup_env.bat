@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Ejecutar desde la carpeta del script
cd /d "%~dp0"

echo [1/6] Detectando Python...
set "PYTHON_CMD="
set "PY_VER="

REM 1) Intentar py launcher con 3.11 primero
where py >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=1,2 delims= " %%a in ('py -3.11 --version 2^>^&1') do (
        if /i "%%a"=="Python" (
            set "PYTHON_CMD=py -3.11"
            set "PY_VER=%%b"
        )
    )
)

REM 2) Si no existe py -3.11, buscar python en PATH
if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        for /f "tokens=1,2 delims= " %%a in ('python --version 2^>^&1') do (
            if /i "%%a"=="Python" set "PY_VER=%%b"
        )
    )
)

if not defined PYTHON_CMD (
    call :print_red "ERROR FATAL: NO SE DETECTO PYTHON 3.11.X EN EL SISTEMA."
    echo Instala Python 3.11.x 64-bit y vuelve a ejecutar este script.
    exit /b 1
)

if not defined PY_VER (
    call :print_red "ERROR FATAL: NO SE PUDO LEER LA VERSION DE PYTHON."
    echo Verifica la instalacion de Python 3.11.x y vuelve a intentar.
    exit /b 1
)

for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

echo     Python detectado: !PY_VER!
if not "!PY_MAJOR!.!PY_MINOR!"=="3.11" (
    call :print_red "ERROR FATAL: VERSION DE PYTHON NO SOPORTADA !PY_VER!."
    echo Runtime obligatorio: Python 3.11.x por compatibilidad con BirdNET/TensorFlow/PyTorch.
    exit /b 1
)

echo [2/6] Creando entorno virtual...
if not exist "venv" (
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Fallo al crear el entorno virtual.
        exit /b 1
    )
) else (
    echo     venv ya existe; se reutilizara.
)

echo [3/6] Activando entorno virtual...
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] No se pudo activar venv.
    exit /b 1
)

echo [4/6] Actualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Fallo al actualizar pip.
    exit /b 1
)

echo [5/6] Instalando setuptools y wheel...
python -m pip install wheel setuptools
if errorlevel 1 (
    echo [ERROR] Fallo al instalar wheel/setuptools.
    exit /b 1
)

echo [6/6] Instalando dependencias del proyecto...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo al instalar requirements.txt.
    exit /b 1
)

echo.
echo Entorno listo.
echo Activalo cuando lo necesites con:
echo     call venv\Scripts\activate

endlocal
exit /b 0

:print_red
set "MSG=%~1"
where powershell >nul 2>nul
if errorlevel 1 (
    echo %MSG%
    exit /b 0
)
powershell -NoProfile -Command "Write-Host '%MSG%' -ForegroundColor Red"
if errorlevel 1 echo %MSG%
exit /b 0
