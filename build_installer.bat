@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "dist\EcoAcousticSentinel\EcoAcousticSentinel.exe" goto missing_build

set "ISCC_CMD="

where iscc >nul 2>nul
if not errorlevel 1 set "ISCC_CMD=iscc"

if not defined ISCC_CMD if defined ProgramFiles(x86) if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_CMD=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC_CMD if defined ProgramFiles if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_CMD=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if not defined ISCC_CMD goto missing_iscc

echo [INFO] Compilando instalador con:
echo        %ISCC_CMD%
"%ISCC_CMD%" "installer.iss"
if errorlevel 1 goto compile_failed

echo.
echo [OK] Instalador generado.
echo      Revisa la carpeta:
echo      %CD%\dist
exit /b 0

:missing_build
echo [ERROR] No se encontro el build de la app.
echo         Ejecuta primero: python build_exe.py
exit /b 1

:missing_iscc
echo [ERROR] No se encontro ISCC.exe (Inno Setup Compiler).
echo         Instala Inno Setup 6 y vuelve a intentar.
echo         Descarga: https://jrsoftware.org/isdl.php
exit /b 1

:compile_failed
echo [ERROR] Fallo la compilacion de installer.iss
exit /b 1
