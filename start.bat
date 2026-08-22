@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "POWERSHELL_EXIT_FLAG=-NoExit"
for %%A in (%*) do (
    if /I "%%~A"=="-ValidateOnly" set "POWERSHELL_EXIT_FLAG="
    if /I "%%~A"=="/ValidateOnly" set "POWERSHELL_EXIT_FLAG="
    if /I "%%~A"=="-SmokeTest" set "POWERSHELL_EXIT_FLAG="
    if /I "%%~A"=="/SmokeTest" set "POWERSHELL_EXIT_FLAG="
)
pushd "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile %POWERSHELL_EXIT_FLAG% -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
set "START_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %START_EXIT_CODE%
