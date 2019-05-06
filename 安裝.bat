@echo off
:::::::::::::::::::::::::::::::::獲取管理員權限::::::::::::::::::::::::::::::::::::::::::::::::::
:init
setlocal DisableDelayedExpansion
set "batchPath=%~0"
for %%k in (%0) do set batchName=%%~nk
set "vbsGetPrivileges=%temp%\OEgetPriv_%batchName%.vbs"
setlocal EnableDelayedExpansion
:checkPrivileges
NET FILE 1>NUL 2>NUL
if '%errorlevel%' == '0' ( goto gotPrivileges ) else ( goto getPrivileges )
:getPrivileges
if '%1'=='ELEV' (echo ELEV & shift /1 & goto gotPrivileges)
ECHO Set UAC = CreateObject^("Shell.Application"^) > "%vbsGetPrivileges%"
ECHO args = "ELEV " >> "%vbsGetPrivileges%"
ECHO For Each strArg in WScript.Arguments >> "%vbsGetPrivileges%"
ECHO args = args ^& strArg ^& " "  >> "%vbsGetPrivileges%"
ECHO Next >> "%vbsGetPrivileges%"
ECHO UAC.ShellExecute "!batchPath!", args, "", "runas", 1 >> "%vbsGetPrivileges%"
"%SystemRoot%\System32\WScript.exe" "%vbsGetPrivileges%" %*
exit /B
:gotPrivileges
setlocal & pushd .
cd /d %~dp0
if '%1'=='ELEV' (del "%vbsGetPrivileges%" 1>nul 2>nul  &  shift /1)
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
pip3 --version 2>nul
if %errorlevel% NEQ 0 (
    pip3 --version 2>nul
    if %errorlevel% NEQ 0 (
	echo 找不到
	pause
        start https://www.python.org/
    ) else (
        goto pok
    )
) ELSE (
   goto pok
)
:p
cls
echo python可以使用後程式將繼續執行
pip3 --version 2>nul
if %errorlevel% NEQ 0 goto p
:pok
python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))" >nul
if %errorlevel% NEQ 0 pip install tensorflow
python -m pip install numpy
REM if %errorlevel% EQU　0 exit
pause