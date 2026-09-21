@echo off
setlocal
set HERE=%~dp0
set PYTHONPATH=%HERE%src;%HERE%vendor\leapct;%HERE%vendor\xraylib;%HERE%vendor\cuda;%PYTHONPATH%
python "%HERE%src\simulate_ct6.py" %*
