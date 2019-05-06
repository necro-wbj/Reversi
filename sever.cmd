@echo off
set FLASK_APP=Game_online.py
set FLASK_ENV=development
python -m flask run --host=0.0.0.0 --no-reload
pause