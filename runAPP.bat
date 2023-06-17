@echo off

echo  http://localhost:8000
cmd /k "conda activate yolo7 & flask run"

@REM pytest -v --disable-warnings

