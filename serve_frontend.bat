@echo off
echo Starting minimal frontend server...
cd /d %~dp0
python -m http.server 3000 --directory minimal_frontend
pause