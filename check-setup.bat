@echo off
echo Checking Caddy Reverse Proxy Setup...

echo 1. Checking container status...
docker-compose ps

echo.
echo 2. Testing Caddy health endpoint...
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8080/health' -UseBasicParsing"

echo.
echo 3. Testing frontend through Caddy...
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8080/' -Method HEAD -UseBasicParsing"

echo.
echo 4. Testing API through Caddy...
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8080/api/' -Method HEAD -UseBasicParsing"

echo.
echo Setup verification complete!
echo Access your application at: http://localhost
pause