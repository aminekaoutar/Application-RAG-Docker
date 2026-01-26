@echo off
echo Testing Dual Caddy Setup...

echo 1. Checking container status...
docker-compose ps

echo.
echo 2. Testing Private Caddy health endpoint...
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8080/health' -UseBasicParsing"

echo.
echo 3. Testing Public Caddy health endpoint...
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8081/health' -UseBasicParsing"

echo.
echo 4. Testing frontend through Private Caddy...
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8080/' -Method HEAD -UseBasicParsing"

echo.
echo 5. Testing API through Private Caddy...
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8080/api/' -Method HEAD -UseBasicParsing"

echo.
echo Setup verification complete!
echo Private Caddy: http://localhost:8080
echo Public Caddy: http://localhost:8081 (internal only)
pause