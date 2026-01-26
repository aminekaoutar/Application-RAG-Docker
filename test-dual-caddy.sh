#!/bin/bash
echo "Testing Dual Caddy Setup..."

# Check if containers are running
echo "1. Checking container status..."
docker-compose ps

echo ""
echo "2. Testing Private Caddy health endpoint..."
curl -s http://localhost:8080/health

echo ""
echo "3. Testing Public Caddy health endpoint..."
curl -s http://localhost:8081/health

echo ""
echo "4. Testing frontend through Private Caddy..."
curl -I http://localhost:8080/

echo ""
echo "5. Testing API through Private Caddy..."
curl -I http://localhost:8080/api/

echo ""
echo "Setup verification complete!"
echo "Private Caddy: http://localhost:8080"
echo "Public Caddy: http://localhost:8081 (internal only)"