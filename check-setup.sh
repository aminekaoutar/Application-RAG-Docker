#!/bin/bash
echo "Checking Caddy Reverse Proxy Setup..."

# Check if containers are running
echo "1. Checking container status..."
docker-compose ps

echo ""
echo "2. Testing Caddy health endpoint..."
curl -s http://localhost/health

echo ""
echo "3. Testing frontend through Caddy..."
curl -I http://localhost/

echo ""
echo "4. Testing API through Caddy..."
curl -I http://localhost/api/

echo ""
echo "Setup verification complete!"
echo "Access your application at: http://localhost"