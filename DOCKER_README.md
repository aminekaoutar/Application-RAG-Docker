# Docker Setup

## Prerequisites
1. Install Docker Desktop
2. Get a Groq API key from https://console.groq.com/

## Setup
1. Rename `.env` file and add your Groq API key:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

## Running the Application
```bash
# Method 1: Using the start script (Windows)
./start-docker.bat

# Method 2: Manual docker-compose
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

## Access
- Application (through Caddy): http://localhost:8080
- Direct frontend: http://localhost:3000
- Direct backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Caddy Admin: http://localhost:2019

## Stopping
```bash
# Method 1: Using the stop script (Windows)
./stop-docker.bat

# Method 2: Manual docker-compose
docker-compose down

# Stop and remove volumes
docker-compose down -v
```