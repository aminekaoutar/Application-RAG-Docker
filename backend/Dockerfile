FROM python:3.11-slim

WORKDIR /app

COPY api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]