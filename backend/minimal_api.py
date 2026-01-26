from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Web Scraping API",
    description="Simple API for web scraping and data processing",
    version="1.0.0"
)

class ScrapeRequest(BaseModel):
    url: str
    max_depth: int = 1

class QueryRequest(BaseModel):
    question: str

# Mock data storage
mock_data = {
    "digisand.ma": {
        "title": "DigiSand - Digital Marketing Agency",
        "description": "We are a digital marketing agency specializing in web development, SEO, and online advertising.",
        "services": ["Web Development", "SEO Optimization", "Social Media Marketing", "Content Creation"],
        "contact": {"email": "contact@digisand.ma", "phone": "+212 6 12 34 56 78"}
    }
}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Web Scraping API is running", "status": "healthy"}

@app.post("/scrape")
async def scrape_website(request: ScrapeRequest):
    """Mock scraping endpoint"""
    logger.info(f"Scraping URL: {request.url}")
    
    # Simulate scraping process
    if "digisand" in request.url.lower():
        return {
            "url": request.url,
            "status": "success",
            "data": mock_data.get("digisand.ma"),
            "message": "Successfully scraped DigiSand website"
        }
    else:
        return {
            "url": request.url,
            "status": "partial_success",
            "data": {"title": "Generic Website", "content": "Basic website content"},
            "message": "Scraped basic website information"
        }

@app.post("/query")
async def query_data(request: QueryRequest):
    """Mock RAG query endpoint"""
    logger.info(f"Processing query: {request.question}")
    
    # Simple keyword matching for demo
    question_lower = request.question.lower()
    
    if "digisand" in question_lower or "digital" in question_lower:
        response = "DigiSand is a digital marketing agency that specializes in web development, SEO optimization, social media marketing, and content creation. They help businesses establish strong online presence."
    elif "service" in question_lower or "offer" in question_lower:
        response = "DigiSand offers web development, SEO optimization, social media marketing, and content creation services to help businesses grow their online presence."
    elif "contact" in question_lower:
        response = "You can contact DigiSand at contact@digisand.ma or call +212 6 12 34 56 78."
    else:
        response = "I don't have specific information about that. DigiSand is a digital marketing agency that helps businesses with their online presence through web development and marketing services."
    
    return {
        "question": request.question,
        "answer": response,
        "confidence": 0.85,
        "source": "mock_data"
    }

@app.get("/data")
async def get_scraped_data():
    """Get all scraped data"""
    return {"data": mock_data, "count": len(mock_data)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)