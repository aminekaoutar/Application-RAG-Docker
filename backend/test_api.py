import requests
import json

# Test the API endpoints
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_query():
    """Test query endpoint."""
    print("Testing query endpoint...")
    payload = {
        "question": "What services does the company offer?",
        "conversation_id": "test-123"
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Conversation ID: {result['conversation_id']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_stats():
    """Test stats endpoint."""
    print("Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("🧪 Testing API endpoints...\n")
    
    try:
        test_health()
        test_query()
        test_stats()
        print("✅ All tests completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")