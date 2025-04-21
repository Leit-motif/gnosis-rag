import requests
import json

# Base URL for your API
base_url = "http://localhost:8000"

# Test the query endpoint
session_id = "test_session_123"
query = "This is a test query"

print("Testing query endpoint...")
try:
    # Use GET with query parameters instead of POST with JSON body
    response = requests.get(
        f"{base_url}/query",
        params={
            "q": query,
            "session_id": session_id
        }
    )
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        print("Query successful!")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Query failed!")
        print(response.text)
except Exception as e:
    print(f"Error: {str(e)}") 