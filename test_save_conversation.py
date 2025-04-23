import requests
import json
import uuid
import os
import sys
import time

# Base URL for your API
base_url = "https://rice-depth-shopper-fewer.trycloudflare.com"  # Change this if your URL changes
#base_url = "http://localhost:8000"  # Uncomment for local testing

# Generate a unique session ID for testing
test_session_id = str(uuid.uuid4())
print(f"Test session ID: {test_session_id}")

# First, create a test conversation by making a query
print("\nStep 1: Creating test conversation via query...")
query = "This is a test query to create a conversation"

# Use GET with query parameters
response = requests.get(
    f"{base_url}/query",
    params={
        "q": query,
        "session_id": "session123"  # Use a simple, predictable session ID
    }
)

if response.status_code == 200:
    print("Query successful!")
    print(json.dumps(response.json(), indent=2))
    
    # Use the session_id from the response
    session_id = response.json().get("session_id", "session123")
    print(f"Using session ID: {session_id}")
else:
    print(f"Failed to create test conversation: {response.status_code}")
    print(response.text)
    exit(1)

# Wait a moment to ensure the session is properly created
time.sleep(2)

# Now try the debug endpoint to see available sessions
print("\nStep 2: Getting available session info...")
save_data = {
    "session_id": "debug",
    "conversation_name": "Test Conversation Debug"
}

response = requests.post(f"{base_url}/debug_save_conversation", json=save_data)
print(f"Debug status code: {response.status_code}")
if response.status_code == 200:
    print("Debug info:")
    print(json.dumps(response.json(), indent=2))
else:
    print("Debug endpoint not found or failed")

# Finally, try to save the conversation using the session ID
print("\nStep 3: Attempting to save conversation...")
save_data = {
    "session_id": session_id,
    "conversation_name": "Test Conversation Save"
}

response = requests.post(f"{base_url}/save_conversation", json=save_data)
print(f"Status code: {response.status_code}")
print("Response:")
print(json.dumps(response.json(), indent=2))

if response.status_code == 200:
    print("\nSuccess! Check your Obsidian vault for the saved conversation.")
    if "file_path" in response.json():
        print(f"Saved to: {response.json()['file_path']}")
else:
    print("\nFailed to save conversation. See error details above.") 