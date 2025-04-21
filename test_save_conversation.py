import requests
import json
import uuid
import os
import sys

# Generate a unique session ID for testing
test_session_id = str(uuid.uuid4())
print(f"Test session ID: {test_session_id}")

# Print environment variables to check vault path
print("\nEnvironment Variables:")
print(f"OBSIDIAN_VAULT_PATH: {os.environ.get('OBSIDIAN_VAULT_PATH', 'Not set')}")

# Base URL for your API
base_url = "http://localhost:8000"  # Change this if your API runs on a different port

# First, check server status
print("\nChecking server status...")
try:
    response = requests.get(f"{base_url}/")
    print(f"Server status code: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to server. Make sure it's running on http://localhost:8000")
    sys.exit(1)

# First, let's create a test conversation by making a query
query = "This is a test query to create a conversation"

print("\nCreating test conversation...")
# Use GET with query parameters instead of POST with JSON body
response = requests.get(
    f"{base_url}/query",
    params={
        "q": query,
        "session_id": test_session_id
    }
)
if response.status_code == 200:
    print("Test query successful")
else:
    print(f"Failed to create test conversation: {response.status_code}")
    print(response.text)
    exit(1)

# Now try the debug endpoint to diagnose issues
save_data = {
    "session_id": test_session_id,
    "conversation_name": "Test Conversation Save Debug"
}

print("\nRunning diagnostics...")
response = requests.post(f"{base_url}/debug_save_conversation", json=save_data)
print(f"Debug status code: {response.status_code}")
print("Debug info:")
print(json.dumps(response.json(), indent=2))

# Finally, try to save the conversation
save_data = {
    "session_id": test_session_id,
    "conversation_name": "Test Conversation Save"
}

print("\nAttempting to save conversation...")
response = requests.post(f"{base_url}/save_conversation", json=save_data)
print(f"Status code: {response.status_code}")
print("Response:")
print(json.dumps(response.json(), indent=2))

if response.status_code == 200:
    print("\nSuccess! Check your Obsidian vault for the saved conversation.")
    if "file_path" in response.json():
        print(f"Saved to: {response.json()['file_path']}")
        
        # Get absolute path for verification
        vault_path = os.environ.get('OBSIDIAN_VAULT_PATH', 'Unknown')
        if vault_path != 'Unknown':
            abs_file_path = os.path.join(vault_path, response.json()['file_path'])
            print(f"Absolute path: {abs_file_path}")
            
            # Check if file actually exists
            if os.path.exists(abs_file_path):
                print(f"File exists at: {abs_file_path}")
                print(f"File size: {os.path.getsize(abs_file_path)} bytes")
            else:
                print(f"ERROR: File does not exist at: {abs_file_path}")
else:
    print("\nFailed to save conversation. See error details above.") 