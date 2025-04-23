import requests
import json

# Base URL for your API
base_url = "https://rice-depth-shopper-fewer.trycloudflare.com"

# Data from the CustomGPT request
save_data = {
    "session_id": "session_20250420_201935",
    "conversation_name": "Core Beliefs and Intellectual Trajectory Summary"
}

print("Attempting to save conversation directly...")
response = requests.post(f"{base_url}/save_conversation", json=save_data)
print(f"Status code: {response.status_code}")
print("Response:")
print(json.dumps(response.json(), indent=2)) 