import requests
import json

# Base URL for your API
local_url = "http://localhost:8000"
cloudflare_url = "https://rice-depth-shopper-fewer.trycloudflare.com"

def test_endpoint(base_url, endpoint, method="GET", json_data=None):
    full_url = f"{base_url}{endpoint}"
    print(f"\nTesting {method} {full_url}")
    
    try:
        if method == "GET":
            response = requests.get(full_url)
        elif method == "POST":
            if json_data:
                response = requests.post(full_url, json=json_data)
            else:
                response = requests.post(full_url)
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS!")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print("Response is not JSON")
        else:
            print(f"ERROR: {response.text}")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")

# Test OpenAPI spec (this should always work)
test_endpoint(local_url, "/openapi.yaml")
test_endpoint(cloudflare_url, "/openapi.yaml")

# Test plugin manifest
test_endpoint(local_url, "/.well-known/ai-plugin.json")
test_endpoint(cloudflare_url, "/.well-known/ai-plugin.json")

# Test plain save_conversation endpoint with test data
save_data = {
    "session_id": "default",
    "conversation_name": "Test Conversation"
}
test_endpoint(local_url, "/save_conversation", "POST", save_data)
test_endpoint(cloudflare_url, "/save_conversation", "POST", save_data)

# Test with different URL formats to check for routing issues
test_endpoint(cloudflare_url, "/api/save_conversation", "POST", save_data)
test_endpoint(cloudflare_url, "/save-conversation", "POST", save_data)
test_endpoint(cloudflare_url, "/saveConversation", "POST", save_data) 