import requests
import json
import uuid
import sys

# Base URL for your API
base_url = "https://rice-depth-shopper-fewer.trycloudflare.com"  # Change this if your URL changes
#base_url = "http://localhost:8000"  # Uncomment for local testing

# Generate a simple session ID
test_session_id = str(uuid.uuid4())
print(f"Using test session ID: {test_session_id}")

# Define the conversation we want to save directly (the one about Cloudflare vs Ngrok)
# This simulates what the Custom GPT would send
conversation_messages = [
    {
        "role": "user",
        "content": "What are the differences between Cloudflare tunnels and Ngrok for stable connections?"
    },
    {
        "role": "assistant", 
        "content": """Cloudflare Named Tunnels (Free & Stable): Cloudflare offers permanent, named tunnels for free. Setting one up takes a few more steps initially (logging into Cloudflare, creating the tunnel via their dashboard or CLI), but the result is a stable URL (either your-name.trycloudflare.com or potentially a subdomain of your own domain if you have one) that does not change when you restart cloudflared. This would solve the problem permanently and allow you to use "Import from URL".
Ngrok Paid Tiers (Stable): ngrok offers stable URLs (reserved domains/subdomains) on its paid plans. This is also a solution to the dynamic URL problem, but it costs money.
In Summary:
Cloudflare Quick Tunnel vs. Ngrok Free Tier: Both have dynamic URLs, making them equally inconvenient for this specific Custom GPT integration task (requiring manual updates).
Cloudflare Named Tunnel vs. Ngrok Paid Tier: Both offer stable URLs. Cloudflare's named tunnel is the free way to get a stable URL. Ngrok's paid tier is often considered very straightforward to set up but has a cost."""
    }
]

# Prepare the save request with the messages directly included
save_data = {
    "session_id": test_session_id,  # This can be any value but we still need to provide it
    "conversation_name": "Cloudflare vs Ngrok for Stable Tunnels",
    "messages": conversation_messages
}

print("\nSaving conversation with direct messages...")
response = requests.post(f"{base_url}/save_conversation", json=save_data)

if response.status_code == 200:
    print("Success! Conversation saved directly with messages.")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Failed to save conversation: {response.status_code}")
    print(response.text)
    sys.exit(1)

print("\nDone! Check your Obsidian vault for the saved conversation.") 