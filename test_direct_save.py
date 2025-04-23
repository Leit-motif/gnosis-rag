import requests
import json
import sys

# Base URL for your API
base_url = "https://rice-depth-shopper-fewer.trycloudflare.com"  # Change this if your URL changes
#base_url = "http://localhost:8000"  # Uncomment for local testing

def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

# Step 1: Check all available conversations
print("\nStep 1: Checking all conversations in memory...")
response = requests.get(f"{base_url}/debug_all_conversations")

if response.status_code == 200:
    print("Success! Available conversations:")
    print_json(response.json())
else:
    print(f"Failed to get conversations: {response.status_code}")
    print(response.text)
    sys.exit(1)

# Step 2: Save exact content
print("\nStep 2: Saving exact content...")

# Use the content you want to save
content_to_save = """
Cloudflare Named Tunnels (Free & Stable): Cloudflare offers permanent, named tunnels for free. Setting one up takes a few more steps initially (logging into Cloudflare, creating the tunnel via their dashboard or CLI), but the result is a stable URL (either your-name.trycloudflare.com or potentially a subdomain of your own domain if you have one) that does not change when you restart cloudflared. This would solve the problem permanently and allow you to use "Import from URL".
Ngrok Paid Tiers (Stable): ngrok offers stable URLs (reserved domains/subdomains) on its paid plans. This is also a solution to the dynamic URL problem, but it costs money.
In Summary:
Cloudflare Quick Tunnel vs. Ngrok Free Tier: Both have dynamic URLs, making them equally inconvenient for this specific Custom GPT integration task (requiring manual updates).
Cloudflare Named Tunnel vs. Ngrok Paid Tier: Both offer stable URLs. Cloudflare's named tunnel is the free way to get a stable URL. Ngrok's paid tier is often considered very straightforward to set up but has a cost.
"""

save_data = {
    "conversation_name": "Cloudflare vs Ngrok for Stable Tunnels",
    "content": content_to_save.strip()
}

response = requests.post(f"{base_url}/save_exact_content", json=save_data)

if response.status_code == 200:
    print("Success! Content saved directly to your vault.")
    print_json(response.json())
else:
    print(f"Failed to save content: {response.status_code}")
    print(response.text)
    sys.exit(1)

print("\nDone! Check your Obsidian vault for the saved content.") 