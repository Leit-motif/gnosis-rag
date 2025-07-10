import dropbox

# Your app credentials
APP_KEY = "dtcbygbeqferfyr"
APP_SECRET = "6x5yvizfgg2vbxc"

# Step 1: Get authorization URL
auth = dropbox.oauth.DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET, token_access_type='offline')
authorize_url = auth.start()
print(f"1. Go to: {authorize_url}")
print("2. Click 'Allow' (you might have to log in first)")
print("3. Copy the authorization code")

# Step 2: Exchange code for refresh token
auth_code = input("Enter the authorization code: ").strip()
try:
    oauth_result = auth.finish(auth_code)
    print(f"Access token: {oauth_result.access_token}")
    print(f"Refresh token: {oauth_result.refresh_token}")
    print(f"Expires at: {oauth_result.expires_at}")
except Exception as e:
    print(f"Error: {e}")
