"""
Dropbox client for Obsidian vault synchronization.
Handles authentication and provides basic connection verification.
"""

import logging
from typing import Optional
import dropbox
from dropbox.exceptions import ApiError, AuthError

from api.config import settings

logger = logging.getLogger(__name__)


class DropboxAuthenticationError(Exception):
    """Raised when Dropbox authentication fails."""
    pass


class DropboxClient:
    """
    Dropbox client for handling vault synchronization.
    
    This client manages authentication with the Dropbox API and provides
    basic connection verification functionality.
    """
    
    def __init__(self):
        """
        Initialize the Dropbox client with credentials from configuration.
        
        Raises:
            DropboxAuthenticationError: If authentication fails or credentials are missing.
        """
        self.app_key = settings.dropbox_app_key
        self.app_secret = settings.dropbox_app_secret
        self.refresh_token = settings.dropbox_refresh_token
        
        # Validate required credentials
        if not all([self.app_key, self.app_secret, self.refresh_token]):
            missing = []
            if not self.app_key:
                missing.append("DROPBOX_APP_KEY")
            if not self.app_secret:
                missing.append("DROPBOX_APP_SECRET")
            if not self.refresh_token:
                missing.append("DROPBOX_REFRESH_TOKEN")
            
            raise DropboxAuthenticationError(
                f"Missing required Dropbox credentials: {', '.join(missing)}. "
                "Please set the required environment variables."
            )
        
        self.dbx: Optional[dropbox.Dropbox] = None
        self._authenticate()
    
    def _authenticate(self) -> None:
        """
        Establish connection with Dropbox API using OAuth2 refresh token.
        
        Raises:
            DropboxAuthenticationError: If authentication fails.
        """
        try:
            self.dbx = dropbox.Dropbox(
                app_key=self.app_key,
                app_secret=self.app_secret,
                oauth2_refresh_token=self.refresh_token
            )
            logger.info("Successfully authenticated with Dropbox API")
            
        except AuthError as e:
            error_msg = f"Dropbox authentication failed: {str(e)}"
            logger.error(error_msg)
            raise DropboxAuthenticationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during Dropbox authentication: {str(e)}"
            logger.error(error_msg)
            raise DropboxAuthenticationError(error_msg)
    
    def verify_connection(self) -> bool:
        """
        Verify that the Dropbox connection is working by making a test API call.
        
        Returns:
            bool: True if connection is working, False otherwise.
        """
        if not self.dbx:
            logger.error("Dropbox client not authenticated")
            return False
        
        try:
            # Make a simple API call to verify connection
            account_info = self.dbx.users_get_current_account()
            logger.info(f"Dropbox connection verified for account: {account_info.email}")
            return True
            
        except AuthError as e:
            logger.error(f"Dropbox authentication error during verification: {str(e)}")
            return False
        except ApiError as e:
            logger.error(f"Dropbox API error during verification: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Dropbox connection verification: {str(e)}")
            return False
    
    def get_account_info(self) -> Optional[dict]:
        """
        Get basic account information for the authenticated user.
        
        Returns:
            Optional[dict]: Account information if successful, None otherwise.
        """
        if not self.dbx:
            logger.error("Dropbox client not authenticated")
            return None
        
        try:
            account_info = self.dbx.users_get_current_account()
            return {
                "account_id": account_info.account_id,
                "email": account_info.email,
                "display_name": account_info.name.display_name,
                "country": account_info.country
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            return None
    
    def is_authenticated(self) -> bool:
        """
        Check if the client is properly authenticated.
        
        Returns:
            bool: True if authenticated, False otherwise.
        """
        return self.dbx is not None 