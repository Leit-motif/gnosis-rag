"""
Dropbox client for Obsidian vault synchronization.
Handles authentication and provides basic connection verification.
"""

import logging
import os
import time
from functools import wraps
from typing import Optional, List, Dict, Any
import dropbox
from dropbox.exceptions import ApiError, AuthError

from api.config import settings

logger = logging.getLogger(__name__)


class DropboxAuthenticationError(Exception):
    """Raised when Dropbox authentication fails."""
    pass


class DropboxOperationError(Exception):
    """Raised when a Dropbox operation fails."""
    pass


def retry_on_error(max_retries: int = 3, initial_backoff: float = 1.0):
    """
    Decorator to retry operations on transient errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (ApiError, ConnectionError, OSError) as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Operation {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    logger.warning(f"Operation {func.__name__} failed (attempt {retries}/{max_retries}): {str(e)}. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
            
            return None
        return wrapper
    return decorator


class DropboxClient:
    """
    Dropbox client for handling vault synchronization.
    
    This client manages authentication with the Dropbox API and provides
    file operations for syncing Obsidian vault files.
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
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize a path for Dropbox API (must start with /).
        
        Args:
            path: The path to normalize
            
        Returns:
            str: Normalized path
        """
        if not path.startswith('/'):
            path = '/' + path
        return path.replace('\\', '/')
    
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
    
    @retry_on_error(max_retries=3, initial_backoff=1.0)
    def upload_file(self, local_path: str, dropbox_path: str) -> bool:
        """
        Upload a file from local filesystem to Dropbox.
        
        Args:
            local_path: Path to the local file
            dropbox_path: Destination path on Dropbox
            
        Returns:
            bool: True if upload successful, False otherwise
            
        Raises:
            DropboxOperationError: If upload fails after retries
        """
        if not self.dbx:
            logger.error("Dropbox client not authenticated")
            return False
        
        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist: {local_path}")
            return False
        
        dropbox_path = self._normalize_path(dropbox_path)
        
        try:
            with open(local_path, 'rb') as f:
                file_content = f.read()
                
            # Use upload session for files larger than 150MB, regular upload for smaller files
            file_size = len(file_content)
            if file_size > 150 * 1024 * 1024:  # 150MB
                logger.info(f"Uploading large file ({file_size} bytes) using upload session: {dropbox_path}")
                self._upload_large_file(file_content, dropbox_path)
            else:
                logger.info(f"Uploading file ({file_size} bytes): {dropbox_path}")
                self.dbx.files_upload(
                    file_content,
                    dropbox_path,
                    mode=dropbox.files.WriteMode.overwrite,
                    autorename=False
                )
            
            logger.info(f"Successfully uploaded file: {local_path} -> {dropbox_path}")
            return True
            
        except ApiError as e:
            error_msg = f"Dropbox API error during upload: {str(e)}"
            logger.error(error_msg)
            raise DropboxOperationError(error_msg)
        except OSError as e:
            error_msg = f"File system error during upload: {str(e)}"
            logger.error(error_msg)
            raise DropboxOperationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during upload: {str(e)}"
            logger.error(error_msg)
            raise DropboxOperationError(error_msg)
    
    def _upload_large_file(self, file_content: bytes, dropbox_path: str) -> None:
        """
        Upload a large file using Dropbox upload session.
        
        Args:
            file_content: The file content as bytes
            dropbox_path: Destination path on Dropbox
        """
        CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
        
        session_start_result = self.dbx.files_upload_session_start(file_content[:CHUNK_SIZE])
        cursor = dropbox.files.UploadSessionCursor(
            session_id=session_start_result.session_id,
            offset=CHUNK_SIZE
        )
        
        # Upload remaining chunks
        for i in range(CHUNK_SIZE, len(file_content), CHUNK_SIZE):
            chunk = file_content[i:i + CHUNK_SIZE]
            if i + CHUNK_SIZE >= len(file_content):
                # Last chunk
                commit = dropbox.files.CommitInfo(path=dropbox_path, mode=dropbox.files.WriteMode.overwrite)
                self.dbx.files_upload_session_finish(chunk, cursor, commit)
            else:
                self.dbx.files_upload_session_append_v2(chunk, cursor)
                cursor.offset += len(chunk)
    
    @retry_on_error(max_retries=3, initial_backoff=1.0)
    def download_file(self, dropbox_path: str, local_path: str) -> bool:
        """
        Download a file from Dropbox to local filesystem.
        
        Args:
            dropbox_path: Path to the file on Dropbox
            local_path: Destination path on local filesystem
            
        Returns:
            bool: True if download successful, False otherwise
            
        Raises:
            DropboxOperationError: If download fails after retries
        """
        if not self.dbx:
            logger.error("Dropbox client not authenticated")
            return False
        
        dropbox_path = self._normalize_path(dropbox_path)
        
        try:
            # Ensure local directory exists
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
                logger.debug(f"Created local directory: {local_dir}")
            
            # Download file
            metadata, response = self.dbx.files_download(dropbox_path)
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded file: {dropbox_path} -> {local_path}")
            return True
            
        except ApiError as e:
            if hasattr(e.error, 'get_path') and hasattr(e.error.get_path(), 'get_not_found'):
                logger.warning(f"File not found on Dropbox: {dropbox_path}")
                return False
            else:
                error_msg = f"Dropbox API error during download: {str(e)}"
                logger.error(error_msg)
                raise DropboxOperationError(error_msg)
        except OSError as e:
            error_msg = f"File system error during download: {str(e)}"
            logger.error(error_msg)
            raise DropboxOperationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during download: {str(e)}"
            logger.error(error_msg)
            raise DropboxOperationError(error_msg)
    
    @retry_on_error(max_retries=3, initial_backoff=1.0)
    def file_exists(self, dropbox_path: str) -> bool:
        """
        Check if a file exists on Dropbox.
        
        Args:
            dropbox_path: Path to check on Dropbox
            
        Returns:
            bool: True if file exists, False otherwise
        """
        if not self.dbx:
            logger.error("Dropbox client not authenticated")
            return False
        
        dropbox_path = self._normalize_path(dropbox_path)
        
        try:
            self.dbx.files_get_metadata(dropbox_path)
            logger.debug(f"File exists on Dropbox: {dropbox_path}")
            return True
        except ApiError as e:
            if hasattr(e.error, 'get_path') and hasattr(e.error.get_path(), 'get_not_found'):
                logger.debug(f"File does not exist on Dropbox: {dropbox_path}")
                return False
            else:
                logger.error(f"Error checking file existence: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error checking file existence: {str(e)}")
            return False
    
    @retry_on_error(max_retries=3, initial_backoff=1.0)
    def list_files(self, dropbox_path: str = "/", recursive: bool = True) -> List[Dict[str, Any]]:
        """
        List files in a Dropbox directory.
        
        Args:
            dropbox_path: Directory path on Dropbox
            recursive: Whether to list files recursively
            
        Returns:
            List[Dict[str, Any]]: List of file information dictionaries
        """
        if not self.dbx:
            logger.error("Dropbox client not authenticated")
            return []
        
        dropbox_path = self._normalize_path(dropbox_path)
        files = []
        
        try:
            result = self.dbx.files_list_folder(dropbox_path, recursive=recursive)
            
            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        files.append({
                            "path": entry.path_display,
                            "name": entry.name,
                            "size": entry.size,
                            "modified": entry.server_modified.isoformat() if entry.server_modified else None,
                            "content_hash": entry.content_hash
                        })
                
                if not result.has_more:
                    break
                    
                result = self.dbx.files_list_folder_continue(result.cursor)
            
            logger.info(f"Listed {len(files)} files from Dropbox directory: {dropbox_path}")
            return files
            
        except ApiError as e:
            if hasattr(e.error, 'get_path') and hasattr(e.error.get_path(), 'get_not_found'):
                logger.warning(f"Directory not found on Dropbox: {dropbox_path}")
                return []
            else:
                logger.error(f"Error listing files: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Unexpected error listing files: {str(e)}")
            return []
    
    def get_file_metadata(self, dropbox_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific file on Dropbox.
        
        Args:
            dropbox_path: Path to the file on Dropbox
            
        Returns:
            Optional[Dict[str, Any]]: File metadata if successful, None otherwise
        """
        if not self.dbx:
            logger.error("Dropbox client not authenticated")
            return None
        
        dropbox_path = self._normalize_path(dropbox_path)
        
        try:
            metadata = self.dbx.files_get_metadata(dropbox_path)
            
            if isinstance(metadata, dropbox.files.FileMetadata):
                return {
                    "path": metadata.path_display,
                    "name": metadata.name,
                    "size": metadata.size,
                    "modified": metadata.server_modified.isoformat() if metadata.server_modified else None,
                    "content_hash": metadata.content_hash
                }
            else:
                logger.warning(f"Path is not a file: {dropbox_path}")
                return None
                
        except ApiError as e:
            if hasattr(e.error, 'get_path') and hasattr(e.error.get_path(), 'get_not_found'):
                logger.debug(f"File not found: {dropbox_path}")
                return None
            else:
                logger.error(f"Error getting file metadata: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error getting file metadata: {str(e)}")
            return None 