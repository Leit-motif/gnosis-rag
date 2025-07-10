from google.cloud import storage
from .base import VaultStorage
import os


class GCSStorage(VaultStorage):
    def __init__(self, bucket_name: str, prefix: str = ""):
        """Initialize the GCSStorage with a bucket name and optional prefix."""
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def list_files(self, file_extension: str = '.md') -> list[str]:
        """Returns a list of all file paths/keys with the given extension in the GCS bucket."""
        raise NotImplementedError

    def read_file(self, file_path: str) -> str:
        """Reads the content of a single file from GCS into memory."""
        raise NotImplementedError

    def save_vector_store(self, local_tarball_path: str):
        """Uploads the vector store tarball to GCS atomically."""
        blob_name = os.path.join(self.prefix, os.path.basename(local_tarball_path)) if self.prefix else os.path.basename(local_tarball_path)
        temp_blob_name = blob_name + ".tmp"
        temp_blob = self.bucket.blob(temp_blob_name)
        final_blob = self.bucket.blob(blob_name)
        try:
            temp_blob.upload_from_filename(local_tarball_path)
            # Atomic replace: use compose if exists, else rename
            if final_blob.exists():
                final_blob.delete()
            temp_blob.rename(blob_name)
        except Exception as e:
            raise IOError(f"Failed to upload vector store tarball to GCS: {e}")

    def load_vector_store(self, local_tarball_path: str):
        """Downloads the vector store tarball from GCS into memory."""
        blob_name = os.path.join(self.prefix, os.path.basename(local_tarball_path)) if self.prefix else os.path.basename(local_tarball_path)
        blob = self.bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"Vector store tarball not found in GCS at {blob_name}")
        try:
            blob.download_to_filename(local_tarball_path)
        except Exception as e:
            raise IOError(f"Failed to download vector store tarball from GCS: {e}")

    def get_metadata(self, file_path: str) -> dict:
        """Gets metadata like last modified time for a file in GCS."""
        raise NotImplementedError 