import shutil
from pathlib import Path
from .base import VaultStorage


class LocalStorage(VaultStorage):
    def __init__(self, root_path: str):
        """Initialize the LocalStorage with a root directory path."""
        self.root_path = Path(root_path)

    def list_files(self, file_extension: str = '.md') -> list[str]:
        """Returns a list of all file paths/keys with the given extension in the local directory."""
        raise NotImplementedError

    def read_file(self, file_path: str) -> str:
        """Reads the content of a single file from the local filesystem into memory."""
        raise NotImplementedError

    def save_vector_store(self, local_tarball_path: str):
        """Saves the vector store tarball locally (could be a file copy)."""
        dest_path = self.root_path / Path(local_tarball_path).name
        try:
            shutil.copy2(local_tarball_path, dest_path)
        except Exception as e:
            raise IOError(f"Failed to save vector store tarball: {e}")

    def load_vector_store(self, local_tarball_path: str):
        """Loads the vector store tarball from the local filesystem into memory."""
        src_path = self.root_path / Path(local_tarball_path).name
        if not src_path.exists():
            raise FileNotFoundError(f"Vector store tarball not found at {src_path}")
        try:
            shutil.copy2(src_path, local_tarball_path)
        except Exception as e:
            raise IOError(f"Failed to load vector store tarball: {e}")

    def get_metadata(self, file_path: str) -> dict:
        """Gets metadata like last modified time for a file in the local filesystem."""
        raise NotImplementedError 