class VaultStorage:
    def list_files(self, file_extension: str = '.md') -> list[str]:
        """Returns a list of all file paths/keys with the given extension."""
        raise NotImplementedError

    def read_file(self, file_path: str) -> str:
        """Reads the content of a single file into memory."""
        raise NotImplementedError

    def save_vector_store(self, local_tarball_path: str):
        """Uploads the vector store tarball."""
        raise NotImplementedError

    def load_vector_store(self, local_tarball_path: str):
        """Downloads the vector store tarball."""
        raise NotImplementedError

    def get_metadata(self, file_path: str) -> dict:
        """Gets metadata like last modified time."""
        raise NotImplementedError 