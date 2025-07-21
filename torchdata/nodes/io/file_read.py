import logging
from typing import Any, Dict, Optional

from smart_open import open
from torchdata.nodes import BaseNode

logger = logging.getLogger(__name__)


class FileReader(BaseNode[Dict]):
    """Universal node that reads file contents from any filesystem supported by smart_open.

    Uses smart_open to support local files, S3, GCS, Azure, HTTP, and more filesystems.
    Works seamlessly with any filesystem that smart_open supports.

    Features:
    - Supports any filesystem that smart_open supports (local, S3, GCS, Azure, HTTP, etc.)
    - Handles compressed files (.gz, .bz2) transparently
    - Maintains state for checkpointing and resumption
    - Preserves metadata from source nodes
    - Works with both text and binary files

    Output format:
        {
            "data": "file contents",      # File contents as str (text mode) or bytes (binary mode)
            "metadata": {
                "file_path": "path/to/file.txt"  # Original file path/URI
                ...                       # Additional metadata from input
            }
        }

    Input: Dictionary containing file path/URI in dict["data"]
    Output: Dict containing file contents and metadata

    Examples:
        >>> # Read from local files
        >>> node = FileReader(source_node)
        >>>
        >>> # Read from S3 with custom client
        >>> node = FileReader(
        ...     source_node,
        ...     transport_params={'client': boto3.client('s3')}
        ... )
        >>>
        >>> # Read binary files
        >>> node = FileReader(source_node, mode="rb")
        >>>
        >>> # Read compressed files
        >>> node = FileReader(
        ...     source_node,
        ...     transport_params={'compression': '.gz'}
        ... )
    """

    SOURCE_KEY = "source"
    DATA_KEY = "data"
    METADATA_KEY = "metadata"

    def __init__(
        self,
        source_node: BaseNode[Dict],
        mode: str = "r",
        encoding: Optional[str] = "utf-8",
        transport_params: Optional[Dict] = None,
    ):
        """Initialize the FileReader.

        Args:
            source_node: Source node that yields dicts with file paths
            mode: File open mode ('r' for text, 'rb' for binary)
            encoding: Text encoding (None for binary mode)
            transport_params: Parameters for smart_open transport layer
                For S3:
                    {'client': boto3.client('s3')}  # Use specific client
                For compression:
                    {'compression': '.gz'}  # Force gzip compression
                    {'compression': '.bz2'}  # Force bz2 compression
                    {'compression': 'disable'}  # Disable compression
        """
        super().__init__()
        self.source = source_node
        self.mode = mode
        self.encoding = encoding
        self.transport_params = transport_params or {}

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset must fully initialize the node's state.

        First resets source to beginning, then restores its state if provided.

        Args:
            initial_state: Optional state dictionary for resumption
        """
        super().reset(initial_state)
        if initial_state is not None:
            self.source.reset(initial_state.get(self.SOURCE_KEY))
        else:
            self.source.reset(None)

    def next(self) -> Dict:
        """Get next file path from source and return its contents."""
        file_path_item = next(self.source)
        file_path = file_path_item[self.DATA_KEY]

        # Preserve metadata from source
        source_metadata = file_path_item.get(self.METADATA_KEY, {})

        try:
            # Use smart_open to handle any file system
            with open(
                file_path,
                self.mode,
                encoding=None if "b" in self.mode else self.encoding,
                transport_params=self.transport_params,
            ) as f:
                content = f.read()

            # Create output with metadata
            metadata = {"file_path": file_path}

            # Include metadata from source
            if source_metadata:
                metadata.update(source_metadata)

            return {self.DATA_KEY: content, self.METADATA_KEY: metadata}

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the node."""
        return {self.SOURCE_KEY: self.source.state_dict()}
