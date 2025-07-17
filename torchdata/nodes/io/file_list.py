import logging
import os
from typing import Any, Dict, List, Optional

import fsspec
from torchdata.nodes import BaseNode

logger = logging.getLogger(__name__)


class FileLister(BaseNode[Dict]):
    """Node that lists files from any fsspec-supported filesystem matching specified patterns.

    Uses fsspec to provide universal file access capabilities for local and remote filesystems.
    Supports any filesystem that fsspec supports (local, S3, GCS, Azure, HTTP, etc.).

    Features:
    - Lists files from any fsspec-supported filesystem
    - Supports glob patterns for file matching
    - Maintains state for checkpointing and resumption

    Input: None (configured with uri_base and patterns at initialization)
    Output: Dictionary with file URI and metadata
        {
            "data": "s3://bucket/key.txt",  # File URI as string
            "metadata": {
                "fs_protocol": "s3",        # Filesystem protocol
                "source": "s3"             # Source type
            }
        }

    Examples:
        >>> # List local files
        >>> node = FileLister("path/to/local/dir", ["*.txt", "*.json"])
        >>> for item in node:
        ...     print(item["data"])  # prints "path/to/local/dir/file1.txt", etc.
        >>>
        >>> # List S3 files with anonymous access
        >>> node = FileLister("s3://my-bucket/prefix", ["*.csv"],
        ...                   fs_options={"anon": True})
        >>> for item in node:
        ...     print(item["data"])  # prints "s3://my-bucket/prefix/file1.csv", etc.
        >>>
        >>> # List GCS files
        >>> node = FileLister("gs://my-bucket/prefix", ["*.parquet"])
        >>> for item in node:
        ...     print(item["data"])  # prints "gs://my-bucket/prefix/file1.parquet", etc.
    """

    SOURCE_KEY = "source"
    CURRENT_IDX_KEY = "current_idx"

    def __init__(self, uri_base: str, patterns: List[str], fs_options: Optional[Dict[str, Any]] = None):
        """Initialize the FileLister.

        Args:
            uri_base: Base URI for file listing (any fsspec-supported URI)
            patterns: List of glob patterns to match files against
            fs_options: Options to pass to fsspec filesystem
        """
        super().__init__()

        self.uri_base = uri_base
        self.patterns = patterns if patterns else ["*"]
        self.fs_options = dict(fs_options or {})

        # Parse protocol from URI using fsspec
        self.protocol, _ = fsspec.core.split_protocol(uri_base)
        if self.protocol is None:
            self.protocol = "file"  # Default to local filesystem

        # Create filesystem instance
        self.fs = fsspec.filesystem(self.protocol, **self.fs_options)

        # Initialize state
        self._file_paths = None
        self._current_idx = 0

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        """Reset node state and populate file paths."""
        super().reset(initial_state)

        # Always initialize files list
        self._file_paths = set()

        try:
            # Use fsspec's glob for file discovery
            for pattern in self.patterns:
                pattern_path = os.path.join(self.uri_base, pattern)

                try:
                    matches = self.fs.glob(pattern_path)

                    # Filter out directories from matches if pattern is **/* or *
                    if pattern == "**/*" or pattern == "*":
                        matches = [match for match in matches if self.fs.isfile(match)]

                    self._file_paths.update(matches)

                except Exception as e:
                    logger.warning(f"Error globbing pattern {pattern}: {e}")

            # Convert to sorted list for deterministic ordering
            self._file_paths = sorted(self._file_paths)

            # For non-file protocols, ensure URIs are properly formatted
            if self.protocol != "file":
                self._file_paths = [f"{self.protocol}://{path}" for path in self._file_paths]

        except Exception as e:
            logger.error(f"Error listing files from {self.uri_base}: {e}")
            self._file_paths = []

        self._current_idx = 0

        # If there's an initial state, restore position
        if initial_state:
            self._current_idx = initial_state.get(self.CURRENT_IDX_KEY, 0)

    def next(self) -> Dict:
        """Get the next file path."""
        if not self._file_paths:
            raise StopIteration("No matching files found")

        if self._current_idx >= len(self._file_paths):
            raise StopIteration("No more files")

        path = self._file_paths[self._current_idx]
        self._current_idx += 1

        # Create metadata based on filesystem protocol
        metadata = {"fs_protocol": self.protocol, "source": self.protocol}

        return {"data": path, "metadata": metadata}

    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {self.CURRENT_IDX_KEY: self._current_idx}
