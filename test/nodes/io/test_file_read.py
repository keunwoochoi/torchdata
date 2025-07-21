"""Tests for universal FileReader."""

import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, mock_open, patch

import boto3
import pytest
from moto import mock_aws
from torchdata.nodes import BaseNode

from torchdata.nodes.io.file_read import FileReader


class MockSourceNode(BaseNode[Dict]):
    """Mock source node that provides file paths for testing."""

    def __init__(self, file_paths: List[str], metadata: Dict[str, Any] = None):
        super().__init__()
        self.file_paths = file_paths
        self.metadata = metadata or {}
        self._current_idx = 0

    def reset(self, initial_state=None):
        super().reset(initial_state)
        if initial_state is not None:
            self._current_idx = initial_state.get("current_idx", 0)
        else:
            self._current_idx = 0

    def next(self) -> Dict:
        if self._current_idx >= len(self.file_paths):
            raise StopIteration("No more files")

        path = self.file_paths[self._current_idx]
        self._current_idx += 1

        return {FileReader.DATA_KEY: path, FileReader.METADATA_KEY: dict(self.metadata)}

    def get_state(self):
        return {"current_idx": self._current_idx}


@pytest.fixture(scope="function")
def temp_dir(tmp_path_factory):
    """Create a temporary directory with test files."""
    tmp_dir_name = f"file_reader_test_{uuid.uuid4().hex[:8]}"
    tmp_path = tmp_path_factory.mktemp(tmp_dir_name)

    # Create test files in a deterministic order
    files = [
        ("file1.txt", "content1"),
        ("file2.txt", "content2"),
        ("file3.txt", "content3"),
        ("data.json", '{"key": "value"}'),
        ("image.png", b"fake png data"),
    ]

    for name, content in files:
        if isinstance(content, str):
            (tmp_path / name).write_text(content)
        else:
            (tmp_path / name).write_bytes(content)

    # Create a subdirectory with files
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file4.txt").write_text("content4")

    return tmp_path


@pytest.fixture(scope="function")
def mock_bucket():
    """Create a mock S3 bucket with test files."""
    with mock_aws():
        # Create S3 client and bucket with region
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket_name = "test-bucket"

        # Create bucket (no location constraint for us-east-1)
        s3.create_bucket(Bucket=bucket_name)

        # Create test files in a deterministic order
        files = [
            ("file1.txt", "content1"),
            ("file2.txt", "content2"),
            ("file3.txt", "content3"),
            ("data.json", '{"key": "value"}'),
            ("image.png", b"fake png data"),
            ("subdir/file4.txt", "content4"),
        ]

        for key, content in files:
            if isinstance(content, str):
                s3.put_object(Bucket=bucket_name, Key=key, Body=content.encode())
            else:
                s3.put_object(Bucket=bucket_name, Key=key, Body=content)

        yield bucket_name


def test_file_reader_local_text(temp_dir):
    """Test FileReader with local text files."""
    file_paths = [str(temp_dir / "file1.txt"), str(temp_dir / "file2.txt")]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"
    assert content1[FileReader.METADATA_KEY]["file_path"].endswith("file1.txt")

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"
    assert content2[FileReader.METADATA_KEY]["file_path"].endswith("file2.txt")


def test_file_reader_local_binary(temp_dir):
    """Test FileReader with local binary files."""
    file_paths = [str(temp_dir / "image.png")]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"fake png data"
    assert content[FileReader.METADATA_KEY]["file_path"].endswith("image.png")


def test_file_reader_state_management(temp_dir):
    """Test state management in FileReader with local files."""
    file_paths = [str(temp_dir / "file1.txt"), str(temp_dir / "file2.txt"), str(temp_dir / "file3.txt")]
    source_node = MockSourceNode(file_paths)
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"

    # Save state after first file
    state = reader_node.get_state()

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"

    # Create new reader and restore state
    new_source = MockSourceNode(file_paths)
    new_reader = FileReader(new_source)
    new_reader.reset(state)

    # Should continue from second file
    new_content2 = next(new_reader)
    assert new_content2[FileReader.DATA_KEY] == "content2"


@pytest.mark.aws
def test_file_reader_s3_text(mock_bucket):
    """Test FileReader with S3 text files."""
    file_paths = [f"s3://{mock_bucket}/file1.txt", f"s3://{mock_bucket}/file2.txt"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"
    assert content1[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/file1.txt"
    assert content1[FileReader.METADATA_KEY]["source"] == "s3"

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"
    assert content2[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/file2.txt"


@pytest.mark.aws
def test_file_reader_s3_binary(mock_bucket):
    """Test FileReader with S3 binary files."""
    file_paths = [f"s3://{mock_bucket}/image.png"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"fake png data"
    assert content[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/image.png"


@pytest.mark.aws
def test_file_reader_s3_state_management(mock_bucket):
    """Test state management in FileReader with S3 files."""
    file_paths = [f"s3://{mock_bucket}/file1.txt", f"s3://{mock_bucket}/file2.txt"]
    source_node = MockSourceNode(file_paths, {"source": "s3"})
    reader_node = FileReader(source_node)

    # Read first file
    content1 = next(reader_node)
    assert content1[FileReader.DATA_KEY] == "content1"

    # Save state after first file
    state = reader_node.get_state()

    # Read second file
    content2 = next(reader_node)
    assert content2[FileReader.DATA_KEY] == "content2"

    # Create new reader and restore state
    new_source = MockSourceNode(file_paths, {"source": "s3"})
    new_reader = FileReader(new_source)
    new_reader.reset(state)

    # Should continue from second file
    new_content2 = next(new_reader)
    assert new_content2[FileReader.DATA_KEY] == "content2"
    assert new_content2[FileReader.METADATA_KEY]["file_path"] == f"s3://{mock_bucket}/file2.txt"


@pytest.mark.azure
@patch("smart_open.open")
def test_file_reader_azure_text(mock_smart_open):
    """Test FileReader with Azure Blob Storage text files."""
    # Mock smart_open for Azure Blob Storage
    mock_smart_open.return_value.__enter__.return_value.read.return_value = "azure_content1"

    file_paths = ["abfs://container@account.dfs.core.windows.net/file1.txt"]
    source_node = MockSourceNode(file_paths, {"source": "abfs"})
    reader_node = FileReader(source_node, transport_params={"anon": False})

    # Read file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == "azure_content1"
    assert content[FileReader.METADATA_KEY]["file_path"] == "abfs://container@account.dfs.core.windows.net/file1.txt"
    assert content[FileReader.METADATA_KEY]["source"] == "abfs"

    # Verify smart_open was called with correct parameters
    mock_smart_open.assert_called_once_with(
        "abfs://container@account.dfs.core.windows.net/file1.txt",
        "r",
        encoding="utf-8",
        transport_params={"anon": False},
    )


@pytest.mark.azure
@patch("smart_open.open")
def test_file_reader_azure_binary(mock_smart_open):
    """Test FileReader with Azure Blob Storage binary files."""
    # Mock smart_open for Azure binary data
    mock_smart_open.return_value.__enter__.return_value.read.return_value = b"azure_binary_data"

    file_paths = ["abfs://container@account.dfs.core.windows.net/image.png"]
    source_node = MockSourceNode(file_paths, {"source": "abfs"})
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"azure_binary_data"
    assert content[FileReader.METADATA_KEY]["file_path"] == "abfs://container@account.dfs.core.windows.net/image.png"

    # Verify smart_open was called with correct parameters for binary mode
    mock_smart_open.assert_called_once_with(
        "abfs://container@account.dfs.core.windows.net/image.png", "rb", encoding=None, transport_params={}
    )


@pytest.mark.gcs
@patch("smart_open.open")
def test_file_reader_gcs_text(mock_smart_open):
    """Test FileReader with Google Cloud Storage text files."""
    # Mock smart_open for GCS
    mock_smart_open.return_value.__enter__.return_value.read.return_value = "gcs_content1"

    file_paths = ["gs://my-bucket/file1.txt"]
    source_node = MockSourceNode(file_paths, {"source": "gs"})
    reader_node = FileReader(source_node, transport_params={"client": "mock_gcs_client"})

    # Read file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == "gcs_content1"
    assert content[FileReader.METADATA_KEY]["file_path"] == "gs://my-bucket/file1.txt"
    assert content[FileReader.METADATA_KEY]["source"] == "gs"

    # Verify smart_open was called with correct parameters
    mock_smart_open.assert_called_once_with(
        "gs://my-bucket/file1.txt", "r", encoding="utf-8", transport_params={"client": "mock_gcs_client"}
    )


@pytest.mark.gcs
@patch("smart_open.open")
def test_file_reader_gcs_binary(mock_smart_open):
    """Test FileReader with Google Cloud Storage binary files."""
    # Mock smart_open for GCS binary data
    mock_smart_open.return_value.__enter__.return_value.read.return_value = b"gcs_binary_data"

    file_paths = ["gs://my-bucket/data.parquet"]
    source_node = MockSourceNode(file_paths, {"source": "gs"})
    reader_node = FileReader(source_node, mode="rb")

    # Read binary file
    content = next(reader_node)
    assert content[FileReader.DATA_KEY] == b"gcs_binary_data"
    assert content[FileReader.METADATA_KEY]["file_path"] == "gs://my-bucket/data.parquet"

    # Verify smart_open was called with correct parameters for binary mode
    mock_smart_open.assert_called_once_with("gs://my-bucket/data.parquet", "rb", encoding=None, transport_params={})


def test_file_reader_compression_handling():
    """Test FileReader with compressed files."""
    with patch("smart_open.open") as mock_smart_open:
        # Mock smart_open for compressed file
        mock_smart_open.return_value.__enter__.return_value.read.return_value = "decompressed_content"

        file_paths = ["s3://bucket/compressed.txt.gz"]
        source_node = MockSourceNode(file_paths, {"source": "s3"})
        reader_node = FileReader(source_node, transport_params={"compression": ".gz"})

        # Read compressed file
        content = next(reader_node)
        assert content[FileReader.DATA_KEY] == "decompressed_content"
        assert content[FileReader.METADATA_KEY]["file_path"] == "s3://bucket/compressed.txt.gz"

        # Verify smart_open was called with compression parameters
        mock_smart_open.assert_called_once_with(
            "s3://bucket/compressed.txt.gz", "r", encoding="utf-8", transport_params={"compression": ".gz"}
        )


def test_file_reader_error_handling():
    """Test FileReader error handling."""
    with patch("smart_open.open") as mock_smart_open:
        # Mock smart_open to raise an exception
        mock_smart_open.side_effect = IOError("File not found")

        file_paths = ["nonexistent://file.txt"]
        source_node = MockSourceNode(file_paths)
        reader_node = FileReader(source_node)

        # Should raise the original exception
        with pytest.raises(IOError, match="File not found"):
            next(reader_node)
