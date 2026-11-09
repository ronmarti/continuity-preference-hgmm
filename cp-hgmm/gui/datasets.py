"""Dataset loading and durable storage for the CP-HGMM GUI."""

import fcntl
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sktime.datasets import load_from_tsfile_to_dataframe

from models.utils.dataset_torch_preparator import DatasetContainer


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_DATASET_ID_RE = re.compile(r"^[0-9a-f]{32}$")

DATASET_FORMAT_HELP = """\
Upload one **TRAIN** and one **TEST** file in sktime's `.ts` format. Files must
be labeled, dense, equal-length series, with `@timestamps false` and
`@equalLength true`. Univariate and multivariate data are supported. Values
must be finite real numbers; missing values, timestamps, unequal-length
series, and regression/unlabeled files are not supported. Both files must have
the same series length (at least **3**) and number of dimensions. Every TEST
class must also occur in TRAIN. Each file is limited to 20 MiB.

For example, a small univariate dataset with three observations per series:

```text
@problemName Example
@timestamps false
@univariate true
@equalLength true
@seriesLength 3
@classLabel true walking running
@data
0.1,0.2,0.3:walking
0.4,0.5,0.6:running
```

For multivariate data, separate dimensions with `:` and put the class label
after the final `:` (for example `0.1,0.2,0.3:1,2,3:walking`). The display
name is metadata; uploaded files are stored under generated IDs, not under
their supplied filenames.
"""


class DatasetValidationError(ValueError):
    """Raised when an uploaded time-series file pair is not supported."""


class DatasetStorageError(RuntimeError):
    """Raised when uploaded data cannot be saved or read from storage."""


def _storage_dir() -> Path:
    configured = os.environ.get("CPHGMM_DATASETS_DIR")
    if configured:
        return Path(configured).expanduser().absolute()
    runs_dir = Path(os.environ.get("CPHGMM_RUNS_DIR", "runs")).expanduser().absolute()
    return runs_dir.parent / "datasets"


def _safe_name(name: str) -> str:
    if not isinstance(name, str):
        raise DatasetValidationError("Dataset name must be text.")
    name = name.strip()
    if not name or name in {".", ".."} or any(c in name for c in "/\\"):
        raise DatasetValidationError(
            "Choose a non-empty dataset name without path separators."
        )
    if len(name) > 128 or any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise DatasetValidationError("Dataset name must be at most 128 printable characters.")
    return name


def _validate_upload_file(filename: str, content: bytes, split: str) -> None:
    if not isinstance(filename, str) or not filename:
        raise DatasetValidationError(f"Choose a {split} .ts file.")
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not basename.lower().endswith(".ts"):
        raise DatasetValidationError(f"The {split} upload must have a .ts extension.")
    if not isinstance(content, bytes):
        raise DatasetValidationError(f"The {split} file content must be bytes.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DatasetValidationError(
            f"The {split} file is {len(content):,} bytes; the limit is "
            f"{MAX_UPLOAD_BYTES:,} bytes per file. Reduce the file size and try again."
        )


def _check_file_declarations(path: Path, split: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DatasetValidationError(f"The {split} .ts file must be UTF-8 text.") from error
    declarations = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("@data"):
            break
        if stripped.startswith("@"):
            parts = stripped.split(None, 1)
            declarations[parts[0].lower()] = parts[1].strip().lower() if len(parts) > 1 else ""

    if declarations.get("@timestamps") == "true":
        raise DatasetValidationError(
            f"The {split} file uses timestamps; timestamped .ts data is not supported."
        )
    if declarations.get("@timestamps") != "false":
        raise DatasetValidationError(
            f"The {split} file must declare `@timestamps false`."
        )
    if declarations.get("@equallength") == "false":
        raise DatasetValidationError(
            f"The {split} file declares unequal-length series; only equal-length data is supported."
        )
    if declarations.get("@equallength") != "true":
        raise DatasetValidationError(
            f"The {split} file must declare `@equalLength true`."
        )
    if not declarations.get("@classlabel", "").startswith("true "):
        raise DatasetValidationError(
            f"The {split} file must declare labels with `@classLabel true ...`."
        )


def _parse_and_validate(path: Path, split: str):
    _check_file_declarations(path, split)
    try:
        X, labels = load_from_tsfile_to_dataframe(path)
    except Exception as error:
        raise DatasetValidationError(
            f"Could not parse the {split} .ts file: {error}"
        ) from error

    if len(X) == 0:
        raise DatasetValidationError(f"The {split} split must contain at least one series.")
    if X.shape[1] == 0:
        raise DatasetValidationError(f"The {split} split must contain at least one dimension.")
    if len(labels) != len(X):
        raise DatasetValidationError(f"The {split} file has missing or incomplete class labels.")

    clean_labels = []
    for label in labels:
        if label is None or pd.isna(label) or not str(label).strip():
            raise DatasetValidationError(f"The {split} file contains a missing class label.")
        clean_labels.append(str(label))
    if not clean_labels:
        raise DatasetValidationError(f"The {split} split must contain at least one class.")

    lengths = set()
    try:
        for dimension in X.columns:
            for series in X[dimension]:
                values = np.asarray(series)
                lengths.add(len(values))
                numeric_values = np.asarray(values, dtype=np.float64)
                if not np.isfinite(numeric_values).all():
                    raise DatasetValidationError(
                        f"The {split} file contains missing or non-finite values; "
                        "all observations must be finite real numbers."
                    )
    except DatasetValidationError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise DatasetValidationError(
            f"The {split} file must contain dense real-valued series with no missing values."
        ) from error

    if len(lengths) != 1:
        raise DatasetValidationError(
            f"The {split} split contains unequal-length series; all series must have the same length."
        )
    length = next(iter(lengths))
    if length < 3:
        raise DatasetValidationError(
            f"Series must contain at least 3 observations; the {split} length is {length}."
        )
    return {
        "samples": len(X),
        "dimensions": X.shape[1],
        "length": length,
        "labels": clean_labels,
    }


@contextmanager
def _storage_lock(root: Path):
    lock_path = root / ".upload.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_metadata(dataset_dir: Path):
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict) or metadata.get("id") != dataset_dir.name:
        return None
    if not isinstance(metadata.get("name"), str):
        return None
    return metadata


def list_uploaded_datasets() -> list[dict]:
    """Return stored uploads, ordered by display name."""
    root = _storage_dir()
    if not root.exists():
        return []
    if not root.is_dir():
        raise DatasetStorageError(f"Dataset storage path is not a directory: {root}")
    entries = []
    try:
        for dataset_dir in root.iterdir():
            if (
                dataset_dir.name.startswith(".")
                or not _DATASET_ID_RE.fullmatch(dataset_dir.name)
                or dataset_dir.is_symlink()
                or not dataset_dir.is_dir()
            ):
                continue
            metadata = _read_metadata(dataset_dir)
            if metadata is not None:
                entries.append(metadata)
    except OSError as error:
        raise DatasetStorageError(f"Cannot list uploaded datasets in {root}: {error}") from error
    return sorted(entries, key=lambda item: (item["name"].casefold(), item["id"]))


def save_uploaded_dataset(
    name: str,
    train_filename: str,
    train_content: bytes,
    test_filename: str,
    test_content: bytes,
) -> str:
    """Validate and atomically save a user-uploaded TRAIN/TEST .ts pair."""
    name = _safe_name(name)
    _validate_upload_file(train_filename, train_content, "TRAIN")
    _validate_upload_file(test_filename, test_content, "TEST")

    root = _storage_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise NotADirectoryError(f"Dataset storage path is not a directory: {root}")
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=root) as staging_name:
            staging_dir = Path(staging_name)
            train_path = staging_dir / "train.ts"
            test_path = staging_dir / "test.ts"
            train_path.write_bytes(train_content)
            test_path.write_bytes(test_content)
            train_info = _parse_and_validate(train_path, "TRAIN")
            test_info = _parse_and_validate(test_path, "TEST")

            if train_info["length"] != test_info["length"]:
                raise DatasetValidationError(
                    "TRAIN and TEST series must have the same length "
                    f"(got {train_info['length']} and {test_info['length']})."
                )
            if train_info["dimensions"] != test_info["dimensions"]:
                raise DatasetValidationError(
                    "TRAIN and TEST series must have the same number of dimensions "
                    f"(got {train_info['dimensions']} and {test_info['dimensions']})."
                )
            train_classes = set(train_info["labels"])
            test_classes = set(test_info["labels"])
            unknown_classes = sorted(test_classes - train_classes)
            if unknown_classes:
                raise DatasetValidationError(
                    "TEST contains class label(s) absent from TRAIN: "
                    + ", ".join(repr(label) for label in unknown_classes)
                    + ". Add those classes to TRAIN or correct the labels."
                )

            dataset_id = uuid.uuid4().hex
            metadata = {
                "id": dataset_id,
                "name": name,
                "train_samples": train_info["samples"],
                "test_samples": test_info["samples"],
                "length": train_info["length"],
                "dimensions": train_info["dimensions"],
                "classes": sorted(train_classes),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path = staging_dir / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
                metadata_file.flush()
                os.fsync(metadata_file.fileno())

            for file_path in (train_path, test_path):
                with file_path.open("rb") as data_file:
                    os.fsync(data_file.fileno())
            dir_fd = os.open(staging_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

            with _storage_lock(root):
                existing_names = {
                    item["name"].casefold() for item in list_uploaded_datasets()
                }
                if name.casefold() in existing_names:
                    raise DatasetValidationError(
                        f"An uploaded dataset named {name!r} already exists. "
                        "Choose a different name."
                    )
                final_dir = root / dataset_id
                os.rename(staging_dir, final_dir)
                root_fd = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(root_fd)
                finally:
                    os.close(root_fd)
            return dataset_id
    except DatasetValidationError:
        raise
    except OSError as error:
        raise DatasetStorageError(
            f"Could not save uploaded dataset in {root}: {error}. "
            "Check that the configured dataset directory is writable."
        ) from error


def _resolve_uploaded_dataset(name: str) -> tuple[Path, dict]:
    root = _storage_dir()
    if not root.exists() or not root.is_dir():
        raise DatasetStorageError(f"No uploaded datasets are available in {root}.")

    if isinstance(name, str) and _DATASET_ID_RE.fullmatch(name):
        candidates = [root / name]
    else:
        candidates = [
            root / item["id"]
            for item in list_uploaded_datasets()
            if item["name"] == name
        ]
    for dataset_dir in candidates:
        if dataset_dir.is_symlink() or not dataset_dir.is_dir():
            continue
        metadata = _read_metadata(dataset_dir)
        if metadata is None:
            continue
        train_path = dataset_dir / "train.ts"
        test_path = dataset_dir / "test.ts"
        if (
            train_path.is_symlink()
            or test_path.is_symlink()
            or not train_path.is_file()
            or not test_path.is_file()
        ):
            continue
        return dataset_dir, metadata
    raise DatasetValidationError(
        f"Uploaded dataset {name!r} was not found. Select an ID or name from the uploaded dataset list."
    )


def load_dataset(source: str, name: str) -> DatasetContainer:
    """Load a UCR/UEA dataset or a stored upload as a DatasetContainer."""
    source = source.strip().lower() if isinstance(source, str) else source
    if source == "ucr":
        return DatasetContainer(name, mode="ucr", dtype=torch.double)
    if source != "uploaded":
        raise ValueError("Dataset source must be 'ucr' or 'uploaded'.")

    dataset_dir, metadata = _resolve_uploaded_dataset(name)
    return DatasetContainer(
        metadata["name"],
        mode="ts",
        dtype=torch.double,
        train=str(dataset_dir / "train.ts"),
        test=str(dataset_dir / "test.ts"),
    )
