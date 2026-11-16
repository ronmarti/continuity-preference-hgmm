import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch

from gui import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ts_file(rows, dimensions=1, declared_classes=("walking", "running"), **headers):
    """Build a compact labeled .ts file for exercising the upload API."""
    header = {
        "timestamps": "false",
        "univariate": "true" if dimensions == 1 else "false",
        "equalLength": "true",
        "seriesLength": "3",
        "classLabel": "true " + " ".join(declared_classes),
    }
    header.update(headers)
    lines = ["@problemName test"]
    lines.extend(f"@{key} {value}" for key, value in header.items())
    lines.append("@data")
    lines.extend(rows)
    return ("\n".join(lines) + "\n").encode()


def univariate_pair(train_rows=None, test_rows=None):
    train_rows = train_rows or [
        "1,2,3:walking",
        "4,5,6:running",
    ]
    test_rows = test_rows or ["7,8,9:walking"]
    return (
        ts_file(train_rows),
        ts_file(test_rows),
    )


class UploadedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.test_root = PROJECT_ROOT / f".test-datasets-{uuid.uuid4().hex}"
        self.test_root.mkdir()
        self.storage = self.test_root / "stored"
        self.env = patch.dict(os.environ, {"CPHGMM_DATASETS_DIR": str(self.storage)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(shutil.rmtree, self.test_root, ignore_errors=True)

    def save(self, train, test, name="Sample"):
        return datasets.save_uploaded_dataset(
            name, "my train.ts", train, "my test.ts", test
        )

    def test_saves_and_reloads_univariate_pair(self):
        train, test = univariate_pair()
        dataset_id = self.save(train, test)

        entries = datasets.list_uploaded_datasets()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], dataset_id)
        self.assertEqual(entries[0]["name"], "Sample")
        self.assertEqual(entries[0]["length"], 3)
        self.assertEqual(entries[0]["dimensions"], 1)
        self.assertEqual(entries[0]["train_samples"], 2)

        loaded = datasets.load_dataset("uploaded", dataset_id)
        self.assertEqual(len(loaded.dataset_train), 2)
        self.assertEqual(len(loaded.dataset_test), 1)
        self.assertEqual(tuple(loaded.dataset_train.tensors[1].shape), (2, 3, 1, 1))

    def test_saves_multivariate_pair(self):
        train = ts_file(
            ["1,2,3:11,12,13:walking", "4,5,6:14,15,16:running"],
            dimensions=2,
        )
        test = ts_file(["7,8,9:17,18,19:walking"], dimensions=2)

        dataset_id = self.save(train, test, "Multivariate")
        loaded = datasets.load_dataset("uploaded", dataset_id)

        self.assertEqual(loaded.output_dim, 2)
        self.assertEqual(tuple(loaded.dataset_train.tensors[1].shape), (2, 3, 2, 1))
        self.assertEqual(datasets.list_uploaded_datasets()[0]["dimensions"], 2)

    def test_rejects_pair_with_different_train_test_shapes(self):
        train, _ = univariate_pair()
        short_test = ts_file(["1,2,3,4:walking"], seriesLength="4")
        with self.assertRaisesRegex(datasets.DatasetValidationError, "same length"):
            self.save(train, short_test)

        multivariate_test = ts_file(["1,2,3:4,5,6:walking"], dimensions=2)
        with self.assertRaisesRegex(datasets.DatasetValidationError, "dimensions"):
            self.save(train, multivariate_test)

    def test_rejects_nonfinite_and_missing_values(self):
        _, test = univariate_pair()
        train_nonfinite = ts_file(["1,NaN,3:walking"])
        with self.assertRaisesRegex(datasets.DatasetValidationError, "missing or non-finite"):
            self.save(train_nonfinite, test)

        train_missing = ts_file(["1,,3:walking"])
        with self.assertRaises(datasets.DatasetValidationError):
            self.save(train_missing, test)

    def test_rejects_missing_labels_and_unknown_test_classes(self):
        test = ts_file(["1,2,3:walking"])
        missing_label = ts_file(["1,2,3:"])
        with self.assertRaises(datasets.DatasetValidationError):
            self.save(missing_label, test)

        train = ts_file(["1,2,3:walking"])
        unknown_test = ts_file(["1,2,3:running"])
        with self.assertRaisesRegex(datasets.DatasetValidationError, "absent from TRAIN"):
            self.save(train, unknown_test)

    def test_rejects_timestamps_and_too_short_series(self):
        test = ts_file(["1,2,3:walking"])
        timestamped = ts_file(["1,2,3:walking"], timestamps="true")
        with self.assertRaisesRegex(datasets.DatasetValidationError, "timestamps"):
            self.save(timestamped, test)

        too_short = ts_file(["1,2:walking"], seriesLength="2")
        with self.assertRaisesRegex(datasets.DatasetValidationError, "at least 3"):
            self.save(too_short, ts_file(["1,2:walking"], seriesLength="2"))

    def test_duplicate_names_are_rejected_case_insensitively(self):
        train, test = univariate_pair()
        self.save(train, test, "My Dataset")
        with self.assertRaisesRegex(datasets.DatasetValidationError, "already exists"):
            self.save(train, test, "my dataset")

    def test_untrusted_filenames_are_not_used_as_storage_paths(self):
        train, test = univariate_pair()
        dataset_id = datasets.save_uploaded_dataset(
            "Safe Name", "../outside/train.ts", train, r"..\outside\test.ts", test
        )
        dataset_dir = self.storage / dataset_id
        self.assertTrue((dataset_dir / "train.ts").is_file())
        self.assertTrue((dataset_dir / "test.ts").is_file())
        self.assertFalse((self.test_root / "outside").exists())

        with self.assertRaisesRegex(datasets.DatasetValidationError, "path separators"):
            self.save(train, test, "../outside")

    def test_storage_errors_are_actionable(self):
        blocked_path = self.test_root / "not-a-directory"
        blocked_path.write_text("block", encoding="utf-8")
        with patch.dict(os.environ, {"CPHGMM_DATASETS_DIR": str(blocked_path)}):
            train, test = univariate_pair()
            with self.assertRaisesRegex(datasets.DatasetStorageError, "writable"):
                self.save(train, test)

    def test_ucr_source_routes_to_dataset_container(self):
        sentinel = object()
        with patch.object(datasets, "DatasetContainer", return_value=sentinel) as factory:
            loaded = datasets.load_dataset("ucr", "Coffee")

        self.assertIs(loaded, sentinel)
        factory.assert_called_once_with("Coffee", mode="ucr", dtype=torch.double)

    def test_upload_size_limit_is_enforced_before_writing(self):
        train, test = univariate_pair()
        with patch.object(datasets, "MAX_UPLOAD_BYTES", 10):
            with self.assertRaisesRegex(datasets.DatasetValidationError, "limit is"):
                self.save(train, test)
        self.assertEqual(datasets.list_uploaded_datasets(), [])

    def test_rejects_unequal_length_series_within_a_split(self):
        unequal = ts_file(["1,2,3:walking", "4,5:running"])
        test = ts_file(["1,2,3:walking"])
        with self.assertRaises(datasets.DatasetValidationError):
            self.save(unequal, test)


if __name__ == "__main__":
    unittest.main()
