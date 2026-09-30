import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools/build_csl_strict_train.py"
SPEC = importlib.util.spec_from_file_location("build_csl_strict_train", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildCslStrictTrainTest(unittest.TestCase):
    def test_filter_preserves_order_and_removes_quarantine(self):
        records = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        kept, removed = MODULE.filter_annotations(records, {"b"})
        self.assertEqual([item["name"] for item in kept], ["a", "c"])
        self.assertEqual([item["name"] for item in removed], ["b"])

    def test_missing_quarantine_entry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            MODULE.filter_annotations([{"name": "a"}], {"missing"})

    def test_duplicate_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.filter_annotations([{"name": "a"}, {"name": "a"}], set())


if __name__ == "__main__":
    unittest.main()
