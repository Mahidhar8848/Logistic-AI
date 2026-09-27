import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader
from config import DATASET_DIR

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader(DATASET_DIR)

    def test_discover_datasets(self):
        files = self.loader.discover_datasets()
        self.assertGreater(len(files), 0)
        filenames = [f.name for f in files]
        self.assertIn("trips.csv", filenames)
        self.assertIn("loads.csv", filenames)

    def test_load_dataset(self):
        df = self.loader.load_dataset("routes.csv")
        self.assertGreater(len(df), 0)
        self.assertIn("origin_city", df.columns)

if __name__ == "__main__":
    unittest.main()
