"""Uji fungsi metrik dan reproduksibilitas dataset."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluasi import bangkitkan_data, metrik  # noqa: E402


class UjiMetrik(unittest.TestCase):
    def test_tugas_1(self):
        m = metrik(24, 156, 12, 8)
        self.assertAlmostEqual(m["accuracy"], 0.9, places=6)
        self.assertAlmostEqual(m["precision"], 24 / 36, places=6)
        self.assertAlmostEqual(m["recall"], 0.75, places=6)
        self.assertAlmostEqual(m["f1"], 2 * (24 / 36) * 0.75 / ((24 / 36) + 0.75), places=6)

    def test_penyebut_nol(self):
        m = metrik(0, 10, 0, 0)
        self.assertEqual(m["precision"], 0)
        self.assertEqual(m["recall"], 0)
        self.assertEqual(m["f1"], 0)

    def test_akurasi_sempurna(self):
        m = metrik(10, 90, 0, 0)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)


class UjiDataset(unittest.TestCase):
    def test_ukuran_dan_kelas(self):
        df = bangkitkan_data()
        self.assertEqual(len(df), 500)
        self.assertEqual(list(df.columns), ["amount", "hour", "new_device", "foreign_ip", "isFraud"])
        self.assertEqual(int(df["isFraud"].sum()), 16)
        self.assertEqual(int((df["isFraud"] == 0).sum()), 484)

    def test_deterministik(self):
        a = bangkitkan_data()
        b = bangkitkan_data()
        self.assertTrue(a.equals(b))


if __name__ == "__main__":
    unittest.main()
