import json
import tempfile
import unittest
from pathlib import Path

from src.lit_digest import Paper, PreferenceProfile, PaperStore, detect_category


class LitDigestTests(unittest.TestCase):
    def test_profile_score_prefers_keyword_and_author(self):
        profile = PreferenceProfile(
            keywords={"trapped ion": 2.0, "entanglement": 1.0},
            authors={"alice smith": 2.0},
        )
        paper = Paper(
            source="arxiv",
            source_id="1",
            title="Trapped ion entanglement benchmark",
            summary="We demonstrate fidelity improvement.",
            authors=["Alice Smith"],
            link="http://example.com",
            published_at="2026-01-01",
        )
        self.assertGreater(profile.score(paper), 5.0)

    def test_detect_category_quantum(self):
        paper = Paper(
            source="nature",
            source_id="2",
            title="A qubit control method",
            summary="entanglement and trapped ion hardware",
            authors=[],
            link="",
            published_at="",
        )
        self.assertEqual(detect_category(paper), "quantum")

    def test_store_upsert_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            store = PaperStore(db)
            paper = Paper("arxiv", "x1", "title", "summary", ["A B"], "http://a", "2025-01-01")
            store.upsert(paper, "other", 3.1)
            rows = store.top_papers(1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], "title")
            json.loads(rows[0][2])


if __name__ == "__main__":
    unittest.main()
