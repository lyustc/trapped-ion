import json
import tempfile
import unittest
from pathlib import Path

from src.lit_digest import (
    EMBED_DIM,
    Paper,
    PaperStore,
    PreferenceProfile,
    apply_feedback_update,
    build_preferences_from_zotero_export,
    cosine_similarity,
    detect_category,
    generate_weekly_report,
    load_subscription_config,
    text_to_embedding,
)


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
        self.assertGreater(profile.score(paper), 1.0)

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

    def test_embedding_similarity(self):
        v1 = text_to_embedding("quantum trapped ion", dim=EMBED_DIM)
        v2 = text_to_embedding("quantum trapped ion control", dim=EMBED_DIM)
        v3 = text_to_embedding("protein cell genome", dim=EMBED_DIM)
        self.assertGreater(cosine_similarity(v1, v2), cosine_similarity(v1, v3))

    def test_store_upsert_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            store = PaperStore(db)
            paper = Paper("arxiv", "x1", "title", "summary", ["A B"], "http://a", "2025-01-01")
            store.upsert(paper, "other", 3.1)
            rows = store.top_papers(1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][2], "title")
            json.loads(rows[0][4])

    def test_feedback_updates_preferences(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            pref = Path(td) / "preferences.json"
            pref.write_text(json.dumps({"likes": [{"weight": 2, "keywords": [], "authors": []}]}), encoding="utf-8")

            store = PaperStore(db)
            store.upsert(
                Paper("arxiv", "id1", "title", "quantum control trapped ion entanglement", ["Alice Smith"], "", "2026-01-01"),
                "quantum",
                2.0,
            )
            store.add_feedback("arxiv", "id1", "like", "great")

            apply_feedback_update(db_path=db, history_path=str(pref))
            data = json.loads(pref.read_text(encoding="utf-8"))
            self.assertIn("quantum", data["likes"][0]["keywords"])

    def test_weekly_report_generation(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            out = Path(td) / "weekly.md"
            store = PaperStore(db)
            store.upsert(
                Paper("nature", "n1", "A quantum result", "summary text", ["A B"], "http://x", "2099-01-01"),
                "quantum",
                5.0,
            )
            generate_weekly_report(db_path=db, output_path=str(out), days=36500)
            content = out.read_text(encoding="utf-8")
            self.assertIn("Weekly Research Report", content)
            self.assertIn("LLM Summary", content)

    def test_load_subscription_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "subscriptions.json"
            cfg.write_text(
                json.dumps(
                    {
                        "arxiv_query": "cat:quant-ph",
                        "arxiv_max_results": 12,
                        "rss_feeds": {"prx": "https://journals.aps.org/rss/recent/prx.xml"},
                    }
                ),
                encoding="utf-8",
            )
            query, max_results, feeds = load_subscription_config(str(cfg))
            self.assertEqual(query, "cat:quant-ph")
            self.assertEqual(max_results, 12)
            self.assertIn("prx", feeds)

    def test_build_preferences_from_zotero_export_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            export_path = Path(td) / "zotero.json"
            out_path = Path(td) / "prefs.json"
            export_path.write_text(
                json.dumps(
                    [
                        {
                            "title": "Trapped-ion quantum computing",
                            "abstractNote": "Entanglement and error correction in ion qubits",
                            "creators": [{"firstName": "Rainer", "lastName": "Blatt"}],
                            "tags": [{"tag": "quantum"}, {"tag": "ion trap"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            build_preferences_from_zotero_export(str(export_path), output_path=str(out_path), top_k_keywords=8)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("likes", data)
            self.assertGreaterEqual(len(data["likes"]), 1)
            self.assertIn("Rainer Blatt", data["likes"][0]["authors"])


if __name__ == "__main__":
    unittest.main()
