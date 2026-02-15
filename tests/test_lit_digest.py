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
    detect_article_type,
    detect_category,
    detect_tags,
    generate_weekly_report,
    load_subscription_config,
    text_to_embedding,
    update_preferences_from_zotero_export,
)


class LitDigestTests(unittest.TestCase):
    def test_profile_score_prefers_keyword_and_author(self):
        profile = PreferenceProfile(keywords={"trapped ion": 2.0, "entanglement": 1.0}, authors={"alice smith": 2.0})
        paper = Paper("arxiv", "1", "Trapped ion entanglement benchmark", "fidelity", ["Alice Smith"], "http://x", "2026-01-01")
        self.assertGreater(profile.score(paper), 1.0)

    def test_detect_category_quantum(self):
        paper = Paper("nature", "2", "A qubit control method", "entanglement and trapped ion hardware", [], "", "")
        self.assertEqual(detect_category(paper), "quantum")

    def test_detect_tags_and_type(self):
        p = Paper("x", "1", "A review of quantum simulation experiments", "experimental review", [], "", "")
        tags = detect_tags(p)
        self.assertIn("quantum-simulation", tags)
        self.assertIn("experiment", tags)
        self.assertEqual(detect_article_type(p), "review")

    def test_embedding_similarity(self):
        v1 = text_to_embedding("quantum trapped ion", dim=EMBED_DIM)
        v2 = text_to_embedding("quantum trapped ion control", dim=EMBED_DIM)
        v3 = text_to_embedding("protein cell genome", dim=EMBED_DIM)
        self.assertGreater(cosine_similarity(v1, v2), cosine_similarity(v1, v3))

    def test_store_upsert_query_status_and_tag(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            store = PaperStore(db)
            paper = Paper("arxiv", "x1", "title", "summary", ["A B"], "http://a", "2025-01-01")
            store.upsert(paper, "quantum", 3.1, ["quantum-computing", "theory"], "article")
            rows = store.top_papers(10, status="unread", tag="quantum-computing")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][2], "title")
            store.mark_read_status([("arxiv", "x1")], "read")
            self.assertEqual(len(store.top_papers(10, status="read")), 1)

    def test_prune_old(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            store = PaperStore(db)
            store.upsert(Paper("a", "1", "old", "s", [], "", "2000-01-01T00:00:00+00:00"), "other", 1, [], "article")
            store.upsert(Paper("a", "2", "new", "s", [], "", "2099-01-01T00:00:00+00:00"), "other", 1, [], "article")
            store.prune_old_papers(keep_days=3)
            rows = store.top_papers(10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], "2")

    def test_feedback_updates_preferences(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            pref = Path(td) / "preferences.json"
            pref.write_text(json.dumps({"likes": [{"weight": 2, "keywords": [], "authors": []}]}), encoding="utf-8")
            store = PaperStore(db)
            store.upsert(Paper("arxiv", "id1", "title", "quantum control trapped ion entanglement", ["Alice Smith"], "", "2026-01-01"), "quantum", 2.0, ["quantum-computing"], "article")
            store.add_feedback("arxiv", "id1", "like", "great")
            apply_feedback_update(db, str(pref))
            data = json.loads(pref.read_text(encoding="utf-8"))
            self.assertIn("quantum", data["likes"][0]["keywords"])

    def test_weekly_report_generation(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "papers.db")
            out = Path(td) / "weekly.md"
            store = PaperStore(db)
            store.upsert(Paper("nature", "n1", "A quantum result", "summary text", ["A B"], "http://x", "2099-01-01"), "quantum", 5.0, ["quantum-computing"], "article")
            generate_weekly_report(db_path=db, output_path=str(out), days=36500)
            content = out.read_text(encoding="utf-8")
            self.assertIn("Weekly Research Report", content)
            self.assertIn("LLM Summary", content)

    def test_load_subscription_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "subscriptions.json"
            cfg.write_text(json.dumps({"arxiv_query": "cat:quant-ph", "arxiv_max_results": 12, "rss_feeds": {"prx": "https://journals.aps.org/rss/recent/prx.xml"}}), encoding="utf-8")
            query, max_results, feeds = load_subscription_config(str(cfg))
            self.assertEqual(query, "cat:quant-ph")
            self.assertEqual(max_results, 12)
            self.assertIn("prx", feeds)

    def test_build_preferences_from_zotero_export_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            export_path = Path(td) / "zotero.json"
            out_path = Path(td) / "prefs.json"
            export_path.write_text(json.dumps([{"title": "Trapped-ion quantum computing", "abstractNote": "Entanglement and error correction in ion qubits", "creators": [{"firstName": "Rainer", "lastName": "Blatt"}], "tags": [{"tag": "quantum"}, {"tag": "ion trap"}]}]), encoding="utf-8")
            build_preferences_from_zotero_export(str(export_path), output_path=str(out_path), top_k_keywords=8)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("likes", data)
            self.assertGreaterEqual(len(data["likes"]), 1)
            self.assertIn("Rainer Blatt", data["likes"][0]["authors"])

    def test_update_preferences_from_zotero_merges_into_history(self):
        with tempfile.TemporaryDirectory() as td:
            export_path = Path(td) / "zotero.json"
            pref_path = Path(td) / "preferences.json"
            pref_path.write_text(json.dumps({"likes": [{"weight": 2.0, "keywords": ["existing"], "authors": []}]}), encoding="utf-8")
            export_path.write_text(
                json.dumps([
                    {
                        "title": "Quantum simulation review",
                        "abstractNote": "theory and experiment",
                        "creators": [{"firstName": "Jane", "lastName": "Doe"}],
                        "tags": [{"tag": "quantum simulation"}],
                        "collections": [{"name": "Quantum"}],
                    }
                ]),
                encoding="utf-8",
            )
            update_preferences_from_zotero_export(str(export_path), history_path=str(pref_path), top_k_keywords=8)
            merged = json.loads(pref_path.read_text(encoding="utf-8"))
            self.assertIn("existing", merged["likes"][0]["keywords"])
            self.assertIn("jane doe", [a.lower() for a in merged["likes"][0]["authors"]])



if __name__ == "__main__":
    unittest.main()
