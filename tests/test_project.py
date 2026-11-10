import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import requests

import champion_collection
import champion_descriptions
import data_io
from evaluate_recommender import evaluate
from kit_similarity import ability_matrix, kit_similarity, match_abilities
from recommender import load_player_pools, recommend
from tagging import TAG_LABELS, get_tags, shared_tags, tag_evidence
from validate_data import validate


class DataTests(unittest.TestCase):
    def test_frozen_data_has_complete_joins(self):
        players, champions, counts = validate()
        self.assertGreater(players, 10_000)
        self.assertEqual(champions, len(counts))

    def test_no_riot_key_means_no_request(self):
        with patch.dict(os.environ, {"RIOT_API_KEY": ""}), patch.object(
            champion_collection, "get_champions_by_id"
        ) as catalog:
            with self.assertRaisesRegex(ValueError, "RIOT_API_KEY"):
                champion_collection.main()
        catalog.assert_not_called()

    def test_collector_resumes_without_duplicate_players(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "player_pools.csv"
            with (
                patch.dict(os.environ, {"RIOT_API_KEY": "test-only"}),
                patch.object(champion_collection, "OUTPUT_PATH", output),
                patch.object(champion_collection, "get_champions_by_id",
                             return_value={1: "Annie", 2: "Lee Sin"}),
                patch.object(champion_collection, "get_players_puuid",
                             return_value=["a", "a", "b"]),
                patch.object(champion_collection, "get_top_champions",
                             side_effect=[[1, 2], [1, 2]]) as mastery,
            ):
                champion_collection.main()
                champion_collection.main()
            self.assertEqual(mastery.call_count, 2)
            self.assertEqual(data_io.read_rows(output), [
                {"puuid": "a", "champions": "Annie|Lee Sin"},
                {"puuid": "b", "champions": "Annie|Lee Sin"},
            ])

    def test_collector_appends_after_csv_without_a_final_newline(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "player_pools.csv"
            output.write_text("puuid,champions\na,Annie|Lee Sin", encoding="utf-8")
            with (
                patch.dict(os.environ, {"RIOT_API_KEY": "test-only"}),
                patch.object(champion_collection, "OUTPUT_PATH", output),
                patch.object(champion_collection, "get_champions_by_id",
                             return_value={1: "Annie", 2: "Lee Sin"}),
                patch.object(champion_collection, "get_players_puuid", return_value=["a", "b"]),
                patch.object(champion_collection, "get_top_champions", return_value=[1, 2]),
            ):
                champion_collection.main()
            self.assertEqual(data_io.read_player_pools(output), {
                "a": ["Annie", "Lee Sin"], "b": ["Annie", "Lee Sin"],
            })

    def test_invalid_player_rows_cannot_reach_training(self):
        cases = (
            ("puuid,champions\na,Annie|Lee Sin\na,Annie|Thresh\n", "duplicate ID"),
            ("puuid,champions\na\n", "wrong number of columns"),
            ("champions,puuid\nAnnie|Lee Sin,a\n", "invalid CSV header"),
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "pools.csv"
            for contents, error in cases:
                with self.subTest(error=error):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_player_pools(path)

    def test_failed_description_refresh_preserves_existing_csvs(self):
        with TemporaryDirectory() as folder:
            raw = Path(folder) / "champion_descriptions.csv"
            clean = Path(folder) / "champions_clean.csv"
            for path in (raw, clean):
                path.write_text("saved", encoding="utf-8")
            with (
                patch.object(champion_descriptions, "OUTPUT_PATH", raw),
                patch.object(champion_descriptions, "get_champion_names",
                             return_value=["Annie"]),
                patch.object(champion_descriptions, "get_name_to_ddragon_id",
                             return_value={"Annie": "Annie"}),
                patch.object(champion_descriptions, "get_json",
                             side_effect=requests.Timeout),
            ):
                with self.assertRaises(requests.Timeout):
                    champion_descriptions.main("fixture")
            self.assertEqual(raw.read_text(encoding="utf-8"), "saved")
            self.assertEqual(clean.read_text(encoding="utf-8"), "saved")

    def test_rate_limit_retries_are_bounded(self):
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "1"
        with (
            patch.object(data_io.requests, "get", return_value=response) as fetch,
            patch.object(data_io.time, "sleep") as sleep,
        ):
            with self.assertRaises(requests.HTTPError):
                data_io.get_json("https://example.test")
        self.assertEqual(fetch.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    def test_authenticated_request_rejects_redirects(self):
        response = requests.Response()
        response.status_code = 302
        with patch.object(data_io.requests, "get", return_value=response) as fetch:
            with self.assertRaisesRegex(requests.HTTPError, "redirected"):
                data_io.get_json("https://example.test", headers={"X-Riot-Token": "fixture"})
        self.assertFalse(fetch.call_args.kwargs["allow_redirects"])

    def test_clean_publication_failure_restores_raw_file(self):
        with TemporaryDirectory() as folder:
            raw = Path(folder) / "champion_descriptions.csv"
            clean = raw.with_name("champions_clean.csv")
            for path in (raw, clean):
                path.write_bytes(b"original\r\n")
            original_replace = Path.replace

            def replace(source, target):
                if target == clean:
                    raise PermissionError("Clean CSV is open in another program")
                return original_replace(source, target)

            raw_row = {"champion": "Annie", **dict.fromkeys(data_io.ABILITIES, "new text")}
            with (
                patch.object(champion_descriptions, "OUTPUT_PATH", raw),
                patch.object(Path, "replace", replace),
            ):
                with self.assertRaises(PermissionError):
                    champion_descriptions.publish_descriptions([raw_row], [{"champion": "Annie"}])
            self.assertEqual(raw.read_bytes(), b"original\r\n")
            self.assertEqual(clean.read_bytes(), b"original\r\n")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.abilities = data_io.load_abilities()

    def test_recommendation_normalizes_inputs_and_excludes_mains(self):
        result = recommend(["  lee sin ", "Lee Sin", "Thresh"], 3)
        self.assertEqual(result, recommend(["Lee Sin", "Thresh"], 3))
        self.assertEqual(len(result), 3)
        self.assertTrue({"Lee Sin", "Thresh"}.isdisjoint(name for name, _ in result))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            recommend(["missing champion"])

    def test_evaluator_keeps_test_only_targets_in_denominator(self):
        pools = [[f"a{i}", f"b{i}"] for i in range(10)]
        training, counts, hits = evaluate(pools)
        self.assertEqual((training, counts["rare"]), (8, 2))
        self.assertEqual(sum(hits.values()), 0)

    def test_kit_similarity_is_symmetric_and_matches_real_hooks(self):
        thresh, blitz = self.abilities["Thresh"], self.abilities["Blitzcrank"]
        result = kit_similarity(thresh, blitz)
        self.assertAlmostEqual(result["score"], kit_similarity(blitz, thresh)["score"])
        self.assertEqual(ability_matrix(thresh, blitz).shape, (5, 5))
        self.assertIn(("Q", "Q"), [(a, b) for a, b, _ in result["matches"]])
        self.assertEqual(kit_similarity({}, {})["matches"], [])

    def test_weak_pairs_cannot_displace_a_valid_match(self):
        scores = np.zeros((5, 5))
        scores[:2, :2] = [[0.21, 0.19], [0.19, 0.0]]
        self.assertEqual(match_abilities(scores), [("P", "P", 0.21)])

    def test_tags_keep_evidence_and_distinguish_confusable_mechanics(self):
        self.assertIn("shield break", tag_evidence({"Q": self.abilities["Rell"]["Q"]}))
        self.assertNotIn("shield", tag_evidence({"Q": self.abilities["Rell"]["Q"]}))
        self.assertIn("shield", tag_evidence(self.abilities["Lulu"]))
        self.assertNotIn("hook", tag_evidence({"Q": self.abilities["Amumu"]["Q"]}))
        garen = tag_evidence({"Q": self.abilities["Garen"]["Q"]})
        self.assertEqual(garen["crowd control"], [("Q", "silencing")])
        thresh = get_tags(self.abilities["Thresh"], TAG_LABELS)
        blitz = get_tags(self.abilities["Blitzcrank"], TAG_LABELS)
        self.assertIn("hook", shared_tags(thresh, blitz))

    def test_immunity_and_self_root_are_not_enemy_crowd_control(self):
        yi = tag_evidence({"R": self.abilities["Master Yi"]["R"]})
        belveth = tag_evidence({"E": self.abilities["Bel'Veth"]["E"]})
        self.assertNotIn("crowd control", yi)
        self.assertNotIn("crowd control", belveth)
        self.assertIn("crowd control", tag_evidence({"Q": "Cleanses a root and stuns an enemy."}))

    def test_real_self_movement_and_spell_shield_have_evidence(self):
        amumu = tag_evidence({"Q": self.abilities["Amumu"]["Q"]})
        sivir = tag_evidence({"E": self.abilities["Sivir"]["E"]})
        self.assertEqual(amumu["mobility"], [("Q", "pulls himself")])
        self.assertIn("shield", sivir)
        self.assertNotIn("crowd control", tag_evidence({"Q": "Pulls herself toward an enemy."}))
