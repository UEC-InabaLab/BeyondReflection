"""Fast structural checks for packaged inputs (no statistical analysis)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from build_analysis_tables import build_analysis_tables


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    dialogue_files = list((ROOT / "annotated_kokorochat").glob("*.json"))
    assert len(dialogue_files) == 6_589, len(dialogue_files)
    with tempfile.TemporaryDirectory() as temporary:
        temporary_dir = Path(temporary)
        turns, sessions = build_analysis_tables(
            ROOT / "annotated_kokorochat",
            temporary_dir / "turn_level_analysis.csv",
            temporary_dir / "session_level_analysis.csv",
        )
        assert len(turns) == 229_743, len(turns)
        assert len(sessions) == 6_589, len(sessions)
        assert turns["session_id"].nunique() == 6_589
        assert sessions["session_id"].is_unique
        assert set(turns["distress_level"].dropna().unique()) <= {0, 1, 2, 3}
    sample = json.loads(dialogue_files[0].read_text(encoding="utf-8"))
    assert "dialogue" in sample and "review_by_client_jp" in sample
    client_turn = next(turn for turn in sample["dialogue"] if turn["role"] == "client")
    assert client_turn["distress_level"] in {0, 1, 2, 3}
    assert client_turn["disclosure_depth"] in {0, 1, 2, 3}
    evaluation = DATA / "annotation_evaluation"
    strategy = pd.read_csv(evaluation / "strategy_labels.csv")
    assert len(strategy) == 318
    assert strategy["item_id"].is_unique
    assert not strategy[
        ["llm_tag", "annotator_1_tag", "annotator_2_tag"]
    ].isna().any().any()
    distress = pd.read_csv(evaluation / "distress_labels.csv")
    assert len(distress) == 204
    assert distress[["session_id", "block_id"]].duplicated().sum() == 0
    for column in (
        "llm_distress",
        "annotator_1_distress",
        "annotator_2_distress",
    ):
        assert set(distress[column].unique()) <= {0, 1, 2, 3}
    print("Input validation passed")


if __name__ == "__main__":
    main()
