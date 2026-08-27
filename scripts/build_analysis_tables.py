"""Build the paper's analysis-ready CSV files from ``annotated_kokorochat``.

The two CSV files are derived data, not independent research inputs.  This
module implements the filtering and aggregation rules used by the paper so
that their provenance is explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


GAP_MAX_SECONDS = 300
MIN_UTTERANCES = 10
SESSION_SCORE_P33 = 55
SESSION_SCORE_P66 = 74
TAGS = [
    "Affirmation",
    "Backchannel",
    "ClosedQuestion",
    "Greeting",
    "Inform",
    "OpenQuestion",
    "Other",
    "Paraphrase",
    "Reflection",
    "Suggest",
    "Thanking",
]
TURN_COLUMNS = [
    "session_id",
    "idx",
    "tag",
    "session_phase",
    "distress_level",
    "score",
    "score_tier",
    "counselor_id",
]
SESSION_COLUMNS = [
    "session_id",
    "score",
    "tier",
    "delta_distress",
    "dist_early",
    *[f"rate_{tag}" for tag in TAGS],
]


def normalize_tag(value: object) -> str:
    return str(value or "Other").strip()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("/", "-"))


def load_sessions(annotated_dir: Path) -> list[dict]:
    paths = sorted(annotated_dir.glob("*.json"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"No JSON sessions found in {annotated_dir}")
    sessions = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        sessions.append(
            {
                "session_id": int(path.stem),
                "document": document,
                "utterances": parse_utterances(document["dialogue"]),
            }
        )
    return sessions


def parse_utterances(dialogue: list[dict]) -> list[dict]:
    rows = []
    previous_time: datetime | None = None
    previous_role: str | None = None
    for idx, utterance in enumerate(dialogue):
        current_time = parse_time(utterance["time"])
        role = utterance["role"]
        rows.append(
            {
                "idx": idx,
                "role": role,
                "tag": normalize_tag(utterance.get("tag")),
                "char_count": len(utterance["utterance"]),
                "gap": None if previous_time is None else (current_time - previous_time).total_seconds(),
                "turn_type": "cross" if role != previous_role else "same",
                "distress_level": utterance.get("distress_level"),
                "disclosure_depth": utterance.get("disclosure_depth"),
            }
        )
        previous_time = current_time
        previous_role = role
    return rows


def valid_cross_turns(utterances: list[dict], role: str) -> list[dict]:
    return [
        row
        for row in utterances
        if row["role"] == role
        and row["turn_type"] == "cross"
        and row["gap"] is not None
        and 0 < row["gap"] <= GAP_MAX_SECONDS
    ]


def estimate_alpha(rows: list[dict]) -> float | None:
    if len(rows) < MIN_UTTERANCES:
        return None
    try:
        result = stats.linregress(
            [row["char_count"] for row in rows],
            [row["gap"] for row in rows],
        )
    except ValueError:
        return None
    return None if result.slope < 0 else float(result.slope)


def global_alphas(sessions: list[dict]) -> dict[str, float]:
    result = {}
    for role in ("counselor", "client"):
        rows = [
            row
            for session in sessions
            for row in valid_cross_turns(session["utterances"], role)
        ]
        result[role] = float(
            stats.linregress(
                [row["char_count"] for row in rows],
                [row["gap"] for row in rows],
            ).slope
        )
    return result


def score_tier_thresholds(sessions: list[dict]) -> tuple[float, float]:
    scores = [
        session["document"]["review_by_client_en"]["score"]
        for session in sessions
    ]
    low, high = np.percentile(scores, [33.3, 66.7])
    return float(low), float(high)


def score_tier(score: float, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    return "low" if score <= low else "mid" if score <= high else "high"


def session_tier(score: float) -> str:
    if score <= SESSION_SCORE_P33:
        return "low"
    return "mid" if score <= SESSION_SCORE_P66 else "high"


def build_turn_table(sessions: list[dict]) -> pd.DataFrame:
    fallbacks = global_alphas(sessions)
    thresholds = score_tier_thresholds(sessions)
    records = []

    for session in sessions:
        document = session["document"]
        utterances = session["utterances"]
        session_id = session["session_id"]
        review = document["review_by_client_en"]
        score = review["score"]
        counselor_id = review["counselor_id"]
        alpha = estimate_alpha(valid_cross_turns(utterances, "counselor"))
        if alpha is None:
            alpha = fallbacks["counselor"]

        previous_client_idx: int | None = None
        candidates = []
        for row in utterances:
            if row["role"] == "client":
                previous_client_idx = row["idx"]
                continue
            if (
                row["turn_type"] != "cross"
                or row["gap"] is None
                or previous_client_idx is None
            ):
                continue
            estimated_silence = max(0.0, row["gap"] - alpha * row["char_count"])
            if estimated_silence > GAP_MAX_SECONDS:
                continue
            client = utterances[previous_client_idx]
            if (
                client["distress_level"] is None
                or client["disclosure_depth"] is None
            ):
                continue
            candidates.append(
                {
                    "session_id": session_id,
                    "idx": row["idx"],
                    "tag": row["tag"],
                    "distress_level": int(client["distress_level"]),
                    "score": score,
                    "score_tier": score_tier(score, thresholds),
                    "counselor_id": counselor_id,
                }
            )

        maximum_idx = max(record["idx"] for record in candidates)
        for record in candidates:
            relative_position = record["idx"] / (maximum_idx + 1)
            if relative_position <= 1 / 3:
                record["session_phase"] = "early"
            elif relative_position <= 2 / 3:
                record["session_phase"] = "mid"
            else:
                record["session_phase"] = "late"
            records.append(record)

    return pd.DataFrame(records)[TURN_COLUMNS]


def build_session_table(sessions: list[dict], turns: pd.DataFrame) -> pd.DataFrame:
    tag_rates: dict[int, dict[str, float]] = {}
    for session_id, group in turns.groupby("session_id", sort=False):
        counts = Counter(group["tag"])
        total = len(group)
        tag_rates[int(session_id)] = {
            f"rate_{tag}": counts[tag] / total for tag in TAGS
        }

    records = []
    for session in sessions:
        document = session["document"]
        session_id = session["session_id"]
        client_turns = [
            turn for turn in document["dialogue"] if turn["role"] == "client"
        ]
        if len(client_turns) < 6 or session_id not in tag_rates:
            continue
        third = max(1, len(client_turns) // 3)
        early = client_turns[:third]
        late = client_turns[-third:]
        dist_early = float(np.mean([turn["distress_level"] for turn in early]))
        dist_late = float(np.mean([turn["distress_level"] for turn in late]))
        score = document["review_by_client_en"]["score"]
        records.append(
            {
                "session_id": session_id,
                "score": score,
                "tier": session_tier(score),
                "delta_distress": dist_late - dist_early,
                "dist_early": dist_early,
                **tag_rates[session_id],
            }
        )

    return pd.DataFrame(records)[SESSION_COLUMNS]


def build_analysis_tables(
    annotated_dir: Path,
    turn_output: Path,
    session_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = load_sessions(annotated_dir)
    turns = build_turn_table(sessions)
    session_table = build_session_table(sessions, turns)
    turn_output.parent.mkdir(parents=True, exist_ok=True)
    turns.to_csv(turn_output, index=False)
    session_table.to_csv(session_output, index=False)
    return turns, session_table


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=repository / "annotated_kokorochat",
    )
    parser.add_argument(
        "--turn-output",
        type=Path,
        default=repository / "data" / "turn_level_analysis.csv",
    )
    parser.add_argument(
        "--session-output",
        type=Path,
        default=repository / "data" / "session_level_analysis.csv",
    )
    args = parser.parse_args()
    turns, sessions = build_analysis_tables(
        args.annotated_dir,
        args.turn_output,
        args.session_output,
    )
    print(
        f"Built {len(turns):,} turn-level rows and "
        f"{len(sessions):,} session-level rows from {args.annotated_dir}"
    )


if __name__ == "__main__":
    main()
