"""Reproduce the results reported in the paper, and only those results."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from scripts.build_analysis_tables import build_analysis_tables

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ANNOTATED_KOKOROCHAT = ROOT / "annotated_kokorochat"
OUTPUT = ROOT / "outputs"
EXPECTED = json.loads((ROOT / "expected/paper_results.json").read_text(encoding="utf-8"))

TAGS = [
    "Affirmation",
    "Suggest",
    "Thanking",
    "Other",
    "Paraphrase",
    "Inform",
    "Backchannel",
    "Reflection",
    "Greeting",
    "OpenQuestion",
    "ClosedQuestion",
]
DISPLAY: dict[str, str] = {}


def write_json(name: str, value: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def normalize_tag(value: object) -> str:
    return str(value).strip()


def distress_agreement(folder: Path) -> tuple[pd.DataFrame, dict]:
    aligned = pd.read_csv(folder / "distress_labels.csv")
    required = [
        "session_id",
        "block_id",
        "llm_distress",
        "annotator_1_distress",
        "annotator_2_distress",
    ]
    missing = set(required) - set(aligned.columns)
    if missing:
        raise ValueError(f"distress_labels.csv is missing columns: {sorted(missing)}")
    aligned = aligned[required].astype(int)
    aligned = aligned.sort_values(["session_id", "block_id"]).reset_index(drop=True)
    if len(aligned) != 204 or aligned.duplicated(["session_id", "block_id"]).any():
        raise ValueError("distress_labels.csv must contain 204 unique client blocks")
    for column in required[2:]:
        if not set(aligned[column].unique()) <= {0, 1, 2, 3}:
            raise ValueError(f"distress_labels.csv contains an invalid value in {column}")
    labels = [0, 1, 2, 3]
    result = {
        "distress_n": len(aligned),
        "distress_annotator_1_vs_annotator_2_qwk": cohen_kappa_score(
            aligned["annotator_1_distress"],
            aligned["annotator_2_distress"],
            labels=labels,
            weights="quadratic",
        ),
        "distress_llm_vs_annotator_1_qwk": cohen_kappa_score(
            aligned["annotator_1_distress"],
            aligned["llm_distress"],
            labels=labels,
            weights="quadratic",
        ),
        "distress_llm_vs_annotator_2_qwk": cohen_kappa_score(
            aligned["annotator_2_distress"],
            aligned["llm_distress"],
            labels=labels,
            weights="quadratic",
        ),
    }
    return aligned, result


def agreement() -> dict:
    folder = DATA / "annotation_evaluation"
    labels_frame = pd.read_csv(folder / "strategy_labels.csv")
    required = {"item_id", "llm_tag", "annotator_1_tag", "annotator_2_tag"}
    missing = required - set(labels_frame.columns)
    if missing:
        raise ValueError(f"strategy_labels.csv is missing columns: {sorted(missing)}")
    if len(labels_frame) != 318 or labels_frame["item_id"].duplicated().any():
        raise ValueError("strategy_labels.csv must contain 318 unique counselor items")
    llm = labels_frame["llm_tag"].map(normalize_tag).tolist()
    human_1 = labels_frame["annotator_1_tag"].map(normalize_tag).tolist()
    human_2 = labels_frame["annotator_2_tag"].map(normalize_tag).tolist()

    result = {
        "n": len(llm),
        "annotator_1_vs_annotator_2_kappa": cohen_kappa_score(human_1, human_2),
        "llm_vs_annotator_1_kappa": cohen_kappa_score(human_1, llm),
        "llm_vs_annotator_2_kappa": cohen_kappa_score(human_2, llm),
    }
    rows = []
    labels = sorted(set(llm) | set(human_1) | set(human_2))
    metrics = []
    for human in (human_1, human_2):
        precision, recall, f1, support = precision_recall_fscore_support(
            human, llm, labels=labels, average=None, zero_division=0
        )
        metrics.append((precision, recall, f1, support))
    for index, tag in enumerate(labels):
        rows.append(
            {
                "tag": DISPLAY.get(tag, tag),
                "support": (metrics[0][3][index] + metrics[1][3][index]) / 2,
                "precision": (metrics[0][0][index] + metrics[1][0][index]) / 2,
                "recall": (metrics[0][1][index] + metrics[1][1][index]) / 2,
                "f1": (metrics[0][2][index] + metrics[1][2][index]) / 2,
            }
        )
    pd.DataFrame(rows).sort_values("f1").to_csv(OUTPUT / "tag_level_metrics.csv", index=False)
    distress_rows, distress_result = distress_agreement(folder)
    distress_rows.to_csv(OUTPUT / "distress_level_agreement.csv", index=False)
    result.update(distress_result)
    write_json("annotation_agreement.json", result)
    return result


def tier_analysis() -> dict:
    conversations = []
    for path in ANNOTATED_KOKOROCHAT.glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        score = document.get("review_by_client_jp", {}).get("点数")
        if isinstance(score, (int, float)):
            tags = [
                normalize_tag(turn["tag"])
                for turn in document.get("dialogue", [])
                if turn.get("role") == "counselor" and turn.get("tag")
            ]
            conversations.append((float(score), tags))
    scores = np.asarray([score for score, _ in conversations])
    low_threshold, high_threshold = np.percentile(scores, [33.33, 66.66])

    def tier(score: float) -> str:
        return "low" if score < low_threshold else "mid" if score < high_threshold else "high"

    counts = {name: Counter() for name in ("high", "mid", "low")}
    totals = Counter()
    for score, tags in conversations:
        name = tier(score)
        counts[name].update(tags)
        totals[name] += len(tags)

    all_tags = [tag for tag, _ in counts["high"].most_common() if tag != "Non-annotation"]
    rows = []
    for tag in all_tags:
        row = {"tag": DISPLAY.get(tag, tag)}
        for name in ("high", "mid", "low"):
            row[f"{name}_n"] = counts[name][tag]
            row[f"{name}_percent"] = counts[name][tag] / totals[name] * 100
        row["high_minus_low"] = row["high_percent"] - row["low_percent"]
        rows.append(row)
    summary = pd.DataFrame(rows)

    pairs = [("high", "mid"), ("high", "low"), ("mid", "low")]
    tests = []
    for tag in all_tags:
        for first, second in pairs:
            table = [
                [counts[first][tag], totals[first] - counts[first][tag]],
                [counts[second][tag], totals[second] - counts[second][tag]],
            ]
            _, p, _, _ = stats.chi2_contingency(table)
            tests.append({"tag": DISPLAY.get(tag, tag), "first": first, "second": second, "p": p})
    tests_frame = pd.DataFrame(tests)
    tests_frame["p_adjusted"] = multipletests(tests_frame["p"], method="bonferroni")[1]
    tests_frame["significant_at_0.01"] = tests_frame["p_adjusted"] < 0.01
    summary.to_csv(OUTPUT / "tier_usage_rates.csv", index=False)
    tests_frame.to_csv(OUTPUT / "tier_pairwise_tests.csv", index=False)
    plot_tiers(summary, tests_frame)
    return {
        "sessions": len(conversations),
        "counselor_utterances": int(sum(totals.values())),
        "high_utterances": int(totals["high"]),
        "mid_utterances": int(totals["mid"]),
        "low_utterances": int(totals["low"]),
    }


def plot_tiers(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 7))
    for offset, name, label in [(-width, "high", "High-tier sessions"), (0, "mid", "Mid-tier sessions"), (width, "low", "Low-tier sessions")]:
        ax.bar(x + offset, summary[f"{name}_percent"], width, label=label)
    ax.set_xticks(x, summary["tag"], rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Percentage of tags (%)", fontsize=13)
    ax.legend(fontsize=12)
    positions = {"high": -width, "mid": 0, "low": width}
    for index, row in summary.iterrows():
        maximum = max(row["high_percent"], row["mid_percent"], row["low_percent"])
        offset = 1.0
        for first, second in [("high", "mid"), ("high", "low"), ("mid", "low")]:
            match = tests[(tests.tag == row.tag) & (tests["first"] == first) & (tests["second"] == second)]
            if bool(match.iloc[0]["significant_at_0.01"]):
                x1, x2 = index + positions[first], index + positions[second]
                y0, y1 = maximum + offset, maximum + offset + 0.3
                ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0], color="black", linewidth=1.1)
                ax.text((x1 + x2) / 2, y1 - 0.24, "*", ha="center", va="bottom", fontsize=14)
                offset += 1.5
    fig.tight_layout()
    fig.savefig(OUTPUT / "tag_ratio_pairwise_kai.png", dpi=300)
    plt.close(fig)


def partial_spearman(frame: pd.DataFrame, outcome: str, predictor: str, control: str) -> tuple[float, float]:
    ranked = frame[[outcome, predictor, control]].rank(method="average")
    design = np.column_stack([np.ones(len(ranked)), ranked[control].to_numpy()])
    residual_outcome = ranked[outcome].to_numpy() - design @ np.linalg.lstsq(design, ranked[outcome], rcond=None)[0]
    residual_predictor = ranked[predictor].to_numpy() - design @ np.linalg.lstsq(design, ranked[predictor], rcond=None)[0]
    return stats.spearmanr(residual_outcome, residual_predictor)


def distress_analysis() -> tuple[pd.DataFrame, dict]:
    sessions = pd.read_csv(DATA / "session_level_analysis.csv")
    rows = []
    for tag in TAGS:
        column = f"rate_{tag}"
        rho, p = stats.spearmanr(sessions[column], sessions["delta_distress"], nan_policy="omit")
        partial_rho, partial_p = partial_spearman(sessions, "delta_distress", column, "dist_early")
        rows.append(
            {
                "tag": DISPLAY.get(tag, tag),
                "rho": rho,
                "p": p,
                "partial_rho": partial_rho,
                "partial_p": partial_p,
                "n": len(sessions),
            }
        )
    result = pd.DataFrame(rows)
    # The original analysis corrected a prespecified family of 20 indicators.
    result["p_adjusted"] = np.minimum(result["p"] * 20, 1.0)
    result["partial_p_adjusted"] = np.minimum(result["partial_p"] * 20, 1.0)
    result.to_csv(OUTPUT / "distress_correlations.csv", index=False)

    rho, p = stats.spearmanr(sessions["score"], sessions["delta_distress"])
    score_result: dict[str, object] = {"rho": rho, "p": p, "n": len(sessions), "tiers": {}}
    for name in ("high", "mid", "low"):
        values = sessions.loc[sessions["tier"] == name, "delta_distress"]
        score_result["tiers"][name] = {
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "n": len(values),
        }
    write_json("score_distress.json", score_result)
    return result, score_result


def prepare_turns() -> pd.DataFrame:
    turns = pd.read_csv(DATA / "turn_level_analysis.csv")
    turns["tag"] = turns["tag"].map(normalize_tag)
    turns = turns.sort_values(["session_id", "idx"]).copy()
    turns["n_turns"] = turns.groupby("session_id")["idx"].transform("count")
    turns["next_is_affirmation"] = turns.groupby("session_id")["tag"].shift(-1).eq("Affirmation").astype(int)
    return turns


def temporal_analysis() -> dict:
    turns = prepare_turns()

    def summarize(group: pd.DataFrame) -> pd.Series:
        half = len(group) // 2
        early, late = group.iloc[:half], group.iloc[half:]
        return pd.Series(
            {
                "early_affirmation": early["tag"].eq("Affirmation").mean(),
                "late_affirmation": late["tag"].eq("Affirmation").mean(),
                "early_distress": early["distress_level"].mean(),
                "delta_distress": late["distress_level"].mean() - early["distress_level"].mean(),
                "score": group["score"].iloc[0],
            }
        )

    halves = turns.groupby("session_id", sort=False).apply(summarize, include_groups=False).reset_index()
    early_delta = stats.spearmanr(halves["early_affirmation"], halves["delta_distress"])
    early_score = stats.spearmanr(halves["early_affirmation"], halves["score"])
    reverse = stats.spearmanr(halves["early_distress"], halves["late_affirmation"])

    model_data = turns.dropna(subset=["distress_level", "session_phase", "counselor_id"]).copy()
    glmm = BinomialBayesMixedGLM.from_formula(
        "next_is_affirmation ~ distress_level + C(session_phase) + n_turns",
        {"counselor_id": "0 + C(counselor_id)"},
        model_data,
    ).fit_map()
    glmm_index = list(glmm.model.exog_names).index("distress_level")
    glmm_coef = float(glmm.params[glmm_index])
    glmm_se = float(glmm.fe_sd[glmm_index])

    gee = GEE.from_formula(
        "next_is_affirmation ~ distress_level + C(session_phase) + n_turns",
        groups="session_id",
        data=model_data,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit()
    result = {
        "n_turns": len(model_data),
        "sessions": int(model_data["session_id"].nunique()),
        "counselors": int(model_data["counselor_id"].nunique()),
        "early_affirmation_delta_rho": early_delta.statistic,
        "early_affirmation_delta_p": early_delta.pvalue,
        "early_affirmation_score_rho": early_score.statistic,
        "early_affirmation_score_p": early_score.pvalue,
        "early_distress_late_affirmation_rho": reverse.statistic,
        "early_distress_late_affirmation_p": reverse.pvalue,
        "mixed_logit_or": math.exp(glmm_coef),
        "mixed_logit_ci_low": math.exp(glmm_coef - 1.96 * glmm_se),
        "mixed_logit_ci_high": math.exp(glmm_coef + 1.96 * glmm_se),
        "gee_or": math.exp(float(gee.params["distress_level"])),
        "gee_p": float(gee.pvalues["distress_level"]),
    }
    write_json("temporal_robustness.json", result)
    return result


MODEL_TAGS = [
    "rate_Affirmation",
    "rate_Reflection",
    "rate_OpenQuestion",
    "rate_ClosedQuestion",
    "rate_Suggest",
    "rate_Backchannel",
    "rate_Paraphrase",
    "rate_Other",
]


def fit_mixed(formula: str, frame: pd.DataFrame):
    return smf.mixedlm(formula, data=frame, groups=frame["counselor_id"]).fit(method="powell", disp=False)


def counselor_analysis() -> dict:
    sessions = pd.read_csv(DATA / "session_level_analysis.csv")
    counselors = pd.read_csv(DATA / "turn_level_analysis.csv", usecols=["session_id", "counselor_id"]).drop_duplicates("session_id")
    frame = sessions.merge(counselors, on="session_id", how="inner").dropna(
        subset=["delta_distress", "rate_Affirmation", "counselor_id"]
    )

    def null_icc(outcome: str) -> float:
        model = MixedLM(frame[outcome], np.ones(len(frame)), groups=frame["counselor_id"]).fit(method="powell", disp=False)
        random_variance = float(model.cov_re.iloc[0, 0])
        return random_variance / (random_variance + float(model.scale))

    fixed = " + ".join(MODEL_TAGS)
    random_delta = fit_mixed(f"delta_distress ~ {fixed} + dist_early", frame)
    random_score = fit_mixed(f"score ~ {fixed}", frame)

    frame["rate_Affirmation_between"] = frame.groupby("counselor_id")["rate_Affirmation"].transform("mean")
    frame["rate_Affirmation_within"] = frame["rate_Affirmation"] - frame["rate_Affirmation_between"]
    other = " + ".join(column for column in MODEL_TAGS if column != "rate_Affirmation")
    mundlak_delta = fit_mixed(
        f"delta_distress ~ rate_Affirmation_within + rate_Affirmation_between + {other} + dist_early",
        frame,
    )
    mundlak_score = fit_mixed(
        f"score ~ rate_Affirmation_within + rate_Affirmation_between + {other}",
        frame,
    )
    result = {
        "sessions": len(frame),
        "counselors": int(frame["counselor_id"].nunique()),
        "delta_icc": null_icc("delta_distress"),
        "score_icc": null_icc("score"),
        "delta_random_intercept_affirmation": float(random_delta.params["rate_Affirmation"]),
        "score_random_intercept_affirmation": float(random_score.params["rate_Affirmation"]),
        "delta_within_affirmation": float(mundlak_delta.params["rate_Affirmation_within"]),
        "score_within_affirmation": float(mundlak_score.params["rate_Affirmation_within"]),
        "delta_between_affirmation": float(mundlak_delta.params["rate_Affirmation_between"]),
        "delta_between_se": float(mundlak_delta.bse["rate_Affirmation_between"]),
        "delta_between_p": float(mundlak_delta.pvalues["rate_Affirmation_between"]),
        "score_between_affirmation": float(mundlak_score.params["rate_Affirmation_between"]),
        "score_between_se": float(mundlak_score.bse["rate_Affirmation_between"]),
        "score_between_p": float(mundlak_score.pvalues["rate_Affirmation_between"]),
    }
    write_json("counselor_effects.json", result)
    return result


def load_esconv() -> pd.DataFrame:
    documents = json.loads((DATA / "ESConv.json").read_text(encoding="utf-8"))
    strategies = [
        "Affirmation and Reassurance",
        "Reflection of feelings",
        "Question",
        "Providing Suggestions",
        "Others",
    ]
    rows = []
    for index, document in enumerate(documents):
        survey = document["survey_score"]["seeker"]
        if "final_emotion_intensity" not in survey:
            continue
        supporter_turns = [turn for turn in document["dialog"] if turn["speaker"] == "supporter"]
        if not supporter_turns:
            continue
        counts = Counter(
            turn.get("annotation", {}).get("strategy")
            for turn in supporter_turns
            if isinstance(turn.get("annotation", {}), dict)
        )
        row = {
            "conv_id": index,
            "delta_emotion": int(survey["final_emotion_intensity"]) - int(survey["initial_emotion_intensity"]),
        }
        for strategy in strategies:
            row["rate_" + strategy.replace(" ", "_")] = counts[strategy] / len(supporter_turns)
        rows.append(row)
    return pd.DataFrame(rows)


def transfer_analysis() -> dict:
    sessions = pd.read_csv(DATA / "session_level_analysis.csv")
    sessions["target"] = sessions["tier"].map({"low": 0, "high": 1})
    sessions["rate_Question"] = sessions["rate_OpenQuestion"] + sessions["rate_ClosedQuestion"]
    training = sessions.dropna(subset=["target"]).copy()
    kokoro_features = ["rate_Affirmation", "rate_Reflection", "rate_Question", "rate_Suggest", "rate_Other"]
    esconv_features = [
        "rate_Affirmation_and_Reassurance",
        "rate_Reflection_of_feelings",
        "rate_Question",
        "rate_Providing_Suggestions",
        "rate_Others",
    ]
    names = ["Affirmation", "Reflection", "Question", "Suggest", "Other"]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(training[kokoro_features].to_numpy())
    train_y = training["target"].astype(int).to_numpy()
    search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        refit=True,
    )
    search.fit(train_x, train_y)
    model = search.best_estimator_.fit(train_x, train_y)
    esconv = load_esconv()
    probabilities = model.predict_proba(scaler.transform(esconv[esconv_features].to_numpy()))[:, 1]
    correlation = stats.spearmanr(probabilities, esconv["delta_emotion"])
    coefficient_rows = [
        {"feature": name, "coefficient": float(value)}
        for name, value in zip(names, model.coef_[0])
    ]
    pd.DataFrame(coefficient_rows).sort_values("coefficient", key=abs, ascending=False).to_csv(
        OUTPUT / "transfer_coefficients.csv", index=False
    )
    result = {
        "training_sessions": len(training),
        "esconv_sessions": len(esconv),
        "best_c": float(search.best_params_["C"]),
        "rho": correlation.statistic,
        "p": correlation.pvalue,
        "coefficients": {row["feature"]: row["coefficient"] for row in coefficient_rows},
    }
    write_json("transfer.json", result)
    return result


def close(actual: float, expected: float, digits: int = 3) -> bool:
    return round(float(actual), digits) == round(float(expected), digits)


def check_results(results: dict[str, object]) -> None:
    failures: list[str] = []

    def exact(path: str, actual: object, expected: object) -> None:
        if actual != expected:
            failures.append(f"{path}: got {actual!r}, expected {expected!r}")

    def rounded(path: str, actual: float, expected: float, digits: int = 3) -> None:
        if not close(actual, expected, digits):
            failures.append(f"{path}: got {actual:.6g}, expected {expected:.{digits}f}")

    if "agreement" in results:
        actual = results["agreement"]
        expected = EXPECTED["agreement"]
        exact("agreement.n", actual["n"], expected["n"])
        exact(
            "agreement.distress_n",
            actual["distress_n"],
            expected["distress_n"],
        )
        for key in expected.keys() - {"n", "distress_n"}:
            rounded(f"agreement.{key}", actual[key], expected[key])
    if "tier" in results:
        for key, value in EXPECTED["corpus"].items():
            exact(f"corpus.{key}", results["tier"][key], value)
    if "distress" in results:
        score = results["distress"][1]
        expected = EXPECTED["score_distress"]
        rounded("score_distress.rho", score["rho"], expected["rho"])
        exact("score_distress.n", score["n"], expected["n"])
        for tier_name in ("high", "mid", "low"):
            for metric in ("mean", "sd"):
                rounded(
                    f"score_distress.{tier_name}_{metric}",
                    score["tiers"][tier_name][metric],
                    expected[f"{tier_name}_{metric}"],
                )
            exact(
                f"score_distress.{tier_name}_n",
                score["tiers"][tier_name]["n"],
                expected[f"{tier_name}_n"],
            )
    if "temporal" in results:
        for key, value in EXPECTED["temporal"].items():
            rounded(f"temporal.{key}", results["temporal"][key], value)
    if "counselor" in results:
        for key, value in EXPECTED["counselor"].items():
            rounded(f"counselor.{key}", results["counselor"][key], value)
    if "transfer" in results:
        transfer = results["transfer"]
        rounded("transfer.rho", transfer["rho"], EXPECTED["transfer"]["rho"])
        rounded("transfer.p", transfer["p"], EXPECTED["transfer"]["p"])
        for key in ("Affirmation", "Question", "Suggest", "Other", "Reflection"):
            rounded(f"transfer.{key}", transfer["coefficients"][key], EXPECTED["transfer"][key])

    if failures:
        raise AssertionError("Paper-value checks failed:\n- " + "\n- ".join(failures))
    print("All generated values match the paper at the reported precision.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis",
        choices=["all", "agreement", "tier", "distress", "temporal", "counselor", "transfer"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--check", action="store_true", help="compare against rounded values printed in the paper")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("[data] rebuilding analysis tables from annotated_kokorochat")
    turns, sessions = build_analysis_tables(
        ANNOTATED_KOKOROCHAT,
        DATA / "turn_level_analysis.csv",
        DATA / "session_level_analysis.csv",
    )
    print(
        f"[data] built {len(turns):,} turn-level rows and "
        f"{len(sessions):,} session-level rows"
    )

    functions = {
        "agreement": agreement,
        "tier": tier_analysis,
        "distress": distress_analysis,
        "temporal": temporal_analysis,
        "counselor": counselor_analysis,
        "transfer": transfer_analysis,
    }
    selected = list(functions) if args.analysis == "all" else [args.analysis]
    results = {}
    for name in selected:
        print(f"[{name}] running")
        results[name] = functions[name]()
        print(f"[{name}] done")
    if args.check:
        check_results(results)


if __name__ == "__main__":
    main()
