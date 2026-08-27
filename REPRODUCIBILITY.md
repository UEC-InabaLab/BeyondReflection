# Reproducibility

The repository provides a deterministic CPU-only pipeline for the quantitative
results reported in the paper. No model training, LLM call, API key, or GPU is
required.

## End-to-end check

```bash
uv sync --frozen
uv run python scripts/validate_inputs.py
uv run python reproduce.py all --check
```

The first analysis step rebuilds `data/turn_level_analysis.csv` and
`data/session_level_analysis.csv` from the released JSON files. The `--check`
flag compares recomputed values with `expected/paper_results.json` at the
precision reported in the paper and exits with a nonzero status if any check
fails.

## Analysis groups

| Command | Main outputs |
|---|---|
| `agreement` | `annotation_agreement.json`, `tag_level_metrics.csv`, `distress_level_agreement.csv` |
| `tier` | `tier_usage_rates.csv`, `tier_pairwise_tests.csv`, `tag_ratio_pairwise_kai.png` |
| `distress` | `distress_correlations.csv`, `score_distress.json` |
| `temporal` | `temporal_robustness.json` |
| `counselor` | `counselor_effects.json` |
| `transfer` | `transfer.json`, `transfer_coefficients.csv` |

All outputs are written to `outputs/` and are excluded from version control.

## Inputs and derivation

- The score-tier distribution uses all 306,495 counselor utterances.
- The temporal and session-level analyses use a 229,743-row cross-turn table
  and a 6,589-row session table.
- `scripts/build_analysis_tables.py` derives both tables from
  `annotated_kokorochat/`. It pairs counselor responses with the preceding
  client annotation, estimates response delay from message timestamps, retains
  responses with estimated silence of at most five minutes, and computes the
  strategy rates and distress outcomes used by the analyses.
- `data/annotation_evaluation/strategy_labels.csv` contains 318 aligned
  counselor-strategy items, and `distress_labels.csv` contains 204 aligned
  client-distress items. The original annotation workbooks are not required.
- `data/ESConv.json` is used only by the cross-dataset transfer analysis.
- `expected/paper_results.json` contains rounded publication targets for
  verification; it is never used as an analysis input.

Package versions are pinned in `uv.lock`. The principal numerical dependencies
are NumPy 2.0.2, pandas 2.3.0, SciPy 1.13.1, scikit-learn 1.6.1, and
statsmodels 0.14.6.
