# Beyond Reflection

Official data and analysis code for:

> **Beyond Reflection: Affirmation as a Promising Behavioral Marker Associated
> with Quality in Text-Based Counseling**

This repository provides additional annotations for all 6,589 sessions in
[KokoroChat](https://github.com/UEC-InabaLab/KokoroChat), the annotation
instructions, and the code used for the paper's statistical analyses. The
analysis is deterministic and does not require an API key or GPU.


## Repository structure

```text
.
├── annotated_kokorochat/       # KokoroChat with the additional labels
├── data/
│   ├── ESConv.json             # Data used for cross-dataset transfer
│   └── annotation_evaluation/
│       ├── strategy_labels.csv # 318 aligned strategy-label items
│       └── distress_labels.csv # 204 aligned distress-label items
├── expected/
│   └── paper_results.json      # Values reported in the paper
├── prompts/                    # Japanese prompts and English translations
├── scripts/
│   ├── build_analysis_tables.py
│   └── validate_inputs.py
├── reproduce.py                # Main analysis entry point
├── pyproject.toml
└── uv.lock
```

`annotated_kokorochat/` contains the original KokoroChat records plus the
following additional fields:

- counselor turns: `tag` (one of 11 counselor-strategy labels);
- client turns: `distress_level` and `disclosure_depth` (integers from 0 to 3).

Annotation rationales generated while assigning these labels are not included.
The Japanese prompts used to obtain the labels and English translations are
available under `prompts/`.

## Setup

Python 3.11 or later is required. We recommend
[uv](https://docs.astral.sh/uv/) for an exact environment:

```bash
uv sync --frozen
```

Run the command from the repository root after cloning or downloading the
public release.

## Reproducing the results

Run every analysis and verify the results against the values reported in the
paper:

```bash
uv run python reproduce.py all --check
```

Generated tables, JSON summaries, and the figure are written to `outputs/`.
That directory is intentionally ignored by Git because every file in it can be
regenerated.

Individual analysis groups can also be run:

```bash
uv run python reproduce.py agreement
uv run python reproduce.py tier
uv run python reproduce.py distress
uv run python reproduce.py temporal
uv run python reproduce.py counselor
uv run python reproduce.py transfer
```

To run fast structural checks on the released inputs:

```bash
uv run python scripts/validate_inputs.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the analysis-to-output mapping
and implementation notes.

## Data notes

- The repository contains 6,589 Japanese role-play counseling sessions from
  KokoroChat, including 306,495 counselor utterances.
- The score-tier analysis uses all counselor utterances.
- The remaining analyses use a derived cross-turn table containing 229,743
  rows. `scripts/build_analysis_tables.py` rebuilds this table and the
  6,589-row session table from `annotated_kokorochat/` at the start of each
  run; the generated CSV files are not source data.
- The adjusted p-values in the paper retain the prespecified Bonferroni family
  size of 20 behavioral indicators. The released output contains the 11
  counselor-strategy results reported in the paper.
- `data/ESConv.json` is used only for the cross-dataset transfer analysis.
- `data/annotation_evaluation/` contains only the aligned human and LLM labels
  used for the reported agreement scores; annotation workbooks and duplicated
  dialogue text are not included.

## Citation

If you use the additional annotations or analysis code, cite this paper:

```bibtex
@inproceedings{inaba-2026-beyond,
  title     = {Beyond Reflection: Affirmation as a Promising Behavioral Marker
               Associated with Quality in Text-Based Counseling},
  author    = {Inaba, Michimasa},
  booktitle = {Findings of the Association for Computational Linguistics:
               {EMNLP} 2026},
  year      = {2026}
}
```

Because this release contains KokoroChat records, also cite the original
dataset:

```bibtex
@inproceedings{qi-etal-2025-kokorochat,
  title     = {{K}okoro{C}hat: A {J}apanese Psychological Counseling Dialogue
               Dataset Collected via Role-Playing by Trained Counselors},
  author    = {Qi, Zhiyang and Kaneko, Takumasa and Takamizo, Keiko and
               Ukiyo, Mariko and Inaba, Michimasa},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  year      = {2025},
  pages     = {12424--12443},
  url       = {https://aclanthology.org/2025.acl-long.608/}
}
```


## License

The analysis code is released under the MIT License. The datasets have
different terms: KokoroChat-derived files are under CC BY-NC-ND 4.0 and ESConv
is under CC BY-NC 4.0. See [LICENSE](LICENSE) and
[DATA_LICENSE.md](DATA_LICENSE.md) before using or redistributing any part of
this repository.
