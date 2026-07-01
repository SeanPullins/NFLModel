# Data sources

Training data is built from public GitHub sources via:

```bash
python src/download_source_data.py
```

## Current sources

- `phcs971/nfl-draft-dataset`
  - Combine, draft, career AV, and NCAA career data from 1987-2024.
  - Used for draft slots, Career AV target, combine fields, and NCAA production features.
  - NCAA fields used as model inputs: college games, passing, rushing, receiving, tackles, sacks, interceptions, and fumbles.
  - NFL career fields are not used as model inputs because they are outcomes/leakage.

- `array-carpenter/nfl-draft-data`
  - Combine/pro-day measurements from 2007-2026.
  - Used only as a measurement overlay for combine/pro-day fields.

- `nflverse/nflverse-data`
  - Draft-pick history reference.
  - Not currently used for model outcomes.

## Generated local files

```text
data/draft_data.csv
data/combine_data_pfr_with_stats.csv
```

These are generated artifacts and do not need to be committed for the GitHub Action to run.
