# abcd-substance-matching

ABCD Study substance use data processing and propensity score matching utilities.

## Setup

### Configure data structure

Suggest using symbolic links to map ABCD 6.0 release data:

`6.0/rawdata/phenotype/` --> `data/raw/phenotype`

`6.0/concat/substance_use/tlfb/tlfb_raw.parquet` --> `data/raw/tlfb_raw.parquet`


### Make Python environment (uv)

```bash
uv sync
```

### Make R environment (conda)

```bash
conda env create -f environment.yml
```

## Usage

### Python data processing

```bash
uv run scripts/load_data.py
```

Creates datasets for matching under `./data/processed/{early/late}_matched.parquet`


### R matching

```bash
conda activate abcd-substance-matching-r
Rscript scripts/make_matches.R
```

Creates matched (weights appended) datasets under: `./data/processed/{early/late}_matched.parquet`




