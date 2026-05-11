import pandas as pd

from abcd_substance_matching.data import (
    make_full_dataset,
    make_polysubstance,
    process_substance,
    process_tlfb,
    subset_biochem,
    subset_midyear,
    subset_selfreport,
)
from abcd_substance_matching.utils import load_yaml


def load_substance_use(
    filepaths: dict,
    mappings: dict,
    dynamic_vars: dict,
    static_vars: dict,
) -> dict[str, pd.DataFrame]:
    """Load and aggregate substance use data across TLFB, self-report, midyear, and biochem.

    Returns a dict with keys:
        - 'selfreport': joined self-report / midyear / biochem indicators
        - 'tlfb': processed Timeline Follow-Back records
        - 'cannabis', 'alcohol', 'tobacco': per-substance aggregated DataFrames
        - 'polysubstance': cumulative use columns for all substances
    """
    full_dataset = make_full_dataset(
        static_vars,
        dynamic_vars,
        mappings,
        filepaths['data_path'],
    )

    selfreport = (
        subset_selfreport(full_dataset, mappings)
        .join(subset_midyear(full_dataset, mappings))
        .join(subset_biochem(full_dataset, mappings))
    )

    tlfb = (
        pd.read_parquet(filepaths['tlfb_path'])
        .pipe(process_tlfb, full_dataset, mappings)
    )

    substances = {
        substance: process_substance(tlfb, selfreport, mappings, substance)
        for substance in mappings['substances']
    }

    polysubstance = make_polysubstance(tlfb, selfreport, mappings)

    return {
        'selfreport': selfreport,
        'tlfb': tlfb,
        **substances,
        'polysubstance': polysubstance,
    }


def main():
    filepaths = load_yaml("./conf/filepaths.yaml")
    dynamic_vars = load_yaml(filepaths['dynamic_vars'])
    static_vars = load_yaml(filepaths['static_vars'])
    mappings = load_yaml(filepaths['mappings'])

    results = load_substance_use(filepaths, mappings, dynamic_vars, static_vars)

    results['selfreport'].reset_index().to_parquet(
        "./data/processed/selfreport.parquet"
    )
    results['tlfb'].to_parquet("./data/processed/tlfb_processed.parquet", index=False)
    results['polysubstance'].reset_index().to_parquet(
        "./data/processed/polysubstance.parquet"
    )
    for substance in mappings['substances']:
        results[substance].reset_index().to_parquet(
            f"./data/processed/{substance}.parquet"
        )


if __name__ == "__main__":
    main()
