import pandas as pd

from abcd_substance_matching.data import (
    load_covariates,
    make_full_covariates_dataset,
    make_full_dataset,
    make_polysubstance,
    process_substance,
    process_tlfb,
    subset_biochem,
    subset_midyear,
    subset_selfreport,
    make_never_users_dataset
)
from abcd_substance_matching.utils import load_yaml


def main():

    filepaths = load_yaml("./conf/filepaths.yaml")
    dynamic_vars = load_yaml(filepaths['dynamic_vars'])
    static_vars = load_yaml(filepaths['static_vars'])
    mappings = load_yaml(filepaths['mappings'])

    levels = pd.read_excel(filepaths['dictionary_levels_path'], sheet_name='levels')

    full_dataset = make_full_dataset(
        static_vars,
        dynamic_vars,
        mappings,
        filepaths['data_path']
    )

    covars = load_covariates(
        full_dataset,
        static_vars,
        dynamic_vars,
        mappings,
        levels
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

    cannabis_agg = process_substance(tlfb, selfreport, mappings, 'cannabis')
    polysubstance = make_polysubstance(tlfb, selfreport, mappings)

    full_df, early, late = make_full_covariates_dataset(
        covars,
        cannabis_agg,
        polysubstance,
        mappings
    )

    full_df.to_parquet(filepaths['full_df'])
    early.to_parquet(filepaths['early'])
    late.to_parquet(filepaths['late'])


    full_never, early_never, late_never = make_never_users_dataset(full_df, mappings)

    full_never.to_parquet(filepaths['full_never'])
    early_never.to_parquet(filepaths['early_never'])
    late_never.to_parquet(filepaths['late_never'])



if __name__ == "__main__":
    main()
