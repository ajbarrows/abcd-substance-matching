from datetime import timedelta
from pathlib import Path

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

# Module constant for commonly used index columns
INDEX_COLS = ['participant_id', 'session_id']


def map_sub_ses(
    df: pd.DataFrame,
    session_map: dict,
    sub_col: str = "participant_id",
    ses_col: str = "session_id"
) -> pd.DataFrame:
    """Map ABCD 6.0+ to NDAR subject IDs and event names."""
    return (
        df
        .rename(columns={
            sub_col: 'src_subject_id',
            ses_col: 'eventname'
        })
        .assign(src_subject_id=lambda x: x['src_subject_id']
                .str.replace('sub-', "NDAR_INV"))
        .assign(eventname=lambda x: x['eventname'].astype('category'))
        .assign(eventname=lambda x: x['eventname']
                .cat.rename_categories(session_map))
    )


def format_raw_tlfb(df: pd.DataFrame, session_map: dict) -> pd.DataFrame:
    """Format raw Timeline Follow-Back (TLFB) data.

    Args:
        df: Raw TLFB DataFrame.
        session_map: Dictionary mapping session IDs to event names.

    Returns:
        Formatted TLFB DataFrame with proper date types.
    """
    return (
        df
        .assign(
            dt_tlfb=lambda x: pd.to_datetime(x['dt_tlfb'], format='%Y-%m-%d'),
            dt_use=lambda x: pd.to_datetime(x['dt_use'], format='%Y-%m-%d'),
            session_id=lambda x: x['session_id'].cat.rename_categories(session_map)
        )
    )


def compute_reference_dob(date, age):
    """Calculate birth date by subtracting age from date.

    Handles fractional ages and leap years correctly.

    Args:
        date: The reference date.
        age: Age in years (can be fractional).

    Returns:
        The calculated birth date.
    """
    years = int(age)
    fraction = age - years

    birth_date = date - relativedelta(years=years)

    if fraction > 0:
        days = int(fraction * 365.25)
        birth_date = birth_date - timedelta(days=days)

    birth_date = birth_date.replace(tzinfo=None)
    return birth_date


def load_dob(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Load date of birth data from DataFrame.

    Args:
        df: DataFrame containing DOB-related columns.
        mappings: Dictionary with 'dob_columns' key.

    Returns:
        DataFrame with computed reference DOB.
    """
    return (
        df
        .filter(items=INDEX_COLS + mappings['dob_columns'])
        .dropna()
        .assign(reference_date=lambda x: pd.to_datetime(x['reference_date']))
        .assign(reference_dob=lambda x: x.apply(
            lambda row: compute_reference_dob(row['reference_date'], row['age']), axis=1)
        )
    )


def process_tlfb(
    tlfb: pd.DataFrame,
    full_dataset: pd.DataFrame,
    mappings: dict,
    days_in_year: float = 365.25
) -> pd.DataFrame:
    """Process Timeline Follow-Back data with DOB and age calculations.

    Args:
        tlfb: Raw TLFB DataFrame.
        full_dataset: Full dataset containing DOB information.
        mappings: Dictionary with session_map and other mappings.
        days_in_year: Days in a year for age calculation.

    Returns:
        Processed TLFB DataFrame with use_age and estimation_length.
    """
    return (
        tlfb
        .pipe(format_raw_tlfb, mappings['session_map'])
        .set_index(INDEX_COLS)
        .join(
            load_dob(full_dataset, mappings).set_index(INDEX_COLS)
        )
        .dropna()
        .assign(
            use_age=lambda x: (x['dt_use'] - x['reference_dob']).dt.days / days_in_year,
            estimation_length=lambda x: (x['dt_tlfb'] - x['dt_use']).dt.days,
        )
        .drop(columns=['reference_date'])
        .reset_index()
    )


def subset_selfreport(full_dataset: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Subset self-report substance use data.

    Args:
        full_dataset: Full dataset DataFrame.
        mappings: Dictionary with timepoints and tlfb_summary_vars.

    Returns:
        DataFrame with binary self-report indicators.
    """
    return (
        full_dataset
        .filter(items=INDEX_COLS + mappings['tlfb_summary_vars'])
        .query(f"session_id in {mappings['timepoints']}")
        .dropna(subset='sui_day_count')
        .drop(columns=['sui_day_count'])
        .set_index(INDEX_COLS)
        .astype('object')
        .replace({'No': 0, 'Yes': 1})
        .astype('float')
        .fillna(0)
    )


def subset_midyear(full_dataset: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Subset mid-year substance use data.

    Args:
        full_dataset: Full dataset DataFrame.
        mappings: Dictionary with midyear_su_vars and lagging_map.

    Returns:
        DataFrame with cannabis_midyear indicator.
    """
    return (
        full_dataset
        .set_index(INDEX_COLS)
        .filter(like='su_y_mysu')
        .filter(items=mappings['midyear_su_vars'])
        .dropna(how='all', axis=0)
        .assign(cannabis_midyear=lambda x: x.sum(axis=1, numeric_only=True))
        .filter(items=['cannabis_midyear'])
        .reset_index()
        .assign(
            session_id=lambda x: x['session_id'].replace(mappings['lagging_map'])
        )
        .set_index(INDEX_COLS)
    )


def subset_biochem(full_dataset: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Subset biochemical toxicology data.

    Args:
        full_dataset: Full dataset DataFrame.
        mappings: Dictionary with timepoints and toxicology variable lists.

    Returns:
        DataFrame with cannabis_test and nic_test indicators.
    """
    return (
        full_dataset
        .filter(
            items=INDEX_COLS
            + mappings['urine_tox_vars']
            + mappings['sal_tox_vars']
            + mappings['nic_tox_vars']
            + mappings['hair_tox_vars']
        )
        .query(f"session_id in {mappings['timepoints']}")
        .set_index(INDEX_COLS)
        .astype('float')
        .assign(
            cannabis_test=lambda x: (
                np.where(x.filter(regex='urine|saliva|thc') == 1, 1, 0).sum(axis=1) > 0
            ),
            nic_test=lambda x: (
                np.where(x.filter(regex='nic') >= 3, 1, 0).sum(axis=1) > 0
            )
        )
        .filter(like='test')
    )


def join_dob(
    dob: pd.DataFrame,
    tlfb: pd.DataFrame,
    days_in_year: float = 365.25
) -> pd.DataFrame:
    """Join DOB to TLFB dataframe and calculate use age.

    Args:
        dob: DataFrame with reference DOB.
        tlfb: Timeline Follow-Back DataFrame.
        days_in_year: Days in a year for age calculation.

    Returns:
        TLFB DataFrame with use_age and estimation length.
    """
    return (
        tlfb
        .merge(dob, how='left', on=INDEX_COLS)
        .dropna()
        .assign(
            use_age=lambda x: (x['dt_use'] - x['reference_dob']).dt.days / days_in_year,
            estimation_length=lambda x: (x['dt_tlfb'] - x['dt_use']).dt.days,
        )
    )


def join_use_groups(df: pd.DataFrame, initiation_groups: dict) -> pd.DataFrame:
    """Assign initiation groups based on initiation timepoint.

    Args:
        df: DataFrame with initiation_timepoint column.
        initiation_groups: Dictionary mapping timepoints to group names.

    Returns:
        DataFrame with initiation_group column.
    """
    return (
        df
        .assign(
            initiation_group=lambda x:
                x['initiation_timepoint'].replace(initiation_groups)
        )
        .assign(
            initiation_group=lambda x:
                x['initiation_group'].fillna('never').astype('category')
        )
    )


def join_use_consistency(overall_use: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative use across timepoints.

    Args:
        overall_use: DataFrame with use_days column.

    Returns:
        DataFrame with cumulative_use column.
    """
    consistency = (
        overall_use
        .groupby(INDEX_COLS).use_days.sum()
        .reset_index()
        .assign(
            cumulative_use=lambda x: x.groupby(['participant_id']).use_days.cumsum()
        )
        .drop(columns='use_days')
        .set_index(INDEX_COLS)
    )
    return (
        overall_use
        .set_index(INDEX_COLS)
        .join(consistency)
        .reset_index()
    )


def flatten_variables(variables: dict) -> dict:
    """Flatten nested variable dictionary.

    Args:
        variables: Nested dictionary of variable mappings.

    Returns:
        Flat dictionary of variable name mappings.
    """
    for key, value in variables.items():
        if isinstance(value, list):
            variables[key] = {k: v for k, v in zip(value, value)}
    return {k: str(v) for d in variables.values() for k, v in d.items()}


def load_and_join(
    variables: dict,
    data_path: str = '',
    key: list = None
) -> pd.DataFrame:
    """Load and join multiple parquet tables.

    Args:
        variables: Dictionary mapping table names to variable dictionaries.
        data_path: Path to data directory.
        key: Index columns to join on.

    Returns:
        Joined DataFrame with all variables.
    """
    if key is None:
        key = ['participant_id', 'session_id']

    tables = []

    for table_name, var_dict in variables.items():
        fpath = Path(data_path) / f"{table_name}.parquet"

        cols = list(var_dict.keys())
        df = (
            pd
            .read_parquet(fpath, columns=key + cols)
            .rename(columns=var_dict)
            .set_index(key)
        )
        tables.append(df)

    result = tables[0]
    for df in tables[1:]:
        result = result.join(df, how='outer')

    return result.reset_index()


def make_full_dataset(
    static_vars: dict,
    dynamic_vars: dict,
    mappings: dict,
    data_path: str,
) -> pd.DataFrame:
    """Create full dataset by joining static and dynamic variables.

    Args:
        static_vars: Dictionary of static (time-invariant) variables.
        dynamic_vars: Dictionary of dynamic (time-varying) variables.
        mappings: Dictionary with session_map and variable lists.
        data_path: Path to data directory.

    Returns:
        Full dataset DataFrame.
    """
    session_map = mappings['session_map']

    def ffill_several_vars(df, group_col, static_vars):
        for col in static_vars:
            df[col] = df.groupby(group_col)[col].ffill()
        return df

    def combine_longitudinal_values(df):
        cols_to_drop = []
        for col in df.columns:
            if col.endswith('__l'):
                base_col = col[:-3]
                if base_col in df.columns:
                    df[base_col] = df[base_col].combine_first(df[col])
                    cols_to_drop.append(col)
        return df.drop(columns=cols_to_drop)

    df = (
        load_and_join(dynamic_vars, data_path)
        .set_index(['participant_id'])
        .join(
            (
                load_and_join(static_vars, data_path, key=['participant_id'])
                .set_index(['participant_id'])
            ),
            how='outer'
        )
        .reset_index()
    )

    ffill_vars = mappings['demo_vars'] + mappings['prenatal_vars']

    return (
        df
        .query("session_id in @session_map.keys()")
        .assign(
            age=lambda x: x['age'].fillna(x['midyear_pi_age']),
            reference_date=lambda x: x['reference_date'].fillna(x['midyear_pi_date']),
            session_id=lambda x: x['session_id'].replace(session_map)
        )
        .drop(columns=['midyear_pi_age', 'midyear_pi_date'])
        .pipe(ffill_several_vars, 'participant_id', ffill_vars)
        .pipe(combine_longitudinal_values)
    )


def map_categorical(df: pd.DataFrame, levels: pd.DataFrame, variables: dict) -> pd.DataFrame:
    """Map categorical variable codes to labels.

    Args:
        df: DataFrame with categorical columns.
        levels: DataFrame with variable levels from ABCD dictionary.
        variables: Variable name mappings.

    Returns:
        DataFrame with labeled categorical variables.
    """
    edgecase = {
        'curious': 'Not at all curious',
        'soon': 'Definitely not',
        'friends': 'Definitely not',
        'sex': 'Male',
        'race_ethnicity': 'Other',
        'parent_married': 'Decline to answer'
    }

    cat_levels = (
        levels
        .replace(flatten_variables(variables))
        .assign(value=lambda x: x['value'].astype('int').astype('str'))
    )

    for var in df.select_dtypes('category'):
        var_categories = cat_levels[cat_levels['name'] == var]
        levels_dict = (
            var_categories
            .filter(items=['value', 'label'])
            .set_index('value')
            .to_dict()['label']
        )
        ordering = (
            var_categories
            .filter(items=['order_level', 'label'])
            .set_index('order_level')
            .sort_index()
        )

        df[var] = df[var].cat.rename_categories(levels_dict)

        for case in edgecase:
            if case in var:
                df[var] = df[var].fillna(edgecase[case])

        if var not in ['session_id', 'puberty']:
            df[var] = df[var].cat.reorder_categories(ordering['label'], ordered=True)

    return df


def aggregate_use(tlfb: pd.DataFrame, mappings: dict, substance: str = 'all') -> pd.DataFrame:
    """Aggregate substance use by timepoint.

    Args:
        tlfb: Timeline Follow-Back DataFrame.
        mappings: Dictionary with substance-related category mappings.
        substance: Substance to aggregate ('all' or specific substance name).

    Returns:
        Aggregated use DataFrame with use_days column.
    """
    if substance != 'all':
        substance_vars = {k: substance for k in mappings[f'{substance}_related']}

        tlfb_agg = (
            tlfb
            .assign(substance=lambda x: x['substance'].astype('str'))
            .replace(substance_vars)
            .loc[lambda x: x['substance'] == substance]
            .groupby(INDEX_COLS)
            .value_counts(['substance'])
            .reset_index()
            .rename(columns={'count': 'use_days'})
        )
    else:
        tlfb_agg = (
            tlfb
            .groupby(INDEX_COLS + ['substance'], observed=True)
            .count()
            .reset_index()
            .filter(items=INDEX_COLS + ['substance', 'dt_use'])
            .rename(columns={'dt_use': 'use_days'})
        )

    return tlfb_agg


def find_first_use(df: pd.DataFrame, by_substance: bool = False) -> pd.DataFrame:
    """Find the first use of a substance.

    Args:
        df: DataFrame with use_age column.
        by_substance: Whether to find first use per substance.

    Returns:
        DataFrame with first_use_age for each subject.
    """
    if by_substance:
        grouping = INDEX_COLS + ['substance']
    else:
        grouping = INDEX_COLS

    return (
        df
        .filter(items=grouping + ['use_age'])
        .groupby(grouping, observed=True)
        .min()
        .rename(columns={'use_age': 'first_use_age'})
        .reset_index()
    )


def find_avg_event_age(df: pd.DataFrame) -> pd.DataFrame:
    """Find median age at each event in the sample.

    Args:
        df: DataFrame with use_age column.

    Returns:
        DataFrame with median_event_age for each session_id.
    """
    return (
        df
        .groupby('session_id', observed=True)['use_age']
        .median()
        .reset_index()
        .rename(columns={'use_age': 'median_event_age'})
    )


def subset_substance(df: pd.DataFrame, mappings: dict, substance: str = 'all') -> pd.DataFrame:
    """Subset and aggregate data for a specific substance.

    Args:
        df: TLFB DataFrame.
        mappings: Dictionary with substance-related category mappings.
        substance: Substance to subset ('all' or specific substance name).

    Returns:
        Aggregated substance use DataFrame.
    """
    if substance != 'all':
        df = df[df['substance'].isin(mappings[f'{substance}_related'])]
        by_substance = False
    else:
        by_substance = True

    first_use = find_first_use(df, by_substance)
    avg_event_age = find_avg_event_age(df)

    return (
        df
        .pipe(aggregate_use, mappings, substance)
        .merge(avg_event_age, how='left', on='session_id')
        .merge(first_use, how='left', on=INDEX_COLS)
    )


def make_polysubstance(
    tlfb: pd.DataFrame,
    selfreport: pd.DataFrame,
    mappings: dict
) -> pd.DataFrame:
    """Create polysubstance use DataFrame.

    Args:
        tlfb: Timeline Follow-Back DataFrame.
        selfreport: Self-report DataFrame.
        mappings: Dictionary with substances list.

    Returns:
        DataFrame with cumulative use columns for each substance.
    """
    substances = mappings['substances']

    polysubstance = pd.DataFrame()
    for substance in substances:

        polysubstance = pd.concat([
            polysubstance,
            (
                process_substance(tlfb, selfreport, mappings, substance)
                .filter(items=['use_days'])
                .rename(columns={'use_days': f'cumulative_{substance}'})
            )
        ], axis=1)

    return polysubstance.fillna(0)


def zero_pad_aggregation(
    aggregated: pd.DataFrame,
    selfreport: pd.DataFrame,
    substance: str
) -> pd.DataFrame:
    """Zero-pad aggregated data using self-report indicator.

    Args:
        aggregated: Aggregated TLFB DataFrame.
        selfreport: Self-report DataFrame.
        substance: Substance name prefix for filtering.

    Returns:
        DataFrame with zero-padded use data and any_use indicator.
    """
    return (
        aggregated
        .set_index(INDEX_COLS)
        .drop(columns='substance')
        .join(
            selfreport
            .filter(like=substance)
            .sum(axis=1)
            .pipe(pd.DataFrame)
            .set_axis(['self_report_binary'], axis=1)
            .assign(self_report_binary=lambda x: np.where(x['self_report_binary'] > 0, 1, 0)),
            how='outer'
        )
        .reset_index()
        .assign(any_use=lambda x: (x['use_days'].fillna(0) > 0) | (x['self_report_binary'].fillna(0) > 0))
    )


def get_initiation_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    """Get first timepoint of substance use for each subject.

    Args:
        df: DataFrame with any_use column.

    Returns:
        DataFrame with initiation_timepoint for each subject.
    """
    return (
        df
        .reset_index()
        .query("any_use")
        .sort_values(INDEX_COLS)
        .drop_duplicates(['participant_id'], keep='first')
        .filter(INDEX_COLS)
        .reset_index(drop=True)
        .rename(columns={'session_id': 'initiation_timepoint'})
    )


def process_substance(
    tlfb: pd.DataFrame,
    selfreport: pd.DataFrame,
    mappings: dict,
    substance: str
) -> pd.DataFrame:
    """Process substance use data combining TLFB and self-report.

    Args:
        tlfb: Timeline Follow-Back DataFrame.
        selfreport: Self-report DataFrame.
        mappings: Dictionary with initiation_groups and substance mappings.
        substance: Substance name to process.

    Returns:
        Processed substance use DataFrame with initiation groups.
    """
    sub_recode = 'alc' if substance == 'alcohol' else substance
    subset = (
        subset_substance(tlfb, mappings, substance)
        .pipe(zero_pad_aggregation, selfreport, sub_recode)
    )
    initiation_timepoints = get_initiation_timepoints(subset)

    return (
        subset
        .merge(initiation_timepoints, on='participant_id', how='outer')
        .pipe(join_use_groups, mappings['initiation_groups'])
        .pipe(join_use_consistency)
        .set_index(INDEX_COLS)
    )


def load_column_group(
    df: pd.DataFrame,
    varkey: str,
    indicator_var: str,
    variables: dict,
    replace_val: int = 0
) -> pd.DataFrame:
    """Load a group of variables and handle missing data.

    Args:
        df: Input DataFrame.
        varkey: Key in variables dict for the variable group.
        indicator_var: Indicator variable for missing data handling.
        variables: Dictionary of variable mappings.
        replace_val: Value to use for missing data.

    Returns:
        DataFrame with the variable group.
    """
    result = (
        df
        .set_index(INDEX_COLS)
        .filter(items=list(variables[varkey].values()))
        .dropna(how='all')
        .copy()
    )

    try:
        cols_to_check = [col for col in result.columns if col != indicator_var]
        mask = (
            result[indicator_var].notnull() |
            result[cols_to_check].notnull().any(axis=1)
        )
    except KeyError:
        cols_to_check = [col for col in result.columns]
        mask = result[cols_to_check].notnull().any(axis=1)

    for col in cols_to_check:
        if isinstance(result[col].dtype, pd.CategoricalDtype):
            replace_val_str = str(replace_val)
            if replace_val_str not in result[col].cat.categories:
                result[col] = result[col].cat.add_categories([replace_val_str])
            col_mask = mask & result[col].isnull()
            result.loc[col_mask, col] = replace_val_str
        else:
            col_mask = mask & result[col].isnull()
            result.loc[col_mask, col] = replace_val

    try:
        return result.drop(columns=indicator_var)
    except KeyError:
        return result


def sum_cols(df: pd.DataFrame, startswith: str, name: str) -> pd.DataFrame:
    """Sum columns matching a prefix into a new column.

    Args:
        df: Input DataFrame.
        startswith: Column name prefix to match.
        name: Name for the new summed column.

    Returns:
        DataFrame with summed column and original columns removed.
    """
    cols = [c for c in df.columns if startswith in c]
    return (
        df
        .join(
            df
            .filter(items=cols)
            .astype('float')
            .dropna(how='all')
            .assign(**{name: lambda x: x.sum(axis=1)})
            .drop(columns=cols)
        )
        .drop(columns=cols)
    )


def load_cbcl(df: pd.DataFrame, keep_vars: list, variables: list) -> pd.DataFrame:
    """Load CBCL variables with derived scores.

    Args:
        df: Input DataFrame.
        keep_vars: List of CBCL variables to keep.
        variables: List of external CBCL variable names.

    Returns:
        DataFrame with CBCL scores.
    """
    def cbcl_external_nodrug(df, variables):
        return (
            df
            .filter(items=variables)
            .astype('float')
            .dropna(how='all')
            .assign(cbcl_external_nodrug=lambda x: x.sum(axis=1, skipna=False))
            .filter(items=['cbcl_external_nodrug'])
        )

    def cbcl_totalprobs(df):
        totalprobs = [
            'cbcl_anxious_depressed', 'cbcl_withdrawn', 'cbcl_somatic',
            'cbcl_social', 'cbcl_thought', 'cbcl_attention',
            'cbcl_rulebreaking', 'cbcl_aggressive'
        ]
        return (
            df
            .filter(items=totalprobs)
            .dropna(how='all')
            .assign(cbcl_totalprobs=lambda x: x.sum(axis=1, skipna=False))
            .drop(columns=totalprobs)
        )

    return (
        df
        .filter(like='cbcl')
        .join(cbcl_external_nodrug(df, variables))
        .join(cbcl_totalprobs(df))
        .filter(items=keep_vars)
    )


def combine_prenatal_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Combine before/after pregnancy exposure variables.

    Args:
        df: DataFrame with prenatal exposure columns.

    Returns:
        DataFrame with combined prenatal exposure.
    """
    return (
        df
        .filter(like='pregnant')
        .reset_index()
        .melt(id_vars='participant_id')
        .assign(
            split=lambda x: x['variable'].str.split('_'),
            drug=lambda x: x['split'].str[0],
            period=lambda x: x['split'].str[1]
        )
        .pivot(index=['participant_id', 'drug'], columns=['period'], values='value')
        .assign(prenatal=lambda x: (x['after'] == 'Yes') | (x['before'] == 'Yes'))
        .drop(columns=['before', 'after'])
        .reset_index()
        .assign(drug=lambda x: 'prenatal_' + x['drug'])
        .pivot(index='participant_id', columns='drug', values='prenatal')
    )


def process_covars(
    df: pd.DataFrame,
    mappings: dict,
    missing_categories: list = None,
    ignore_vars: list = None
) -> pd.DataFrame:
    """Process covariate data with recoding and cleaning.

    Args:
        df: DataFrame with covariates.
        mappings: Dictionary with category mappings.
        missing_categories: Categories to treat as missing.
        ignore_vars: Variables to skip reordering.

    Returns:
        Processed covariates DataFrame.
    """
    if missing_categories is None:
        missing_categories = ["Don't know", "Decline to answer"]
    if ignore_vars is None:
        ignore_vars = ['parent_highest_ed', 'household_income']

    cat_vars = df.select_dtypes('category')

    for var in cat_vars:
        for category in missing_categories:
            if category in df[var].cat.categories:
                df[var] = df[var].cat.remove_categories(category)

        if var not in ignore_vars:
            ordering = list(df[var].value_counts().index)
            df[var] = df[var].cat.reorder_categories(ordering, ordered=False)

    df = df.set_index('participant_id')

    return (
        df
        .drop(columns=list(df.filter(like='pregnant').columns))
        .join(combine_prenatal_exposure(df.query("session_id == 'baseline'")))
        .assign(
            parent_highest_ed=lambda x: pd.Categorical(
                x['parent_highest_ed'].map(mappings['education_mapping']),
                categories=mappings['education_categories'],
                ordered=True
            ),
            prenatal_other=lambda x: np.where(
                (x['prenatal_cocaine']) |
                (x['prenatal_opioids']) |
                (x['prenatal_oxycontin']),
                'Yes', 'No'
            ),
            race_ethnicity_6level=lambda x: pd.Categorical(
                x['race_ethnicity_6level'].map(mappings['race_mapping']),
                categories=mappings['race_categories']
            ),
            parent_married=lambda x: pd.Categorical(
                x['parent_married'].map(mappings['marital_status_mapping']),
                categories=mappings['marital_categories']
            )
        )
        .drop(columns=['prenatal_cocaine', 'prenatal_opioids', 'prenatal_oxycontin'])
        .reset_index()
    )


def pivot_wider(
    df: pd.DataFrame,
    match_timepoints: list,
    constant_vars: str = 'sex|income|ethnicity|parent|prenatal|age'
) -> pd.DataFrame:
    """Pivot DataFrame to wide format for matching.

    Args:
        df: Long-format DataFrame.
        match_timepoints: Timepoints to include.
        constant_vars: Regex pattern for time-invariant variables.

    Returns:
        Wide-format DataFrame.
    """
    constant_cols = list(df.filter(regex=constant_vars).columns)
    index = ['participant_id', 'initiation_group', 'initiation_timepoint']

    pivoted = (
        df
        .drop(columns=constant_cols)
        .loc[lambda x: x['session_id'].isin(match_timepoints)]
        .pivot(
            index=index,
            columns='session_id',
            values=[c for c in df.columns if c not in index + constant_cols + ['session_id']]
        )
    )
    pivoted.columns = [f'{var}_{event}' for var, event in pivoted.columns]

    return (
        pivoted
        .join(
            df
            .set_index('participant_id')
            .filter(items=constant_cols)
            .reset_index()
            .drop_duplicates('participant_id')
            .set_index('participant_id')
        )
    )


def subset_covariates(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Subset and process covariate data.

    Args:
        df: Full dataset DataFrame.
        mappings: Dictionary with timepoints and variable lists.

    Returns:
        DataFrame with covariates.
    """
    cbcl_final_vars = ['cbcl_internal', 'cbcl_external_nodrug', 'cbcl_attention']
    ffill_var_names = mappings['demo_vars'] + mappings['prenatal_vars']

    final_vars = mappings['demo_vars'] \
            + mappings['nihtb_vars'] \
            + mappings['puberty_vars'] \
            + mappings['bpm_vars'] \
            + mappings['prenatal_vars'] \
            + mappings['race_vars'] \
            + cbcl_final_vars \
            + ['puberty', 'sex_at_birth']

    cbcl = df.set_index(INDEX_COLS).pipe(load_cbcl, cbcl_final_vars, mappings['cbcl_external_nodrug'])
    puberty = (
        df
        .set_index(INDEX_COLS)
        .filter(like='puberty')
        .astype(float)
        .assign(puberty=lambda x: x.sum(axis=1))
        .assign(puberty=lambda x: x['puberty'].astype('category'))
        .drop(columns=['puberty_male', 'puberty_female'])
    )

    tmp = (
        df
        .set_index(INDEX_COLS)
        .drop(columns=['cbcl_attention', 'cbcl_internal'])
        .join(cbcl)
        .drop(columns=['puberty_male', 'puberty_female'])
        .join(puberty)
        .reset_index()
        .query(f"session_id in {mappings['timepoints']}")
    )

    all_subjects = tmp['participant_id'].unique()
    all_events = tmp['session_id'].unique()
    complete_index = pd.MultiIndex.from_product(
        [all_subjects, all_events],
        names=INDEX_COLS
    )

    return (
        tmp
        .drop(columns=ffill_var_names)
        .set_index(INDEX_COLS)
        .join(
            tmp
            .set_index(INDEX_COLS)
            .filter(items=ffill_var_names)
            .reindex(complete_index)
            .groupby('participant_id')
            .ffill()
            .reset_index()
            .set_index(INDEX_COLS)
        )
        .filter(final_vars)
        .reset_index()
    )


def load_covariates(
    full_dataset: pd.DataFrame,
    static_vars: dict,
    dynamic_vars: dict,
    mappings: dict,
    levels: pd.DataFrame
) -> pd.DataFrame:
    """Load and process all covariates.

    Args:
        full_dataset: Full dataset DataFrame.
        static_vars: Dictionary of static variable mappings.
        dynamic_vars: Dictionary of dynamic variable mappings.
        mappings: Dictionary with processing mappings.
        levels: DataFrame with categorical levels.

    Returns:
        Processed covariates DataFrame.
    """
    variables = static_vars | dynamic_vars

    return (
        subset_covariates(full_dataset, mappings)
        .pipe(map_categorical, levels, variables)
        .pipe(process_covars, mappings)
    )


def lifetime_use_threshold(
    df: pd.DataFrame,
    threshold: int,
    substance: str = 'cannabis'
) -> pd.DataFrame:
    """Filter to subjects meeting lifetime use threshold.

    Args:
        df: DataFrame with cumulative use columns.
        threshold: Minimum cumulative use days.
        substance: Substance to check threshold for.

    Returns:
        Filtered DataFrame.
    """
    threshold_met = (
        df
        .sort_values(['participant_id', 'session_id'])
        .drop_duplicates(['participant_id'], keep='last')
        .query(f"cumulative_{substance} >= {threshold}")
        ['participant_id'].to_list()
    )
    return df[
        df['participant_id'].isin(threshold_met) | (df['initiation_group'] == 'never')
    ]


def keep_single_tpt_vars(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Keep only single timepoint for specified variables.

    Args:
        df: Input DataFrame.
        group: Group name ('early' or 'late').

    Returns:
        DataFrame with filtered variables.
    """
    filter_str = 'bpm|cbcl|cumulative'

    match_tpt = {
        'early': 'year_2',
        'late': 'year_4'
    }

    filter_vars = list(df.filter(regex=filter_str).columns)

    single_tpt = (
            df
            .set_index(INDEX_COLS)
            .filter(items=filter_vars)
            .loc[lambda x: x.index.get_level_values('session_id') == match_tpt[group]]
    )
    return (
        df
        .set_index(INDEX_COLS)
        .drop(columns=filter_vars)
        .join(single_tpt)
        .reset_index()
    )


def exclude_incomplete_use_records(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude records with incomplete use data.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with complete use records only.

    Raises:
        ValueError: If no 'any_use' columns are found.
    """
    drop_cols = list(df.filter(like='any_use').columns)
    if not drop_cols:
        raise ValueError("No 'any_use' columns found - check upstream pivot")
    return (
        df
        .dropna(subset=drop_cols)
        .drop(columns=drop_cols)
    )


def process_missing_covars(
    df: pd.DataFrame,
    missing_threshold: float = 0.75
) -> pd.DataFrame:
    """Process missing covariate data.

    Args:
        df: Input DataFrame.
        missing_threshold: Minimum proportion of non-missing values required.

    Returns:
        DataFrame with missing data handled.
    """
    min_rows = int(len(df) * missing_threshold)

    return (
        df
        .dropna(how='all', axis=1)
        .dropna(thresh=min_rows, axis=1)
        .dropna()
    )


def make_full_covariates_dataset(
    covars: pd.DataFrame,
    tlfb_agg: pd.DataFrame,
    polysubstance: pd.DataFrame,
    mappings: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create full covariates dataset with early and late subsets.

    Args:
        covars: Covariates DataFrame.
        tlfb_agg: Aggregated TLFB DataFrame.
        polysubstance: Polysubstance use DataFrame.
        mappings: Dictionary with match_timepoints.

    Returns:
        Tuple of (full_df, early, late) DataFrames.
    """
    full_df = (
        covars
        .set_index(INDEX_COLS)
        .join(
            (
                tlfb_agg
                .filter(items=['any_use', 'initiation_timepoint', 'initiation_group'])
                .join(polysubstance)
            )
        )
    )

    def process_group(full_df, group):
        return (
            full_df
            .loc[lambda x: x['initiation_group'].isin(['never', group])]
            .reset_index()
            .pipe(lifetime_use_threshold, threshold=20)
            .pipe(keep_single_tpt_vars, group)
            .pipe(pivot_wider, mappings[f'{group}_match_timepoints'])
            .pipe(exclude_incomplete_use_records)
            .pipe(process_missing_covars)
            .reset_index()
        )

    early = process_group(full_df, 'early')
    late = process_group(full_df, 'late')

    return full_df, early, late