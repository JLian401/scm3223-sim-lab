from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "forecasting"
    / "all_sales_data.csv"
)


# ---------------------------------------------------------
# Fixed simulation dates
# ---------------------------------------------------------

COMMON_START = "2020-04-01"
TRAIN_END = "2024-06-01"

VALIDATION_START = "2024-07-01"
VALIDATION_END = "2024-12-01"

TEST_START = "2025-01-01"
TEST_END = "2025-06-01"


# ---------------------------------------------------------
# Core data loading
# ---------------------------------------------------------

def load_model_y_data(filepath=DEFAULT_DATA_PATH):
    """
    Load Tesla Model Y state registration data.

    The state registration counts are used as a proxy for demand.

    Expected columns
    ----------------
    Year-Month
    Sales_CO
    Sales_NM
    Sales_NY
    Sales_TX

    Returns
    -------
    pandas.DataFrame
        Monthly NY, CO, TX, and aggregate registration data.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find forecasting data file:\n{filepath}"
        )

    df = pd.read_csv(filepath)

    required_columns = [
        "Year-Month",
        "Sales_CO",
        "Sales_NY",
        "Sales_TX",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )

    # Convert month variable to datetime
    df["Year-Month"] = pd.to_datetime(
        df["Year-Month"],
        errors="coerce",
    )

    if df["Year-Month"].isna().any():
        raise ValueError(
            "Some Year-Month values could not be converted to dates."
        )

    # Rename state variables to simpler names
    df = df.rename(
        columns={
            "Sales_CO": "CO",
            "Sales_NY": "NY",
            "Sales_TX": "TX",
        }
    )

    # Keep only variables needed for Version 1
    df = df[
        [
            "Year-Month",
            "NY",
            "CO",
            "TX",
        ]
    ].copy()

    # Make sure registration counts are numeric
    for col in ["NY", "CO", "TX"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Sort chronologically
    df = df.sort_values("Year-Month").reset_index(drop=True)

    # Restrict to the common period used by the simulation
    # before checking for missing values.
    # Earlier months contain incomplete state coverage,
    # but they are outside the simulation period.
    df = df[
        (df["Year-Month"] >= COMMON_START)
        & (df["Year-Month"] <= TEST_END)
    ].copy().reset_index(drop=True)

    # Check for missing/non-numeric values only within
    # the period actually used by the simulation.
    if df[["NY", "CO", "TX"]].isna().any().any():
        raise ValueError(
            "Some NY, CO, or TX registration values are missing "
            "or non-numeric within the simulation period."
        )

    # Create total registration proxy for demand
    df[["NY", "CO", "TX"]] = df[["NY", "CO", "TX"]].astype(int)
    df["Aggregate"] = (
        df["NY"]
        + df["CO"]
        + df["TX"]
    )

    return df


# ---------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------

def split_forecasting_data(df):
    """
    Split the Model Y dataset into fixed training,
    validation, and final test periods.

    Training:
        Apr 2020 - Jun 2024

    Validation:
        Jul 2024 - Dec 2024

    Test:
        Jan 2025 - Jun 2025

    Returns
    -------
    dict
        Dictionary containing:
        full, train, validation, test
    """

    train = df[
        (df["Year-Month"] >= COMMON_START)
        & (df["Year-Month"] <= TRAIN_END)
    ].copy()

    validation = df[
        (df["Year-Month"] >= VALIDATION_START)
        & (df["Year-Month"] <= VALIDATION_END)
    ].copy()

    test = df[
        (df["Year-Month"] >= TEST_START)
        & (df["Year-Month"] <= TEST_END)
    ].copy()

    return {
        "full": df.reset_index(drop=True),
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


# ---------------------------------------------------------
# Student historical-window choice
# ---------------------------------------------------------

def select_training_window(train_df, months="all"):
    """
    Select how much historical training data the student
    wants to use.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Full training sample.

    months : int or str, default="all"
        Number of most recent months to retain.
        Examples:
            12
            24
            36
            "all"

    Returns
    -------
    pandas.DataFrame
        Selected historical training window.
    """

    if months == "all":
        return train_df.copy().reset_index(drop=True)

    if not isinstance(months, int):
        raise ValueError(
            "months must be an integer or 'all'."
        )

    if months < 1:
        raise ValueError(
            "months must be at least 1."
        )

    if months > len(train_df):
        raise ValueError(
            f"Requested {months} months, but only "
            f"{len(train_df)} training months are available."
        )

    return (
        train_df
        .tail(months)
        .copy()
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------

def get_series(df, region="Aggregate"):
    """
    Extract one forecasting series.

    Parameters
    ----------
    df : pandas.DataFrame

    region : str
        One of:
            Aggregate
            NY
            CO
            TX

    Returns
    -------
    pandas.Series
    """

    valid_regions = [
        "Aggregate",
        "NY",
        "CO",
        "TX",
    ]

    if region not in valid_regions:
        raise ValueError(
            f"region must be one of {valid_regions}"
        )

    return df[region].copy()


def get_dates(df):
    """
    Return the monthly date series.
    """

    return df["Year-Month"].copy()


def prepare_simulation_data(
    filepath=DEFAULT_DATA_PATH,
    training_window="all",
):
    """
    Convenience function used by the simulation.

    Loads data, creates the fixed train/validation/test split,
    and applies the student's selected historical window
    to the training period.

    Returns
    -------
    dict
    """

    df = load_model_y_data(filepath)

    split = split_forecasting_data(df)

    selected_train = select_training_window(
        split["train"],
        months=training_window,
    )

    return {
        "full": split["full"],
        "train_full": split["train"],
        "train_selected": selected_train,
        "validation": split["validation"],
        "test": split["test"],
    }