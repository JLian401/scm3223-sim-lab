from simulations.forecasting.data import (
    prepare_simulation_data,
    get_series,
)

from simulations.forecasting.engine import (
    generate_forecast,
)

from simulations.forecasting.hierarchy import (
    historical_state_shares,
    recent_state_shares,
    top_down_forecast,
    combine_bottom_up_forecasts,
    evaluate_hierarchical_forecast,
    create_hierarchy_metrics_table,
    compare_hierarchical_methods,
)


# =========================================================
# Load data
# =========================================================

sim_data = prepare_simulation_data(
    training_window="all"
)

train_df = sim_data[
    "train_full"
]

validation_df = sim_data[
    "validation"
]


# =========================================================
# 1. Inspect allocation shares
# =========================================================

print("\n")
print("=" * 70)
print("STATE ALLOCATION SHARES")
print("=" * 70)

historical_shares = (
    historical_state_shares(
        train_df
    )
)

recent_shares = (
    recent_state_shares(
        train_df,
        months=3,
    )
)

print(
    "\nHistorical shares:"
)

for state, share in (
    historical_shares.items()
):

    print(
        f"{state}: {share:.4f}"
    )


print(
    "\nRecent 3-month shares:"
)

for state, share in (
    recent_shares.items()
):

    print(
        f"{state}: {share:.4f}"
    )


# =========================================================
# 2. Generate aggregate validation forecast
#
# Use ES alpha=.50 as a simple example.
# =========================================================

aggregate_train = get_series(
    train_df,
    region="Aggregate",
)

aggregate_forecast = (
    generate_forecast(
        historical_data=aggregate_train,
        method="Exponential Smoothing",
        horizon=6,
        alpha=0.5,
    )
)


# =========================================================
# 3. Top-down: recent shares
# =========================================================

top_down_recent = (
    top_down_forecast(
        aggregate_forecast=
        aggregate_forecast,
        historical_df=train_df,
        allocation_rule=
        "recent_3_months",
    )
)


print("\n")
print("=" * 70)
print("TOP-DOWN FORECAST — RECENT 3-MONTH SHARES")
print("=" * 70)

print(
    top_down_recent[
        "forecasts"
    ].round(0)
)


top_down_metrics = (
    evaluate_hierarchical_forecast(
        actual_df=validation_df,
        forecast_df=top_down_recent[
            "forecasts"
        ],
    )
)


print("\nTop-down metrics:")

print(
    create_hierarchy_metrics_table(
        top_down_metrics
    ).round(2)
)

# =========================================================
# Top-down using full historical shares
# =========================================================

top_down_historical = (
    top_down_forecast(
        aggregate_forecast=aggregate_forecast,
        historical_df=train_df,
        allocation_rule="historical",
    )
)

historical_top_down_metrics = (
    evaluate_hierarchical_forecast(
        actual_df=validation_df,
        forecast_df=top_down_historical[
            "forecasts"
        ],
    )
)

print("\n")
print("=" * 70)
print("TOP-DOWN: RECENT VS HISTORICAL ALLOCATION")
print("=" * 70)

recent_table = (
    create_hierarchy_metrics_table(
        top_down_metrics
    )[
        ["Region", "MAPE"]
    ]
    .rename(
        columns={
            "MAPE": "Recent 3-Month MAPE"
        }
    )
)

historical_table = (
    create_hierarchy_metrics_table(
        historical_top_down_metrics
    )[
        ["Region", "MAPE"]
    ]
    .rename(
        columns={
            "MAPE": "Historical Share MAPE"
        }
    )
)

allocation_comparison = (
    recent_table.merge(
        historical_table,
        on="Region",
    )
)

allocation_comparison[
    "Difference"
] = (
    allocation_comparison[
        "Historical Share MAPE"
    ]
    - allocation_comparison[
        "Recent 3-Month MAPE"
    ]
)

print(
    allocation_comparison.round(2)
)


# =========================================================
# 4. Bottom-up
#
# For testing, use ES alpha=.50 for each state.
# Later students can choose different methods/parameters.
# =========================================================

ny_train = get_series(
    train_df,
    region="NY",
)

co_train = get_series(
    train_df,
    region="CO",
)

tx_train = get_series(
    train_df,
    region="TX",
)


ny_forecast = (
    generate_forecast(
        historical_data=ny_train,
        method="Exponential Smoothing",
        horizon=6,
        alpha=0.5,
    )
)

co_forecast = (
    generate_forecast(
        historical_data=co_train,
        method="Exponential Smoothing",
        horizon=6,
        alpha=0.5,
    )
)

tx_forecast = (
    generate_forecast(
        historical_data=tx_train,
        method="Exponential Smoothing",
        horizon=6,
        alpha=0.5,
    )
)


bottom_up_forecast = (
    combine_bottom_up_forecasts(
        ny_forecast=ny_forecast,
        co_forecast=co_forecast,
        tx_forecast=tx_forecast,
    )
)


print("\n")
print("=" * 70)
print("BOTTOM-UP FORECAST")
print("=" * 70)

print(
    bottom_up_forecast.round(0)
)


bottom_up_metrics = (
    evaluate_hierarchical_forecast(
        actual_df=validation_df,
        forecast_df=bottom_up_forecast,
    )
)


print("\nBottom-up metrics:")

print(
    create_hierarchy_metrics_table(
        bottom_up_metrics
    ).round(2)
)


# =========================================================
# 5. Compare
# =========================================================

print("\n")
print("=" * 70)
print("TOP-DOWN VS BOTTOM-UP")
print("=" * 70)

comparison = (
    compare_hierarchical_methods(
        top_down_metrics=
        top_down_metrics,
        bottom_up_metrics=
        bottom_up_metrics,
    )
)

print(
    comparison.round(2)
)