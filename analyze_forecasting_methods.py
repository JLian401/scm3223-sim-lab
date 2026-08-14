import pandas as pd

from simulations.forecasting.data import (
    prepare_simulation_data,
    get_series,
)

from simulations.forecasting.methods import (
    moving_average_forecast,
    weighted_moving_average_forecast,
    exponential_smoothing_forecast,
    arima_forecast,
    ARIMA_CHOICES,
)

from simulations.forecasting.metrics import calculate_metrics


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

TRAINING_WINDOWS = [12, 24, 36, "all"]

MOVING_AVERAGE_WINDOWS = [3, 6, 12]

WEIGHTED_MOVING_AVERAGE_WEIGHTS = {
    "WMA 3-month (0.5, 0.3, 0.2)": [0.5, 0.3, 0.2],
    "WMA 3-month (0.6, 0.3, 0.1)": [0.6, 0.3, 0.1],
    "WMA 6-month declining": [
        0.30,
        0.25,
        0.20,
        0.10,
        0.10,
        0.05,
    ],
}

EXPONENTIAL_SMOOTHING_ALPHAS = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
]


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def add_result(
    results,
    training_window,
    method,
    parameters,
    actual,
    forecast,
):
    """
    Calculate metrics and append one model result.
    """

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    results.append(
        {
            "Training Window": str(training_window),
            "Method": method,
            "Parameters": parameters,
            "MFE": metrics["MFE"],
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "MAPE": metrics["MAPE"],
            "CFE": metrics["CFE"],
            "Tracking Signal": metrics["Tracking Signal"],
        }
    )


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

results = []

for training_window in TRAINING_WINDOWS:

    sim_data = prepare_simulation_data(
        training_window=training_window
    )

    train_df = sim_data["train_selected"]
    validation_df = sim_data["validation"]

    train_series = get_series(
        train_df,
        region="Aggregate",
    )

    validation_actual = get_series(
        validation_df,
        region="Aggregate",
    ).to_numpy()

    horizon = len(validation_actual)

    print("\n" + "=" * 70)
    print(f"TRAINING WINDOW: {training_window}")
    print(
        f"Training period: "
        f"{train_df['Year-Month'].min().date()} "
        f"to "
        f"{train_df['Year-Month'].max().date()}"
    )
    print(
        f"Validation period: "
        f"{validation_df['Year-Month'].min().date()} "
        f"to "
        f"{validation_df['Year-Month'].max().date()}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Simple Moving Average
    # -----------------------------------------------------

    for window in MOVING_AVERAGE_WINDOWS:

        if window <= len(train_series):

            forecast = moving_average_forecast(
                train_series,
                window=window,
                horizon=horizon,
            )

            add_result(
                results,
                training_window,
                "Moving Average",
                f"{window} months",
                validation_actual,
                forecast,
            )

    # -----------------------------------------------------
    # Weighted Moving Average
    # -----------------------------------------------------

    for name, weights in WEIGHTED_MOVING_AVERAGE_WEIGHTS.items():

        if len(weights) <= len(train_series):

            forecast = weighted_moving_average_forecast(
                train_series,
                weights=weights,
                horizon=horizon,
            )

            add_result(
                results,
                training_window,
                "Weighted Moving Average",
                name,
                validation_actual,
                forecast,
            )

    # -----------------------------------------------------
    # Exponential Smoothing
    # -----------------------------------------------------

    for alpha in EXPONENTIAL_SMOOTHING_ALPHAS:

        forecast = exponential_smoothing_forecast(
            train_series,
            alpha=alpha,
            horizon=horizon,
        )

        add_result(
            results,
            training_window,
            "Exponential Smoothing",
            f"alpha={alpha}",
            validation_actual,
            forecast,
        )

    # -----------------------------------------------------
    # ARIMA
    # -----------------------------------------------------

    for name, order in ARIMA_CHOICES.items():

        try:

            forecast = arima_forecast(
                train_series,
                order=order,
                horizon=horizon,
            )

            add_result(
                results,
                training_window,
                "ARIMA",
                name,
                validation_actual,
                forecast,
            )

        except Exception as error:

            print(
                f"Could not estimate {name} "
                f"with training window {training_window}: "
                f"{error}"
            )


# ---------------------------------------------------------
# Convert results to DataFrame
# ---------------------------------------------------------

results_df = pd.DataFrame(results)


# Sort by MAPE
results_df = results_df.sort_values(
    by="MAPE",
    ascending=True,
).reset_index(drop=True)


# ---------------------------------------------------------
# Best validation result by model family
# ---------------------------------------------------------

print("\n")
print("=" * 100)
print("BEST VALIDATION RESULT BY MODEL FAMILY")
print("=" * 100)

best_by_family = (
    results_df
    .sort_values("MAPE", ascending=True)
    .groupby("Method", as_index=False)
    .first()
)

best_by_family = best_by_family[
    [
        "Method",
        "Parameters",
        "MFE",
        "MAE",
        "MAPE",
        "Tracking Signal",
    ]
]

print(
    best_by_family.round(2).to_string(index=False)
)

# ---------------------------------------------------------
# Display overall ranking
# ---------------------------------------------------------

pd.set_option(
    "display.max_rows",
    None,
)

pd.set_option(
    "display.width",
    180,
)

pd.set_option(
    "display.max_columns",
    None,
)


print("\n")
print("=" * 100)
print("OVERALL VALIDATION RESULTS — SORTED BY MAPE")
print("=" * 100)

print(
    results_df[
        [
            "Training Window",
            "Method",
            "Parameters",
            "MFE",
            "MAE",
            "MAPE",
            "Tracking Signal",
        ]
    ].round(2)
)


# ---------------------------------------------------------
# Best model by training window
# ---------------------------------------------------------

print("\n")
print("=" * 100)
print("BEST MODEL WITHIN EACH TRAINING WINDOW")
print("=" * 100)

for training_window in TRAINING_WINDOWS:

    subset = results_df[
        results_df["Training Window"]
        == str(training_window)
    ]

    if len(subset) > 0:

        best = subset.iloc[0]

        print(
            f"\nTraining window: {training_window}"
        )

        print(
            f"Method: {best['Method']}"
        )

        print(
            f"Parameters: {best['Parameters']}"
        )

        print(
            f"MAPE: {best['MAPE']:.2f}%"
        )

        print(
            f"MAE: {best['MAE']:.2f}"
        )

        print(
            f"MFE: {best['MFE']:.2f}"
        )

        print(
            f"Tracking Signal: "
            f"{best['Tracking Signal']:.2f}"
        )


# ---------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------

output_file = (
    "forecasting_validation_results.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)

print("\n")
print(
    f"Full results saved to: {output_file}"
)


print("\n")
print("=" * 100)
print("ALL ARIMA RESULTS")
print("=" * 100)

arima_results = (
    results_df[
        results_df["Method"] == "ARIMA"
    ]
    .sort_values("MAPE", ascending=True)
)

print(
    arima_results[
        [
            "Training Window",
            "Parameters",
            "MFE",
            "MAE",
            "MAPE",
            "Tracking Signal",
        ]
    ]
    .round(2)
    .to_string(index=False)
)