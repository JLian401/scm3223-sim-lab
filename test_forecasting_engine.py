import pandas as pd

from simulations.forecasting.methods import (
    moving_average_forecast,
    weighted_moving_average_forecast,
    exponential_smoothing_forecast,
    arima_forecast,
)

from simulations.forecasting.metrics import (
    calculate_metrics,
    create_error_table,
)


# Simple example historical data
history = pd.Series([
    100,
    110,
    105,
    120,
    125,
    130,
    135,
    140,
    150,
    155,
])


print("\n3-Month Moving Average")
forecast_ma = moving_average_forecast(
    history,
    window=3,
    horizon=6,
)
print(forecast_ma)


print("\nWeighted Moving Average")
forecast_wma = weighted_moving_average_forecast(
    history,
    weights=[0.5, 0.3, 0.2],
    horizon=6,
)
print(forecast_wma)


print("\nExponential Smoothing")
forecast_es = exponential_smoothing_forecast(
    history,
    alpha=0.3,
    horizon=6,
)
print(forecast_es)


print("\nARIMA")
forecast_arima = arima_forecast(
    history,
    order=(1, 1, 0),
    horizon=6,
)
print(forecast_arima)


# Pretend these are the six future actual observations
actual = [
    160,
    165,
    170,
    168,
    175,
    180,
]


print("\nMoving Average Metrics")
metrics = calculate_metrics(
    actual,
    forecast_ma,
)

for name, value in metrics.items():
    print(f"{name}: {value:.2f}")


print("\nDetailed Error Table")
table = create_error_table(
    actual,
    forecast_ma,
    periods=[
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
    ],
)

print(table)