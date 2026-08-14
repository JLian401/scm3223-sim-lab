from simulations.forecasting.engine import (
    run_validation,
    run_final_forecast,
    evaluate_final_forecast,
)


# ---------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("VALIDATION TEST")
print("=" * 70)

validation = run_validation(
    method="Exponential Smoothing",
    region="Aggregate",
    alpha=0.5,
)

print("\nMethod:")
print(validation["method"])

print("\nParameters:")
print(validation["parameters"])

print("\nForecasts:")
print(validation["forecast"])

print("\nActual:")
print(validation["actual"])

print("\nMetrics:")
for name, value in validation["metrics"].items():
    print(f"{name}: {value:.2f}")

print("\nDetailed Validation Table:")
print(validation["error_table"])


# ---------------------------------------------------------
# 2. Locked final forecast
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL FORECAST — BEFORE REVEAL")
print("=" * 70)

final_forecast = run_final_forecast(
    method="Exponential Smoothing",
    region="Aggregate",
    alpha=0.5,
)

print("\nHistory used:")
print(
    final_forecast["history_start"],
    "to",
    final_forecast["history_end"],
)

print("\nForecast Dates:")
print(
    final_forecast[
        "forecast_dates"
    ].dt.strftime("%b %Y").tolist()
)

print("\nForecast:")
print(final_forecast["forecast"])


# ---------------------------------------------------------
# Verify that actual values are NOT present yet
# ---------------------------------------------------------

print("\nDoes result contain actual data?")
print(
    "actual" in final_forecast
)

print("\nDoes result contain metrics?")
print(
    "metrics" in final_forecast
)


# ---------------------------------------------------------
# 3. Reveal test results
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL FORECAST — AFTER REVEAL")
print("=" * 70)

revealed = evaluate_final_forecast(
    final_forecast
)

print("\nForecast:")
print(revealed["forecast"])

print("\nActual:")
print(revealed["actual"])

print("\nFinal Metrics:")
for name, value in revealed["metrics"].items():
    print(f"{name}: {value:.2f}")

print("\nDetailed Final Table:")
print(revealed["error_table"])