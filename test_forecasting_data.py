from simulations.forecasting.data import (
    load_model_y_data,
    split_forecasting_data,
    select_training_window,
    prepare_simulation_data,
)


df = load_model_y_data()

print("\nFULL DATA")
print(df.head())
print(df.tail())
print(f"Rows: {len(df)}")


split = split_forecasting_data(df)

print("\nTRAIN")
print(split["train"].head())
print(split["train"].tail())
print(f"Rows: {len(split['train'])}")


print("\nVALIDATION")
print(split["validation"])
print(f"Rows: {len(split['validation'])}")


print("\nTEST")
print(split["test"])
print(f"Rows: {len(split['test'])}")


print("\nLAST 24 MONTHS OF TRAINING")
train_24 = select_training_window(
    split["train"],
    months=24,
)

print(train_24.head())
print(train_24.tail())
print(f"Rows: {len(train_24)}")


print("\nPREPARED SIMULATION DATA")
sim_data = prepare_simulation_data(
    training_window=36
)

print(
    "Selected training rows:",
    len(sim_data["train_selected"])
)