import copy

import numpy as np
import pandas as pd
import streamlit as st

from simulations.forecasting.data import (
    prepare_simulation_data,
    select_training_window,
)

from simulations.forecasting.engine import (
    run_validation,
    run_final_forecast,
    evaluate_final_forecast,
    evaluate_judgmental_forecast,
    METHOD_MOVING_AVERAGE,
    METHOD_WEIGHTED_MOVING_AVERAGE,
    METHOD_EXPONENTIAL_SMOOTHING,
    METHOD_ARIMA,
)

from simulations.forecasting.methods import (
    ARIMA_CHOICES,
)

from simulations.forecasting.hierarchy import (
    top_down_forecast,
    combine_bottom_up_forecasts,
    evaluate_hierarchical_forecast,
    create_hierarchy_metrics_table,
)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

DEFAULT_SESSION_VALUES = {
    # Part 1
    "forecast_judgment_submitted": False,
    "forecast_judgment_values": None,
    "forecast_judgment_result": None,

    "forecast_model_results": [],

    "forecast_final_locked": False,
    "forecast_final_model": None,
    "forecast_final_result": None,
    "forecast_final_revealed": False,
    "forecast_final_evaluation": None,

    # Part 2
    "forecast_top_down_results": [],
    "forecast_bottom_up_results": [],
}


for key, default_value in DEFAULT_SESSION_VALUES.items():

    if key not in st.session_state:

        st.session_state[key] = copy.deepcopy(
            default_value
        )


# =========================================================
# LOAD DATA
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

test_df = sim_data[
    "test"
]


# =========================================================
# REUSABLE MODEL CONTROL
# =========================================================

def forecasting_model_controls(
    key_prefix,
    title=None,
):
    """
    Render forecasting-method controls.

    Returns
    -------
    dict
        {
            "method": ...,
            "parameters": ...,
            "description": ...,
            "can_run": ...
        }
    """

    if title is not None:

        st.subheader(
            title
        )

    method = st.selectbox(
        "Forecasting method",
        options=[
            METHOD_MOVING_AVERAGE,
            METHOD_WEIGHTED_MOVING_AVERAGE,
            METHOD_EXPONENTIAL_SMOOTHING,
            METHOD_ARIMA,
        ],
        key=f"{key_prefix}_method",
    )

    parameters = {}

    description = ""

    can_run = True


    # -----------------------------------------------------
    # MOVING AVERAGE
    # -----------------------------------------------------

    if method == METHOD_MOVING_AVERAGE:

        window = st.slider(
            "Number of months in moving average",
            min_value=2,
            max_value=12,
            value=3,
            step=1,
            key=f"{key_prefix}_ma_window",
        )

        parameters[
            "window"
        ] = window

        description = (
            f"{window}-month moving average"
        )

        st.caption(
            """
            Shorter windows react more quickly to recent demand changes.
            Longer windows produce smoother forecasts.
            """
        )


    # -----------------------------------------------------
    # WEIGHTED MOVING AVERAGE
    # -----------------------------------------------------

    elif (
        method
        == METHOD_WEIGHTED_MOVING_AVERAGE
    ):

        number_periods = st.selectbox(
            "Number of historical periods",
            options=[
                3,
                4,
                6,
            ],
            key=f"{key_prefix}_wma_periods",
        )

        st.caption(
            """
            Assign weights from most recent to oldest.
            The final weight is calculated automatically.
            """
        )

        default_weights = {
            3: [
                0.50,
                0.30,
            ],

            4: [
                0.40,
                0.30,
                0.20,
            ],

            6: [
                0.30,
                0.25,
                0.20,
                0.10,
                0.10,
            ],
        }

        entered_weights = []

        weight_columns = st.columns(
            number_periods
        )

        for i in range(
            number_periods - 1
        ):

            with weight_columns[i]:

                weight = st.number_input(
                    f"Weight {i + 1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weights[
                        number_periods
                    ][i],
                    step=0.05,
                    format="%.2f",
                    key=(
                        f"{key_prefix}_"
                        f"wma_{number_periods}_"
                        f"weight_{i}"
                    ),
                )

                entered_weights.append(
                    weight
                )

        final_weight = (
            1.0
            - sum(
                entered_weights
            )
        )

        with weight_columns[
            number_periods - 1
        ]:

            st.metric(
                f"Weight {number_periods}",
                f"{final_weight:.2f}",
            )

            st.caption(
                "Automatically calculated"
            )

        weights = (
            entered_weights
            + [final_weight]
        )

        if final_weight < 0:

            st.error(
                """
                The entered weights already sum to more than 1.00.
                Reduce one or more weights.
                """
            )

            can_run = False

        else:

            st.success(
                "Weights sum to 1.00."
            )

        parameters[
            "weights"
        ] = weights

        description = (
            f"{number_periods}-period WMA "
            f"({', '.join(f'{x:.2f}' for x in weights)})"
        )


    # -----------------------------------------------------
    # EXPONENTIAL SMOOTHING
    # -----------------------------------------------------

    elif (
        method
        == METHOD_EXPONENTIAL_SMOOTHING
    ):

        alpha = st.slider(
            "Smoothing constant (alpha)",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key=f"{key_prefix}_alpha",
        )

        parameters[
            "alpha"
        ] = alpha

        description = (
            f"alpha = {alpha:.2f}"
        )

        st.caption(
            """
            Larger alpha values give more weight to recent observations.
            Smaller alpha values give more weight to older information.
            """
        )


    # -----------------------------------------------------
    # ARIMA
    # -----------------------------------------------------

    elif method == METHOD_ARIMA:

        arima_name = st.selectbox(
            "ARIMA specification",
            options=list(
                ARIMA_CHOICES.keys()
            ),
            key=f"{key_prefix}_arima",
        )

        parameters[
            "arima_name"
        ] = arima_name

        description = (
            arima_name
        )

        st.caption(
            """
            ARIMA choices are intentionally constrained.
            The goal is to compare forecasting approaches rather
            than perform unrestricted ARIMA model selection.
            """
        )


    return {
        "method": method,
        "parameters": parameters,
        "description": description,
        "can_run": can_run,
    }


# =========================================================
# PAGE INTRODUCTION
# =========================================================

st.title(
    "📈 Demand Forecasting Lab"
)

st.write(
    """
    You are a demand planner forecasting Tesla Model Y demand across
    **New York, Colorado, and Texas**.

    Monthly vehicle registrations are used as a proxy for demand.
    Your goal is not simply to minimize forecast error, but to understand
    how forecasting methods, parameters, aggregation, and allocation
    decisions affect forecast performance.
    """
)


# =========================================================
# DATA PERIOD SUMMARY
# =========================================================

col1, col2, col3 = st.columns(
    3
)

col1.metric(
    "Historical Data",
    "Apr 2020–Jun 2024",
)

col2.metric(
    "Validation",
    "Jul–Dec 2024",
)

col3.metric(
    "Final Test",
    "Jan–Jun 2025",
)


# =========================================================
# LEARNING OBJECTIVES
# =========================================================

with st.expander(
    "Learning objectives"
):

    st.markdown(
        """
        After completing this simulation, you should be able to:

        - distinguish judgmental and quantitative forecasting;
        - evaluate forecasts using MAPE, MAE/MAD, MFE, and tracking signal;
        - explain how forecasting parameters affect responsiveness;
        - distinguish validation performance from final out-of-sample performance;
        - compare top-down and bottom-up forecasting;
        - explain why aggregate accuracy may differ from state-level accuracy.
        """
    )


# =========================================================
# DATA EXPLANATION
# =========================================================

with st.expander(
    "About the data"
):

    st.write(
        """
        Monthly Tesla Model Y state registrations are used as a proxy
        for demand.

        Registration counts are not identical to actual monthly vehicle
        sales, but they provide a useful demand proxy for this simulation.
        """
    )


# =========================================================
# METRIC EXPLANATION
# =========================================================

with st.expander(
    "How to interpret forecast accuracy measures"
):

    st.markdown(
        """
        **MAPE — Mean Absolute Percentage Error**

        Average absolute forecast error expressed as a percentage of
        actual demand. Lower values generally indicate better accuracy.

        **MAE / MAD — Mean Absolute Error / Mean Absolute Deviation**

        Average absolute forecast error measured in registration units.

        **MFE — Mean Forecast Error**

        Measures forecast bias.

        - Positive MFE = systematic **underforecasting**
        - Negative MFE = systematic **overforecasting**

        **Tracking Signal**

        Measures cumulative forecast bias relative to MAD.

        - Positive = tendency to underforecast
        - Negative = tendency to overforecast
        - Values farther from zero indicate stronger systematic bias.
        """
    )


# =========================================================
# SIMULATION CONTROLS / RESET
# =========================================================

with st.expander(
    "Simulation controls"
):

    st.write(
        """
        Starting a new attempt erases all judgmental forecasts,
        model experiments, final results, and hierarchical forecasting
        attempts from this browser session.
        """
    )

    if st.button(
        "Start New Attempt",
        key="reset_forecasting_simulation",
    ):

        keys_to_delete = [
            key
            for key
            in list(
                st.session_state.keys()
            )
            if (
                key.startswith(
                    "forecast_"
                )
                or key.startswith(
                    "judgment_"
                )
                or key.startswith(
                    "part1_"
                )
                or key.startswith(
                    "td_"
                )
                or key.startswith(
                    "bu_"
                )
                or key.startswith(
                    "comparison_"
                )
            )
        ]

        for key in keys_to_delete:

            del st.session_state[
                key
            ]

        st.rerun()


# =========================================================
# PROGRESS
# =========================================================

part1_progress = []

if st.session_state[
    "forecast_judgment_submitted"
]:

    part1_progress.append(
        "Judgment ✓"
    )

else:

    part1_progress.append(
        "Judgment ○"
    )


if len(
    st.session_state[
        "forecast_model_results"
    ]
) > 0:

    part1_progress.append(
        "Model Lab ✓"
    )

else:

    part1_progress.append(
        "Model Lab ○"
    )


if st.session_state[
    "forecast_final_locked"
]:

    part1_progress.append(
        "Final Model ✓"
    )

else:

    part1_progress.append(
        "Final Model ○"
    )


if st.session_state[
    "forecast_final_revealed"
]:

    part1_progress.append(
        "Final Test ✓"
    )

else:

    part1_progress.append(
        "Final Test ○"
    )


part2_progress = []

if len(
    st.session_state[
        "forecast_top_down_results"
    ]
) > 0:

    part2_progress.append(
        "Top-Down ✓"
    )

else:

    part2_progress.append(
        "Top-Down ○"
    )


if len(
    st.session_state[
        "forecast_bottom_up_results"
    ]
) > 0:

    part2_progress.append(
        "Bottom-Up ✓"
    )

else:

    part2_progress.append(
        "Bottom-Up ○"
    )


# =========================================================
# MAIN PART NAVIGATION
# =========================================================

st.divider()

simulation_part = st.segmented_control(
    "Choose a simulation part",
    options=[
        "Part 1: Forecasting Methods",
        "Part 2: Hierarchical Forecasting",
    ],
    default="Part 1: Forecasting Methods",
    selection_mode="single",
    key="forecast_simulation_part",
    width="stretch",
)

st.divider()


# =========================================================
# =========================================================
# PART 1
# FORECASTING METHODS
# =========================================================
# =========================================================

if (
    simulation_part
    == "Part 1: Forecasting Methods"
):

    st.header(
        "Part 1 — Forecasting Methods"
    )

    st.caption(
        "Progress: "
        + "  •  ".join(
            part1_progress
        )
    )

    st.write(
        """
        Explore demand, make a judgmental forecast, experiment with
        quantitative forecasting methods, and evaluate a true
        out-of-sample forecast.
        """
    )


    stage1, stage2, stage3, stage4, stage5 = st.tabs(
        [
            "1. Explore",
            "2. Judgment",
            "3. Model Lab",
            "4. Lock Model",
            "5. Final Test",
        ]
    )


    # =====================================================
    # PART 1 — STAGE 1
    # EXPLORE
    # =====================================================

    with stage1:

        st.subheader(
            "Explore Historical Demand"
        )

        st.write(
            """
            Examine the historical demand pattern before making
            any forecasts.
            """
        )

        historical_window = st.selectbox(
            "Historical information to display",
            options=[
                12,
                24,
                36,
                "All available",
            ],
            index=2,
            key="part1_explore_window",
        )

        if (
            historical_window
            == "All available"
        ):

            display_df = (
                train_df.copy()
            )

        else:

            display_df = (
                select_training_window(
                    train_df,
                    months=
                    historical_window,
                )
            )

        st.caption(
            f"Displayed period: "
            f"{display_df['Year-Month'].min().strftime('%b %Y')} "
            f"to "
            f"{display_df['Year-Month'].max().strftime('%b %Y')}"
        )


        st.markdown(
            "#### Aggregate Demand"
        )

        aggregate_chart = (
            display_df[
                [
                    "Year-Month",
                    "Aggregate",
                ]
            ]
            .set_index(
                "Year-Month"
            )
        )

        st.line_chart(
            aggregate_chart
        )


        st.markdown(
            "#### State-Level Demand"
        )

        state_chart = (
            display_df[
                [
                    "Year-Month",
                    "NY",
                    "CO",
                    "TX",
                ]
            ]
            .set_index(
                "Year-Month"
            )
        )

        st.line_chart(
            state_chart
        )


        with st.expander(
            "View monthly registration data"
        ):

            display_table = (
                display_df[
                    [
                        "Year-Month",
                        "NY",
                        "CO",
                        "TX",
                        "Aggregate",
                    ]
                ].copy()
            )

            display_table[
                "Year-Month"
            ] = (
                display_table[
                    "Year-Month"
                ]
                .dt.strftime(
                    "%b %Y"
                )
            )

            st.dataframe(
                display_table,
                hide_index=True,
                use_container_width=True,
            )


        st.markdown(
            """
            **Consider:**

            - Is demand stable or changing?
            - Is there evidence of a trend?
            - How different are NY, CO, and TX?
            - Are recent observations likely to be more useful than older observations?
            """
        )


    # =====================================================
    # PART 1 — STAGE 2
    # JUDGMENTAL FORECAST
    # =====================================================

    with stage2:

        st.subheader(
            "Make a Judgmental Forecast"
        )

        st.write(
            """
            Forecast aggregate Model Y demand for
            **July–December 2024** without using a formal model.
            """
        )

        validation_months = (
            validation_df[
                "Year-Month"
            ]
            .dt.strftime(
                "%b %Y"
            )
            .tolist()
        )


        if not st.session_state[
            "forecast_judgment_submitted"
        ]:

            st.warning(
                """
                Once submitted, this judgmental forecast is saved
                for the current attempt.
                """
            )

            judgment_inputs = []

            left_col, right_col = (
                st.columns(
                    2
                )
            )

            for i, month in enumerate(
                validation_months
            ):

                target_column = (
                    left_col
                    if i < 3
                    else right_col
                )

                with target_column:

                    value = (
                        st.number_input(
                            month,
                            min_value=0,
                            value=150000,
                            step=1000,
                            key=
                            f"judgment_{i}",
                        )
                    )

                    judgment_inputs.append(
                        value
                    )


            if st.button(
                "Submit Judgmental Forecast",
                type="primary",
                key=
                "submit_judgment",
            ):

                judgment_array = (
                    np.asarray(
                        judgment_inputs,
                        dtype=float,
                    )
                )

                result = (
                    evaluate_judgmental_forecast(
                        forecasts=
                        judgment_array,
                        stage=
                        "validation",
                        region=
                        "Aggregate",
                    )
                )

                st.session_state[
                    "forecast_judgment_values"
                ] = judgment_array

                st.session_state[
                    "forecast_judgment_result"
                ] = result

                st.session_state[
                    "forecast_judgment_submitted"
                ] = True

                st.rerun()


        else:

            st.success(
                "Judgmental forecast submitted."
            )

            result = (
                st.session_state[
                    "forecast_judgment_result"
                ]
            )

            metric1, metric2, metric3, metric4 = (
                st.columns(
                    4
                )
            )

            metric1.metric(
                "MAPE",
                f"{result['metrics']['MAPE']:.2f}%",
            )

            metric2.metric(
                "MAE / MAD",
                f"{result['metrics']['MAE']:,.0f}",
            )

            metric3.metric(
                "MFE",
                f"{result['metrics']['MFE']:,.0f}",
            )

            metric4.metric(
                "Tracking Signal",
                f"{result['metrics']['Tracking Signal']:.2f}",
            )


            chart_df = pd.DataFrame(
                {
                    "Month":
                        validation_df[
                            "Year-Month"
                        ],

                    "Actual":
                        result[
                            "actual"
                        ],

                    "Your Forecast":
                        result[
                            "forecast"
                        ],
                }
            ).set_index(
                "Month"
            )

            st.line_chart(
                chart_df
            )


            with st.expander(
                "Detailed forecast errors"
            ):

                st.dataframe(
                    result[
                        "error_table"
                    ].round(2),
                    hide_index=True,
                    use_container_width=True,
                )


    # =====================================================
    # PART 1 — STAGE 3
    # MODEL LAB
    # =====================================================

    with stage3:

        st.subheader(
            "Model Laboratory"
        )

        st.write(
            """
            Use historical data through **June 2024**
            to forecast **July–December 2024**.

            Because this is the validation period, you may experiment
            with different methods and parameters.
            """
        )


        if st.session_state[
            "forecast_final_locked"
        ]:

            st.warning(
                """
                Your final model has already been locked.
                Model experimentation is closed for this attempt.
                """
            )


        else:

            model_config = (
                forecasting_model_controls(
                    key_prefix=
                    "part1_model",
                )
            )


            if st.button(
                "Run Forecast",
                type="primary",
                disabled=
                not model_config[
                    "can_run"
                ],
                key=
                "part1_run_model",
            ):

                try:

                    result = (
                        run_validation(
                            method=
                            model_config[
                                "method"
                            ],
                            region=
                            "Aggregate",
                            training_window=
                            "all",
                            **model_config[
                                "parameters"
                            ],
                        )
                    )

                    saved_result = {
                        "Method":
                            model_config[
                                "method"
                            ],

                        "Parameters":
                            model_config[
                                "description"
                            ],

                        "Raw Parameters":
                            copy.deepcopy(
                                model_config[
                                    "parameters"
                                ]
                            ),

                        "MAPE":
                            result[
                                "metrics"
                            ][
                                "MAPE"
                            ],

                        "MAE":
                            result[
                                "metrics"
                            ][
                                "MAE"
                            ],

                        "MFE":
                            result[
                                "metrics"
                            ][
                                "MFE"
                            ],

                        "Tracking Signal":
                            result[
                                "metrics"
                            ][
                                "Tracking Signal"
                            ],

                        "Forecast":
                            result[
                                "forecast"
                            ],

                        "Result":
                            result,
                    }

                    st.session_state[
                        "forecast_model_results"
                    ].append(
                        saved_result
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"The model could not be estimated: {error}"
                    )


        model_results = (
            st.session_state[
                "forecast_model_results"
            ]
        )


        if len(
            model_results
        ) > 0:

            latest = (
                model_results[-1]
            )

            latest_result = (
                latest[
                    "Result"
                ]
            )

            st.divider()

            st.markdown(
                "#### Most Recent Forecast"
            )

            st.write(
                f"**Method:** {latest['Method']}"
            )

            st.write(
                f"**Parameters:** {latest['Parameters']}"
            )


            metric1, metric2, metric3, metric4 = (
                st.columns(
                    4
                )
            )

            metric1.metric(
                "MAPE",
                f"{latest['MAPE']:.2f}%",
            )

            metric2.metric(
                "MAE / MAD",
                f"{latest['MAE']:,.0f}",
            )

            metric3.metric(
                "MFE",
                f"{latest['MFE']:,.0f}",
            )

            metric4.metric(
                "Tracking Signal",
                f"{latest['Tracking Signal']:.2f}",
            )


            validation_plot = (
                pd.DataFrame(
                    {
                        "Month":
                            validation_df[
                                "Year-Month"
                            ],

                        "Actual":
                            latest_result[
                                "actual"
                            ],

                        "Forecast":
                            latest_result[
                                "forecast"
                            ],
                    }
                )
                .set_index(
                    "Month"
                )
            )

            st.line_chart(
                validation_plot
            )


            st.markdown(
                "#### Model Comparison"
            )

            comparison_rows = []

            for i, item in enumerate(
                model_results
            ):

                comparison_rows.append(
                    {
                        "Attempt":
                            i + 1,

                        "Method":
                            item[
                                "Method"
                            ],

                        "Parameters":
                            item[
                                "Parameters"
                            ],

                        "MAPE":
                            item[
                                "MAPE"
                            ],

                        "MAE / MAD":
                            item[
                                "MAE"
                            ],

                        "MFE":
                            item[
                                "MFE"
                            ],

                        "Tracking Signal":
                            item[
                                "Tracking Signal"
                            ],
                    }
                )

            comparison_df = (
                pd.DataFrame(
                    comparison_rows
                )
            )

            comparison_df[
                [
                    "MAPE",
                    "Tracking Signal",
                ]
            ] = (
                comparison_df[
                    [
                        "MAPE",
                        "Tracking Signal",
                    ]
                ]
                .round(2)
            )

            comparison_df[
                [
                    "MAE / MAD",
                    "MFE",
                ]
            ] = (
                comparison_df[
                    [
                        "MAE / MAD",
                        "MFE",
                    ]
                ]
                .round(0)
            )

            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True,
            )


            if not st.session_state[
                "forecast_final_locked"
            ]:

                if st.button(
                    "Clear Model Experiments",
                    key=
                    "part1_clear_models",
                ):

                    st.session_state[
                        "forecast_model_results"
                    ] = []

                    st.rerun()


    # =====================================================
    # PART 1 — STAGE 4
    # LOCK FINAL MODEL
    # =====================================================

    with stage4:

        st.subheader(
            "Select and Lock Final Model"
        )

        st.write(
            """
            Choose one validated model for the final
            **January–June 2025** forecast.
            """
        )

        model_results = (
            st.session_state[
                "forecast_model_results"
            ]
        )


        if len(
            model_results
        ) == 0:

            st.warning(
                """
                Run at least one forecasting model
                in the Model Lab first.
                """
            )


        elif st.session_state[
            "forecast_final_locked"
        ]:

            final_model = (
                st.session_state[
                    "forecast_final_model"
                ]
            )

            st.success(
                "Final model locked."
            )

            st.write(
                f"**Method:** {final_model['Method']}"
            )

            st.write(
                f"**Parameters:** {final_model['Parameters']}"
            )

            st.write(
                f"**Validation MAPE:** "
                f"{final_model['MAPE']:.2f}%"
            )

            st.info(
                """
                The model is refit using all information available
                through December 2024 before producing the final forecast.
                """
            )


        else:

            selection_options = []

            for i, item in enumerate(
                model_results
            ):

                selection_options.append(
                    (
                        f"Attempt {i + 1}: "
                        f"{item['Method']} | "
                        f"{item['Parameters']} | "
                        f"MAPE {item['MAPE']:.2f}%"
                    )
                )


            selected_label = (
                st.selectbox(
                    "Final model",
                    options=
                    selection_options,
                    key=
                    "part1_final_selection",
                )
            )

            selected_index = (
                selection_options.index(
                    selected_label
                )
            )

            selected_model = (
                model_results[
                    selected_index
                ]
            )


            st.write(
                f"**Method:** "
                f"{selected_model['Method']}"
            )

            st.write(
                f"**Parameters:** "
                f"{selected_model['Parameters']}"
            )

            st.write(
                f"**Validation MAPE:** "
                f"{selected_model['MAPE']:.2f}%"
            )


            st.warning(
                """
                Once locked, the method and parameters cannot
                be changed during this attempt.
                """
            )

            confirm_lock = (
                st.checkbox(
                    "I am ready to use this model for the final forecast.",
                    key=
                    "part1_confirm_lock",
                )
            )


            if st.button(
                "Lock Final Model & Generate Forecast",
                type="primary",
                disabled=
                not confirm_lock,
                key=
                "part1_lock_model",
            ):

                try:

                    final_result = (
                        run_final_forecast(
                            method=
                            selected_model[
                                "Method"
                            ],
                            region=
                            "Aggregate",
                            **copy.deepcopy(
                                selected_model[
                                    "Raw Parameters"
                                ]
                            ),
                        )
                    )

                    st.session_state[
                        "forecast_final_model"
                    ] = copy.deepcopy(
                        selected_model
                    )

                    st.session_state[
                        "forecast_final_result"
                    ] = final_result

                    st.session_state[
                        "forecast_final_locked"
                    ] = True

                    st.session_state[
                        "forecast_final_revealed"
                    ] = False

                    st.session_state[
                        "forecast_final_evaluation"
                    ] = None

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"The final forecast could not be generated: {error}"
                    )


    # =====================================================
    # PART 1 — STAGE 5
    # FINAL TEST
    # =====================================================

    with stage5:

        st.subheader(
            "Final Out-of-Sample Test"
        )


        if not st.session_state[
            "forecast_final_locked"
        ]:

            st.warning(
                """
                Lock a final model before viewing
                the final forecast.
                """
            )


        else:

            final_model = (
                st.session_state[
                    "forecast_final_model"
                ]
            )

            final_result = (
                st.session_state[
                    "forecast_final_result"
                ]
            )


            st.write(
                f"**Method:** {final_model['Method']}"
            )

            st.write(
                f"**Parameters:** {final_model['Parameters']}"
            )


            final_forecast_table = (
                pd.DataFrame(
                    {
                        "Month":
                            final_result[
                                "forecast_dates"
                            ]
                            .dt.strftime(
                                "%b %Y"
                            ),

                        "Forecast":
                            np.round(
                                final_result[
                                    "forecast"
                                ],
                                0,
                            ).astype(
                                int
                            ),
                    }
                )
            )

            st.dataframe(
                final_forecast_table,
                hide_index=True,
                use_container_width=True,
            )


            if not st.session_state[
                "forecast_final_revealed"
            ]:

                st.info(
                    """
                    January–June 2025 actual registrations
                    are still hidden.
                    """
                )

                if st.button(
                    "Reveal Actual Demand & Final Results",
                    type="primary",
                    key=
                    "part1_reveal_final",
                ):

                    evaluation = (
                        evaluate_final_forecast(
                            final_result
                        )
                    )

                    st.session_state[
                        "forecast_final_evaluation"
                    ] = evaluation

                    st.session_state[
                        "forecast_final_revealed"
                    ] = True

                    st.rerun()


            else:

                evaluation = (
                    st.session_state[
                        "forecast_final_evaluation"
                    ]
                )

                st.success(
                    "Final test results revealed."
                )


                metric1, metric2, metric3, metric4 = (
                    st.columns(
                        4
                    )
                )

                metric1.metric(
                    "MAPE",
                    f"{evaluation['metrics']['MAPE']:.2f}%",
                )

                metric2.metric(
                    "MAE / MAD",
                    f"{evaluation['metrics']['MAE']:,.0f}",
                )

                metric3.metric(
                    "MFE",
                    f"{evaluation['metrics']['MFE']:,.0f}",
                )

                metric4.metric(
                    "Tracking Signal",
                    f"{evaluation['metrics']['Tracking Signal']:.2f}",
                )


                final_chart = (
                    pd.DataFrame(
                        {
                            "Month":
                                evaluation[
                                    "forecast_dates"
                                ],

                            "Actual":
                                evaluation[
                                    "actual"
                                ],

                            "Forecast":
                                evaluation[
                                    "forecast"
                                ],
                        }
                    )
                    .set_index(
                        "Month"
                    )
                )

                st.line_chart(
                    final_chart
                )


                st.markdown(
                    "#### Validation vs. Final Test"
                )

                validation_test_df = (
                    pd.DataFrame(
                        {
                            "Measure": [
                                "MAPE",
                                "MAE / MAD",
                                "MFE",
                                "Tracking Signal",
                            ],

                            "Validation": [
                                final_model[
                                    "MAPE"
                                ],

                                final_model[
                                    "MAE"
                                ],

                                final_model[
                                    "MFE"
                                ],

                                final_model[
                                    "Tracking Signal"
                                ],
                            ],

                            "Final Test": [
                                evaluation[
                                    "metrics"
                                ][
                                    "MAPE"
                                ],

                                evaluation[
                                    "metrics"
                                ][
                                    "MAE"
                                ],

                                evaluation[
                                    "metrics"
                                ][
                                    "MFE"
                                ],

                                evaluation[
                                    "metrics"
                                ][
                                    "Tracking Signal"
                                ],
                            ],
                        }
                    )
                )

                st.dataframe(
                    validation_test_df.round(
                        2
                    ),
                    hide_index=True,
                    use_container_width=True,
                )


                st.markdown(
                    """
                    **Reflect:**

                    - Did validation performance predict final performance?
                    - Did the model overforecast or underforecast?
                    - Would you choose the same model again?
                    - What does this suggest about relying only on historical performance?
                    """
                )


# =========================================================
# =========================================================
# PART 2
# HIERARCHICAL FORECASTING
# =========================================================
# =========================================================

elif (
    simulation_part
    == "Part 2: Hierarchical Forecasting"
):

    st.header(
        "Part 2 — Hierarchical Forecasting"
    )

    st.caption(
        "Progress: "
        + "  •  ".join(
            part2_progress
        )
    )

    st.write(
        """
        Compare forecasting at different organizational levels
        using top-down and bottom-up approaches.
        """
    )


    stage6, stage7, stage8 = st.tabs(
        [
            "1. Top-Down",
            "2. Bottom-Up",
            "3. Compare",
        ]
    )


    # =====================================================
    # PART 2 — STAGE 1
    # TOP-DOWN
    # =====================================================

    with stage6:

        st.subheader(
            "Top-Down Forecasting"
        )

        st.write(
            """
            First forecast total NY + CO + TX demand.
            Then allocate that forecast across the three states.
            """
        )


        td_config = (
            forecasting_model_controls(
                key_prefix=
                "td",
                title=
                "Step 1: Aggregate Forecast",
            )
        )


        st.divider()

        st.markdown(
            "#### Step 2: Allocation Rule"
        )

        allocation_choice = (
            st.radio(
                "Allocate aggregate demand using:",
                options=[
                    "Recent 3-month sales shares",
                    "Historical sales shares",
                ],
                key=
                "td_allocation_rule",
            )
        )


        if (
            allocation_choice
            == "Recent 3-month sales shares"
        ):

            allocation_rule = (
                "recent_3_months"
            )

            st.caption(
                """
                Uses state shares from the final three
                months of the training period.
                """
            )

        else:

            allocation_rule = (
                "historical"
            )

            st.caption(
                """
                Uses cumulative state shares across
                the full training period.
                """
            )


        if st.button(
            "Run Top-Down Forecast",
            type="primary",
            disabled=
            not td_config[
                "can_run"
            ],
            key=
            "td_run",
        ):

            try:

                aggregate_result = (
                    run_validation(
                        method=
                        td_config[
                            "method"
                        ],
                        region=
                        "Aggregate",
                        training_window=
                        "all",
                        **td_config[
                            "parameters"
                        ],
                    )
                )


                td_result = (
                    top_down_forecast(
                        aggregate_forecast=
                        aggregate_result[
                            "forecast"
                        ],

                        historical_df=
                        train_df,

                        allocation_rule=
                        allocation_rule,
                    )
                )


                td_metrics = (
                    evaluate_hierarchical_forecast(
                        actual_df=
                        validation_df,

                        forecast_df=
                        td_result[
                            "forecasts"
                        ],
                    )
                )


                saved_result = {
                    "Method":
                        td_config[
                            "method"
                        ],

                    "Parameters":
                        td_config[
                            "description"
                        ],

                    "Raw Parameters":
                        copy.deepcopy(
                            td_config[
                                "parameters"
                            ]
                        ),

                    "Allocation Rule":
                        allocation_choice,

                    "Allocation Rule Code":
                        allocation_rule,

                    "Shares":
                        td_result[
                            "shares"
                        ],

                    "Forecasts":
                        td_result[
                            "forecasts"
                        ],

                    "Metrics":
                        td_metrics,
                }


                st.session_state[
                    "forecast_top_down_results"
                ].append(
                    saved_result
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Top-down forecast failed: {error}"
                )


        td_results = (
            st.session_state[
                "forecast_top_down_results"
            ]
        )


        if len(
            td_results
        ) > 0:

            latest_td = (
                td_results[-1]
            )

            st.divider()

            st.markdown(
                "#### Latest Top-Down Result"
            )

            st.write(
                f"**Model:** "
                f"{latest_td['Method']} — "
                f"{latest_td['Parameters']}"
            )

            st.write(
                f"**Allocation:** "
                f"{latest_td['Allocation Rule']}"
            )


            share1, share2, share3 = (
                st.columns(
                    3
                )
            )

            share1.metric(
                "NY Share",
                f"{latest_td['Shares']['NY'] * 100:.1f}%",
            )

            share2.metric(
                "CO Share",
                f"{latest_td['Shares']['CO'] * 100:.1f}%",
            )

            share3.metric(
                "TX Share",
                f"{latest_td['Shares']['TX'] * 100:.1f}%",
            )


            metrics_table = (
                create_hierarchy_metrics_table(
                    latest_td[
                        "Metrics"
                    ]
                )
            )

            st.dataframe(
                metrics_table[
                    [
                        "Region",
                        "MAPE",
                        "MAE",
                        "MFE",
                        "Tracking Signal",
                    ]
                ].round(
                    2
                ),
                hide_index=True,
                use_container_width=True,
            )


            td_region = (
                st.selectbox(
                    "Forecast level to visualize",
                    options=[
                        "Aggregate",
                        "NY",
                        "CO",
                        "TX",
                    ],
                    key=
                    "td_chart_region",
                )
            )


            td_chart = (
                pd.DataFrame(
                    {
                        "Month":
                            validation_df[
                                "Year-Month"
                            ],

                        "Actual":
                            validation_df[
                                td_region
                            ],

                        "Forecast":
                            latest_td[
                                "Forecasts"
                            ][
                                td_region
                            ],
                    }
                )
                .set_index(
                    "Month"
                )
            )

            st.line_chart(
                td_chart
            )


            if len(
                td_results
            ) > 1:

                st.markdown(
                    "#### Experiment History"
                )

                history_rows = []

                for i, item in enumerate(
                    td_results
                ):

                    history_rows.append(
                        {
                            "Attempt":
                                i + 1,

                            "Model":
                                item[
                                    "Method"
                                ],

                            "Parameters":
                                item[
                                    "Parameters"
                                ],

                            "Allocation":
                                item[
                                    "Allocation Rule"
                                ],

                            "Aggregate MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "Aggregate"
                                ][
                                    "MAPE"
                                ],

                            "NY MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "NY"
                                ][
                                    "MAPE"
                                ],

                            "CO MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "CO"
                                ][
                                    "MAPE"
                                ],

                            "TX MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "TX"
                                ][
                                    "MAPE"
                                ],
                        }
                    )

                st.dataframe(
                    pd.DataFrame(
                        history_rows
                    ).round(
                        2
                    ),
                    hide_index=True,
                    use_container_width=True,
                )


    # =====================================================
    # PART 2 — STAGE 2
    # BOTTOM-UP
    # =====================================================

    with stage7:

        st.subheader(
            "Bottom-Up Forecasting"
        )

        st.write(
            """
            Forecast NY, CO, and TX independently.
            The three state forecasts are then added together
            to create aggregate demand.
            """
        )


        ny_config = (
            forecasting_model_controls(
                key_prefix=
                "bu_ny",
                title=
                "New York",
            )
        )

        st.divider()

        co_config = (
            forecasting_model_controls(
                key_prefix=
                "bu_co",
                title=
                "Colorado",
            )
        )

        st.divider()

        tx_config = (
            forecasting_model_controls(
                key_prefix=
                "bu_tx",
                title=
                "Texas",
            )
        )


        bu_can_run = (
            ny_config[
                "can_run"
            ]
            and co_config[
                "can_run"
            ]
            and tx_config[
                "can_run"
            ]
        )


        if st.button(
            "Run Bottom-Up Forecast",
            type="primary",
            disabled=
            not bu_can_run,
            key=
            "bu_run",
        ):

            try:

                ny_result = (
                    run_validation(
                        method=
                        ny_config[
                            "method"
                        ],
                        region=
                        "NY",
                        training_window=
                        "all",
                        **ny_config[
                            "parameters"
                        ],
                    )
                )

                co_result = (
                    run_validation(
                        method=
                        co_config[
                            "method"
                        ],
                        region=
                        "CO",
                        training_window=
                        "all",
                        **co_config[
                            "parameters"
                        ],
                    )
                )

                tx_result = (
                    run_validation(
                        method=
                        tx_config[
                            "method"
                        ],
                        region=
                        "TX",
                        training_window=
                        "all",
                        **tx_config[
                            "parameters"
                        ],
                    )
                )


                bu_forecasts = (
                    combine_bottom_up_forecasts(
                        ny_forecast=
                        ny_result[
                            "forecast"
                        ],

                        co_forecast=
                        co_result[
                            "forecast"
                        ],

                        tx_forecast=
                        tx_result[
                            "forecast"
                        ],
                    )
                )


                bu_metrics = (
                    evaluate_hierarchical_forecast(
                        actual_df=
                        validation_df,

                        forecast_df=
                        bu_forecasts,
                    )
                )


                saved_result = {
                    "NY Method":
                        ny_config[
                            "method"
                        ],

                    "NY Parameters":
                        ny_config[
                            "description"
                        ],

                    "CO Method":
                        co_config[
                            "method"
                        ],

                    "CO Parameters":
                        co_config[
                            "description"
                        ],

                    "TX Method":
                        tx_config[
                            "method"
                        ],

                    "TX Parameters":
                        tx_config[
                            "description"
                        ],

                    "Forecasts":
                        bu_forecasts,

                    "Metrics":
                        bu_metrics,
                }


                st.session_state[
                    "forecast_bottom_up_results"
                ].append(
                    saved_result
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Bottom-up forecast failed: {error}"
                )


        bu_results = (
            st.session_state[
                "forecast_bottom_up_results"
            ]
        )


        if len(
            bu_results
        ) > 0:

            latest_bu = (
                bu_results[-1]
            )

            st.divider()

            st.markdown(
                "#### Latest Bottom-Up Result"
            )


            model_summary = (
                pd.DataFrame(
                    {
                        "State": [
                            "NY",
                            "CO",
                            "TX",
                        ],

                        "Method": [
                            latest_bu[
                                "NY Method"
                            ],

                            latest_bu[
                                "CO Method"
                            ],

                            latest_bu[
                                "TX Method"
                            ],
                        ],

                        "Parameters": [
                            latest_bu[
                                "NY Parameters"
                            ],

                            latest_bu[
                                "CO Parameters"
                            ],

                            latest_bu[
                                "TX Parameters"
                            ],
                        ],
                    }
                )
            )

            st.dataframe(
                model_summary,
                hide_index=True,
                use_container_width=True,
            )


            metrics_table = (
                create_hierarchy_metrics_table(
                    latest_bu[
                        "Metrics"
                    ]
                )
            )

            st.dataframe(
                metrics_table[
                    [
                        "Region",
                        "MAPE",
                        "MAE",
                        "MFE",
                        "Tracking Signal",
                    ]
                ].round(
                    2
                ),
                hide_index=True,
                use_container_width=True,
            )


            bu_region = (
                st.selectbox(
                    "Forecast level to visualize",
                    options=[
                        "Aggregate",
                        "NY",
                        "CO",
                        "TX",
                    ],
                    key=
                    "bu_chart_region",
                )
            )


            bu_chart = (
                pd.DataFrame(
                    {
                        "Month":
                            validation_df[
                                "Year-Month"
                            ],

                        "Actual":
                            validation_df[
                                bu_region
                            ],

                        "Forecast":
                            latest_bu[
                                "Forecasts"
                            ][
                                bu_region
                            ],
                    }
                )
                .set_index(
                    "Month"
                )
            )

            st.line_chart(
                bu_chart
            )


            if len(
                bu_results
            ) > 1:

                st.markdown(
                    "#### Experiment History"
                )

                history_rows = []

                for i, item in enumerate(
                    bu_results
                ):

                    history_rows.append(
                        {
                            "Attempt":
                                i + 1,

                            "NY Model":
                                (
                                    f"{item['NY Method']} | "
                                    f"{item['NY Parameters']}"
                                ),

                            "CO Model":
                                (
                                    f"{item['CO Method']} | "
                                    f"{item['CO Parameters']}"
                                ),

                            "TX Model":
                                (
                                    f"{item['TX Method']} | "
                                    f"{item['TX Parameters']}"
                                ),

                            "Aggregate MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "Aggregate"
                                ][
                                    "MAPE"
                                ],

                            "NY MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "NY"
                                ][
                                    "MAPE"
                                ],

                            "CO MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "CO"
                                ][
                                    "MAPE"
                                ],

                            "TX MAPE":
                                item[
                                    "Metrics"
                                ][
                                    "TX"
                                ][
                                    "MAPE"
                                ],
                        }
                    )

                st.dataframe(
                    pd.DataFrame(
                        history_rows
                    ).round(
                        2
                    ),
                    hide_index=True,
                    use_container_width=True,
                )


    # =====================================================
    # PART 2 — STAGE 3
    # COMPARE APPROACHES
    # =====================================================

    with stage8:

        st.subheader(
            "Compare Top-Down and Bottom-Up"
        )

        td_results = (
            st.session_state[
                "forecast_top_down_results"
            ]
        )

        bu_results = (
            st.session_state[
                "forecast_bottom_up_results"
            ]
        )


        if (
            len(
                td_results
            ) == 0
            or len(
                bu_results
            ) == 0
        ):

            st.warning(
                """
                Complete at least one top-down and one
                bottom-up forecast before comparing approaches.
                """
            )


        else:

            td_options = []

            for i, item in enumerate(
                td_results
            ):

                td_options.append(
                    (
                        f"Attempt {i + 1}: "
                        f"{item['Method']} | "
                        f"{item['Parameters']} | "
                        f"{item['Allocation Rule']}"
                    )
                )


            selected_td_label = (
                st.selectbox(
                    "Top-down attempt",
                    options=
                    td_options,
                    key=
                    "comparison_td_attempt",
                )
            )

            selected_td = (
                td_results[
                    td_options.index(
                        selected_td_label
                    )
                ]
            )


            bu_options = []

            for i, item in enumerate(
                bu_results
            ):

                bu_options.append(
                    (
                        f"Attempt {i + 1}: "
                        f"NY={item['NY Method']} | "
                        f"CO={item['CO Method']} | "
                        f"TX={item['TX Method']}"
                    )
                )


            selected_bu_label = (
                st.selectbox(
                    "Bottom-up attempt",
                    options=
                    bu_options,
                    key=
                    "comparison_bu_attempt",
                )
            )

            selected_bu = (
                bu_results[
                    bu_options.index(
                        selected_bu_label
                    )
                ]
            )


            comparison_rows = []

            for region in [
                "Aggregate",
                "NY",
                "CO",
                "TX",
            ]:

                comparison_rows.append(
                    {
                        "Region":
                            region,

                        "Top-Down MAPE":
                            selected_td[
                                "Metrics"
                            ][
                                region
                            ][
                                "MAPE"
                            ],

                        "Bottom-Up MAPE":
                            selected_bu[
                                "Metrics"
                            ][
                                region
                            ][
                                "MAPE"
                            ],

                        "Top-Down MAE":
                            selected_td[
                                "Metrics"
                            ][
                                region
                            ][
                                "MAE"
                            ],

                        "Bottom-Up MAE":
                            selected_bu[
                                "Metrics"
                            ][
                                region
                            ][
                                "MAE"
                            ],
                    }
                )


            comparison_df = (
                pd.DataFrame(
                    comparison_rows
                )
            )


            comparison_df[
                "MAPE Difference"
            ] = (
                comparison_df[
                    "Top-Down MAPE"
                ]
                - comparison_df[
                    "Bottom-Up MAPE"
                ]
            )


            st.dataframe(
                comparison_df.round(
                    2
                ),
                hide_index=True,
                use_container_width=True,
            )


            mape_chart = (
                comparison_df[
                    [
                        "Region",
                        "Top-Down MAPE",
                        "Bottom-Up MAPE",
                    ]
                ]
                .set_index(
                    "Region"
                )
            )

            st.bar_chart(
                mape_chart
            )


            comparison_region = (
                st.selectbox(
                    "Forecast level to visualize",
                    options=[
                        "Aggregate",
                        "NY",
                        "CO",
                        "TX",
                    ],
                    key=
                    "comparison_region",
                )
            )


            path_chart = (
                pd.DataFrame(
                    {
                        "Month":
                            validation_df[
                                "Year-Month"
                            ],

                        "Actual":
                            validation_df[
                                comparison_region
                            ],

                        "Top-Down":
                            selected_td[
                                "Forecasts"
                            ][
                                comparison_region
                            ],

                        "Bottom-Up":
                            selected_bu[
                                "Forecasts"
                            ][
                                comparison_region
                            ],
                    }
                )
                .set_index(
                    "Month"
                )
            )

            st.line_chart(
                path_chart
            )


            st.markdown(
                """
                ### Interpret the comparison

                - Which approach performs better at the aggregate level?
                - Does the same approach perform better for every state?
                - Can state-level errors offset one another?
                - How much does the top-down allocation rule matter?
                - Which approach would you use for company-wide planning?
                - Which approach would you use for state-level inventory planning?
                """
            )