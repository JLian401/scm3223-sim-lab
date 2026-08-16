import random
import streamlit as st
import pandas as pd

from sequential_inventory_engine import (
    InventorySimulation,
    DecisionValidationError,
    PRODUCTS,
    PRODUCT_LABELS,
    PRODUCT_SPECS,
    CHINA_MOQ,
    TJL_MOQ,
    ORDER_INCREMENT,
    TJL_MONTHLY_CAPACITY,
    APPROVED_SCENARIO_IDS,
    DEFAULT_SCENARIO_ID,
    MAX_ATTEMPTS,
    ENGINE_VERSION,
)

PAGE_VERSION = "2026-08-16-scenario-bank-v2"

st.set_page_config(
    page_title="Beautiful Bags Inventory Simulation",
    page_icon="👜",
    layout="wide",
)

SIM_KEY = "beautiful_bags_sim"
STARTED_KEY = "beautiful_bags_started"
ATTEMPT_KEY = "beautiful_bags_attempt_number"
USED_SCENARIOS_KEY = "beautiful_bags_used_scenarios"
CURRENT_SCENARIO_KEY = "beautiful_bags_current_scenario"


def money(x):
    return f"${x:,.0f}"


def qty(x):
    return f"{int(round(x)):,}"


def ensure_attempt_state():
    if ATTEMPT_KEY not in st.session_state:
        st.session_state[ATTEMPT_KEY] = 1
    if USED_SCENARIOS_KEY not in st.session_state:
        st.session_state[USED_SCENARIOS_KEY] = [DEFAULT_SCENARIO_ID]
    if CURRENT_SCENARIO_KEY not in st.session_state:
        st.session_state[CURRENT_SCENARIO_KEY] = DEFAULT_SCENARIO_ID


def clear_current_attempt_widgets():
    st.session_state.pop(SIM_KEY, None)
    st.session_state.pop(STARTED_KEY, None)
    for key in list(st.session_state.keys()):
        if key.startswith("bb_"):
            del st.session_state[key]


def start_next_attempt():
    ensure_attempt_state()
    current_attempt = st.session_state[ATTEMPT_KEY]
    if current_attempt >= MAX_ATTEMPTS:
        return False

    used = set(st.session_state[USED_SCENARIOS_KEY])
    available = [s for s in APPROVED_SCENARIO_IDS if s not in used]
    if not available:
        return False

    next_scenario = random.SystemRandom().choice(available)
    st.session_state[ATTEMPT_KEY] = current_attempt + 1
    st.session_state[CURRENT_SCENARIO_KEY] = next_scenario
    st.session_state[USED_SCENARIOS_KEY] = st.session_state[USED_SCENARIOS_KEY] + [next_scenario]
    clear_current_attempt_widgets()
    return True


def restart_current_attempt():
    clear_current_attempt_widgets()


def order_input(label, key, max_value, help_text=None):
    return st.number_input(
        label,
        min_value=0,
        max_value=int(max_value),
        value=0,
        step=ORDER_INCREMENT,
        key=key,
        help=help_text,
    )


def product_table():
    rows = []
    for p in PRODUCTS:
        s = PRODUCT_SPECS[p]
        rows.append({
            "Product": PRODUCT_LABELS[p],
            "Forecast Mean": f"{s.forecast_mean:,}",
            "Forecast SD": f"{s.forecast_sd:,}",
            "China Cost": f"${s.china_cost:.2f}",
            "TJL Cost": f"${s.tjl_cost:.2f}",
            "Regular Revenue / Unit": f"${s.regular_revenue:.2f}",
            "Expected End-of-Season Value": f"${s.salvage_value:.2f}",
            "China Maximum": f"{s.china_max:,}",
        })
    return pd.DataFrame(rows)


def render_progress(current_month, completed):
    labels = ["Preseason", "Month 1", "Month 2", "Month 3", "Month 4"]
    if not st.session_state.get(STARTED_KEY):
        active = 0
    elif completed:
        active = 4
    else:
        active = min(current_month + 1, 4)

    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < active:
            col.success(f"✓ {labels[i]}")
        elif i == active:
            col.info(labels[i])
        else:
            col.caption(labels[i])


def render_month_results(view):
    month = view["month"]
    st.subheader(f"Month {month} Results")

    rows = []
    for p in PRODUCTS:
        rows.append({
            "Product": PRODUCT_LABELS[p],
            "Sales This Month": int(view["products"][p]["sales"]),
            "Ending Inventory": int(round(view["products"][p]["ending_inventory"])),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Financial position")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales Revenue to Date", money(view["sales_revenue_to_date"]))
    c2.metric("Sourcing Cost Committed", money(view["sourcing_cost_committed"]))
    c3.metric("Holding Cost to Date", money(view["holding_cost_to_date"]))
    c4.metric("Interim Profit", money(view["interim_profit"]))
    st.caption(view["note"])

    outstanding = view.get("outstanding_tjl_orders", [])
    if outstanding:
        st.markdown("#### Outstanding TJL orders")
        rows = []
        for o in outstanding:
            rows.append({
                "Product": PRODUCT_LABELS[o["product"]],
                "Quantity": o["quantity"],
                "Expected Arrival Week": o["arrival_week"],
                "Lead Time": f'{o.get("lead_time_weeks", "-")} week(s)',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No TJL replenishment orders are currently outstanding.")


def render_replenishment_form(sim, month, view):
    st.divider()
    st.subheader(f"Replenishment Decision After Month {month}")

    lead_weeks = int(view["next_tjl_lead_time_weeks"])
    arrival_week = month * 4 + lead_weeks + 1
    st.info(
        f"**TJL lead time for this decision: {lead_weeks} week(s).** "
        f"An order placed now will be available at the beginning of Week {arrival_week}. "
        "Demand that cannot be served before the order arrives is lost."
    )

    st.write(
        f"You may order at most **{TJL_MONTHLY_CAPACITY:,} total units from TJL** "
        f"across all three products. For any product you choose to replenish, "
        f"the minimum is **{TJL_MOQ:,} units**, in increments of **{ORDER_INCREMENT}**."
    )
    st.caption(
        "You observe sales, not true customer demand. Use the information available "
        "to decide how much flexibility to use."
    )

    with st.form(f"bb_replenishment_form_{month}", clear_on_submit=False):
        cols = st.columns(3)
        order = {}
        for col, p in zip(cols, PRODUCTS):
            with col:
                order[p] = st.number_input(
                    PRODUCT_LABELS[p],
                    min_value=0,
                    max_value=TJL_MONTHLY_CAPACITY,
                    value=0,
                    step=ORDER_INCREMENT,
                    key=f"bb_m{month}_{p}",
                )

        total = sum(order.values())
        st.write(f"**Total TJL order: {total:,} / {TJL_MONTHLY_CAPACITY:,} units**")
        submitted = st.form_submit_button(
            f"Submit Month {month} Replenishment and Continue",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            sim.place_replenishment(month, order)
            sim.advance_one_month()
            st.rerun()
        except DecisionValidationError as e:
            st.error(str(e))


def render_final_results(sim):
    final = sim.final_results()

    st.subheader("Season Complete")
    st.success("You have completed the 16-week selling season.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Profit", money(final["final_profit"]))
    c2.metric("Sales Revenue", money(final["sales_revenue"]))
    c3.metric("Sourcing Cost", money(final["sourcing_cost"]))
    c4.metric("Holding Cost", money(final["holding_cost"]))

    c5, c6 = st.columns(2)
    c5.metric("End-of-Season Recovery Value", money(final["salvage_revenue"]))
    c6.metric("Total Ending Inventory", qty(sum(final["ending_inventory"].values())))

    rows = []
    for p in PRODUCTS:
        rows.append({
            "Product": PRODUCT_LABELS[p],
            "Units Sold": final["units_sold"][p],
            "Ending Inventory": int(round(final["ending_inventory"][p])),
            "Stockout Occurred?": "Yes" if final["stockout_occurred"][p] else "No",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Behind the Scenes")
    st.write(
        "These values were hidden while you were making decisions. They are shown "
        "now for debriefing so you can compare **sales** with **true customer demand**."
    )

    debrief_rows = []
    for p in PRODUCTS:
        debrief_rows.append({
            "Product": PRODUCT_LABELS[p],
            "True Seasonal Demand": final["seasonal_true_demand"][p],
            "Units Sold": final["units_sold"][p],
            "Lost Demand": final["lost_sales"][p],
            "Ending Inventory": int(round(final["ending_inventory"][p])),
        })
    st.dataframe(pd.DataFrame(debrief_rows), use_container_width=True, hide_index=True)

    with st.expander("See your sourcing decisions"):
        ledger = sim.state.order_ledger
        if ledger:
            rows = []
            for o in ledger:
                rows.append({
                    "Stage": o["stage"].replace("_", " ").title(),
                    "Source": o["source"],
                    "Product": PRODUCT_LABELS[o["product"]],
                    "Quantity": o["quantity"],
                    "Unit Cost": f'${o["unit_cost"]:.2f}',
                    "Lead Time": (
                        f'{o["lead_time_weeks"]} week(s)'
                        if "lead_time_weeks" in o else "Preseason"
                    ),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "Debrief question: Where did observed sales give you a misleading picture of "
        "true demand, and how did that affect your replenishment decisions?"
    )


ensure_attempt_state()

st.title("👜 Beautiful Bags Inventory Management Simulation")
st.write(
    "Manage three products through a 16-week selling season. Your goal is to maximize "
    "profit while balancing availability, sourcing cost, inventory holding cost, "
    "and the risk of leftover inventory."
)

with st.sidebar:
    st.header("Simulation Controls")
    st.metric("Attempt", f'{st.session_state[ATTEMPT_KEY]} of {MAX_ATTEMPTS}')

    if st.button("Restart Current Attempt", use_container_width=True):
        restart_current_attempt()
        st.rerun()

    can_advance = st.session_state[ATTEMPT_KEY] < MAX_ATTEMPTS
    if st.button("Start New Attempt", use_container_width=True, disabled=not can_advance):
        if start_next_attempt():
            st.rerun()
    if not can_advance:
        st.caption("Maximum of five attempts reached.")

    st.caption("Attempt 1 is the common class scenario. Later attempts use another approved scenario without repetition.")
    st.caption(f"Build: {PAGE_VERSION} / {ENGINE_VERSION}")

sim = st.session_state.get(SIM_KEY)
render_progress(
    current_month=sim.state.current_month if sim else 0,
    completed=sim.state.completed if sim else False,
)

if not st.session_state.get(STARTED_KEY, False):
    st.subheader("Your Assignment")
    st.write(
        "You manage **Blue Jeans**, **What's Next**, and **High Fashion** at the same time. "
        "Before the season begins, choose initial sourcing quantities. After each of the "
        "first three months, review sales and inventory and decide whether to replenish from TJL."
    )

    with st.expander("Product and sourcing information", expanded=True):
        st.dataframe(product_table(), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### China")
        st.write(
            f"- Lower unit cost\n"
            f"- Long lead time; used as the primary preseason source\n"
            f"- MOQ: **{CHINA_MOQ:,} units per product** if used\n"
            f"- Order increment: **{ORDER_INCREMENT} units**\n"
            f"- Product-specific preseason maximums"
        )
    with c2:
        st.markdown("#### TJL")
        st.write(
            f"- Higher unit cost but responsive\n"
            f"- In-season lead time: **1–3 weeks**\n"
            f"- The exact lead time is shown before each replenishment decision\n"
            f"- MOQ: **{TJL_MOQ:,} units per product** if used\n"
            f"- Order increment: **{ORDER_INCREMENT} units**\n"
            f"- In-season shared capacity: **{TJL_MONTHLY_CAPACITY:,} units per month**"
        )

    st.info(
        "During the simulation, you will observe **sales**, not true customer demand. "
        "If inventory reaches zero, additional demand is lost and is not backordered."
    )

    st.subheader("Preseason Sourcing Decision")
    st.write(
        "Choose the quantities that will be available at the beginning of Week 1. "
        "You may use China, TJL, or both."
    )

    with st.form("bb_initial_order_form"):
        st.markdown("#### China initial order")
        china_cols = st.columns(3)
        china = {}
        for col, p in zip(china_cols, PRODUCTS):
            spec = PRODUCT_SPECS[p]
            with col:
                china[p] = order_input(
                    PRODUCT_LABELS[p],
                    f"bb_china_{p}",
                    spec.china_max,
                    f"0 or at least {CHINA_MOQ:,}; maximum {spec.china_max:,}.",
                )

        st.markdown("#### TJL preseason order")
        st.caption(
            "Preseason TJL orders are available at the beginning of Week 1. "
            "They do not use the later monthly replenishment capacity."
        )
        tjl_cols = st.columns(3)
        tjl = {}
        for col, p in zip(tjl_cols, PRODUCTS):
            with col:
                tjl[p] = order_input(
                    PRODUCT_LABELS[p],
                    f"bb_tjl0_{p}",
                    20_000,
                    f"0 or at least {TJL_MOQ:,}; {ORDER_INCREMENT}-unit increments.",
                )

        submitted = st.form_submit_button(
            "Confirm Preseason Orders and Start Season",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            sim = InventorySimulation(
                china_initial=china,
                tjl_initial=tjl,
                scenario_id=st.session_state[CURRENT_SCENARIO_KEY],
            )
            sim.advance_one_month()
            st.session_state[SIM_KEY] = sim
            st.session_state[STARTED_KEY] = True
            st.rerun()
        except DecisionValidationError as e:
            st.error(str(e))

else:
    sim = st.session_state[SIM_KEY]
    current_month = sim.state.current_month
    view = sim.student_month_view(current_month)
    render_month_results(view)

    if not sim.state.completed:
        render_replenishment_form(sim, current_month, view)
    else:
        st.divider()
        render_final_results(sim)
