import streamlit as st

st.set_page_config(
    page_title="SCM 3223 Simulation Lab",
    page_icon="📦",
    layout="wide",
)

# Define pages
home = st.Page(
    "pages/home.py",
    title="Home",
    icon="🏠",
    default=True,
)

forecasting = st.Page(
    "pages/forecasting.py",
    title="Demand Forecasting",
    icon="📈",
)

inventory = st.Page(
    "pages/inventory.py",
    title="Inventory Management",
    icon="📦",
)

order_fulfillment = st.Page(
    "pages/order_fulfillment.py",
    title="Order Fulfillment",
    icon="🛒",
)

distribution = st.Page(
    "pages/distribution.py",
    title="Distribution",
    icon="🏬",
)

transportation = st.Page(
    "pages/transportation.py",
    title="Transportation",
    icon="🚚",
)

integrated = st.Page(
    "pages/integrated_supply_chain.py",
    title="Integrated Supply Chain",
    icon="🌐",
)

# Navigation
pg = st.navigation(
    {
        "SCM Simulation Lab": [home],
        "Simulations": [
            forecasting,
            inventory,
            order_fulfillment,
            distribution,
            transportation,
            integrated,
        ],
    }
)

pg.run()