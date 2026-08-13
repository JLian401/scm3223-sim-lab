import streamlit as st

st.title("SCM 3223 Simulation Lab")

st.subheader("Learn Supply Chain Management by Making Decisions")

st.write(
    """
    Welcome to the SCM 3223 Simulation Lab.

    This platform contains interactive simulations designed to help you
    apply supply chain management concepts through hands-on decision making.

    Select a simulation from the navigation menu to begin.
    """
)

st.divider()

st.header("Simulations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Demand Forecasting")
    st.write(
        "Use historical demand information to make forecasts and "
        "evaluate forecasting accuracy."
    )

    st.subheader("📦 Inventory Management")
    st.write(
        "Make inventory decisions while balancing product availability, "
        "holding costs, and stockouts."
    )

    st.subheader("🛒 Order Fulfillment")
    st.write(
        "Determine how customer orders should be fulfilled while "
        "considering inventory availability, service, and cost."
    )

with col2:
    st.subheader("🏬 Distribution")
    st.write(
        "Manage inventory and product flows across a distribution network."
    )

    st.subheader("🚚 Transportation")
    st.write(
        "Make transportation decisions involving cost, speed, "
        "capacity, and service."
    )

    st.subheader("🌐 Integrated Supply Chain")
    st.write(
        "Manage interconnected forecasting, inventory, fulfillment, "
        "distribution, and transportation decisions."
    )

st.divider()

st.info(
    "The simulations will become available as we progress through the course."
)