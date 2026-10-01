import streamlit as st
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# ============================================================
# Sooner Bicycle Order Management Simulation — Version 1
# ============================================================

st.set_page_config(page_title="Sooner Bicycle Order Management", layout="wide")

# -----------------------------
# Core parameters
# -----------------------------
INITIAL_INVENTORY = 100
BASE_STOCK_TARGET = 100
UNIT_PRODUCTION_COST = 92.0
STANDARD_SHIP_COST = 3.0
PREMIUM_SHIP_COST = 10.0
STANDARD_TRANSIT_DAYS = 10
PREMIUM_TRANSIT_DAYS = 4
ANNUAL_HOLDING_RATE = 0.20
DAILY_HOLDING_COST = UNIT_PRODUCTION_COST * ANNUAL_HOLDING_RATE / 365
LATE_PENALTY_RATE = 0.05
MEAN_DAILY_PRODUCTION = 28
PRODUCTION_RANGE = (24, 32)
ACTIVE_WEEKS = 13
DISPLAY_WEEKS = 15

# Fixed daily production draws used on productive days only.
# Values are intentionally centered near 28 and are identical for all teams.
PRODUCTION_SEQUENCE = [
    28, 30, 26, 29, 27, 31, 25, 28, 29, 27,
    30, 28, 26, 32, 27, 29, 28, 25, 30, 27,
    28, 31, 26, 29, 28, 27, 30, 24, 29, 28,
    31, 27, 26, 30, 28, 29, 25, 32, 27, 28,
    30, 26, 29, 28, 27, 31, 25, 30, 28, 29,
    26, 28, 32, 27, 29, 28, 25, 30, 27, 31,
    28, 26, 29, 30, 27, 28, 31, 25, 29, 28,
    30, 26, 28, 32, 27, 29, 25, 30, 28, 27,
    31, 26, 29, 28, 30, 25, 27, 32, 28, 29,
    26, 30, 28, 27, 31, 25, 29, 28, 30, 26,
]

# Fixed Monday-morning inventory count adjustments.
INVENTORY_ADJUSTMENTS = {
    1: 0, 2: 1, 3: 0, 4: 2, 5: 1,
    6: 0, 7: 2, 8: 1, 9: 0, 10: 3,
    11: 1, 12: 0, 13: 2, 14: 1, 15: 0,
}

# RFQ table. Hidden thresholds are intentionally modest deviations from stated requests.
RFQS = [
    dict(id="R01", week=1, customer="Sooner Cycles", ctype="Regular", qty=150, target=118, req_lt=24, max_price=123, max_lt=30),
    dict(id="R02", week=2, customer="Value Bikes", ctype="Discount", qty=250, target=108, req_lt=31, max_price=110, max_lt=40),
    dict(id="R03", week=2, customer="Elite Cycling", ctype="Premium", qty=100, target=None, req_lt=18, max_price=132, max_lt=21),
    dict(id="R04", week=3, customer="Sooner Cycles", ctype="Regular", qty=175, target=116, req_lt=25, max_price=121, max_lt=31),
    dict(id="R05", week=5, customer="Campus Events", ctype="Urgent", qty=75, target=136, req_lt=10, max_price=145, max_lt=11),
    dict(id="R06", week=5, customer="Value Bikes", ctype="Discount", qty=300, target=105, req_lt=35, max_price=108, max_lt=44),
    dict(id="R07", week=6, customer="Elite Cycling", ctype="Premium", qty=125, target=126, req_lt=20, max_price=134, max_lt=23),
    dict(id="R08", week=7, customer="Red River Bikes", ctype="Regular", qty=125, target=None, req_lt=21, max_price=124, max_lt=27),
    dict(id="R09", week=9, customer="Budget Wheel", ctype="Discount", qty=225, target=110, req_lt=29, max_price=112, max_lt=38),
    dict(id="R10", week=9, customer="Norman Sports Event", ctype="Urgent", qty=50, target=None, req_lt=8, max_price=146, max_lt=9),
    dict(id="R11", week=10, customer="Elite Cycling", ctype="Premium", qty=150, target=123, req_lt=22, max_price=131, max_lt=25),
    dict(id="R12", week=11, customer="Sooner Cycles", ctype="Regular", qty=200, target=114, req_lt=28, max_price=119, max_lt=34),
    dict(id="R13", week=13, customer="Campus Events", ctype="Urgent", qty=100, target=132, req_lt=12, max_price=141, max_lt=13),
    dict(id="R14", week=13, customer="Value Bikes", ctype="Discount", qty=275, target=107, req_lt=33, max_price=109, max_lt=42),
]

@dataclass
class Order:
    order_id: str
    rfq_id: str
    customer: str
    ctype: str
    qty: int
    price: float
    quote_day: int
    promised_day: int
    reserved_qty: int
    production_needed: int
    produced_for_order: int = 0
    full_ready_day: Optional[int] = None
    ship_day: Optional[int] = None
    delivery_day: Optional[int] = None
    ship_mode: Optional[str] = None
    late_days: int = 0
    holding_cost: float = 0.0
    late_penalty: float = 0.0
    transport_cost: float = 0.0
    status: str = "Open"


def init_state():
    defaults = {
        "week": 1,
        "day": 1,
        "available_inventory": INITIAL_INVENTORY,
        "orders": [],
        "quotes": {},
        "processed_rfq_ids": set(),
        "production_queue": [],
        "current_order_id": None,
        "setup_days_remaining": 0,
        "initial_setup_done": False,
        "prod_idx": 0,
        "holding_cost_total": 0.0,
        "revenue_total": 0.0,
        "transport_cost_total": 0.0,
        "late_penalty_total": 0.0,
        "inventory_adjustment_total": 0,
        "week_log": [],
        "week_started": False,
        "simulation_finished": False,
        "last_quote_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_orders() -> List[Order]:
    return st.session_state.orders


def find_order(order_id: str) -> Optional[Order]:
    for o in get_orders():
        if o.order_id == order_id:
            return o
    return None


def monday_adjustment():
    wk = st.session_state.week
    loss = INVENTORY_ADJUSTMENTS.get(wk, 0)
    actual_loss = min(loss, st.session_state.available_inventory)
    st.session_state.available_inventory -= actual_loss
    st.session_state.inventory_adjustment_total += actual_loss
    return actual_loss


def total_reserved_inventory():
    total = 0
    for o in get_orders():
        if o.status not in ("Delivered", "Delivered Late"):
            # Units already ready/reserved for this order remain physically held until shipment.
            total += o.reserved_qty + o.produced_for_order
    return total


def production_backlog():
    return sum(max(0, o.production_needed - o.produced_for_order) for o in get_orders() if o.status not in ("Delivered", "Delivered Late"))


def current_production_status():
    if st.session_state.setup_days_remaining > 0:
        return "Setup / changeover"
    if st.session_state.current_order_id:
        o = find_order(st.session_state.current_order_id)
        return f"Producing {o.order_id} ({o.customer})" if o else "Production"
    if st.session_state.production_queue:
        return "Waiting to start next order"
    if st.session_state.available_inventory < BASE_STOCK_TARGET:
        return "Producing for inventory"
    return "Stopped — inventory target reached"


def accept_quote(rfq: Dict, price: float, promised_lt: int):
    quote_day = st.session_state.day
    promised_day = quote_day + promised_lt
    accepted = (price <= rfq["max_price"]) and (promised_lt <= rfq["max_lt"])
    result_record = {
        "rfq_id": rfq["id"],
        "customer": rfq["customer"],
        "ctype": rfq["ctype"],
        "qty": rfq["qty"],
        "price": price,
        "promised_lt": promised_lt,
        "accepted": accepted,
        "quote_day": quote_day,
        "order_id": None,
        "reserved_qty": 0,
        "production_needed": 0,
    }
    st.session_state.quotes[rfq["id"]] = result_record.copy()
    st.session_state.processed_rfq_ids.add(rfq["id"])

    if accepted:
        reserve = min(st.session_state.available_inventory, rfq["qty"])
        st.session_state.available_inventory -= reserve
        need_prod = rfq["qty"] - reserve
        order = Order(
            order_id=f"O{len(get_orders())+1:02d}",
            rfq_id=rfq["id"],
            customer=rfq["customer"],
            ctype=rfq["ctype"],
            qty=rfq["qty"],
            price=price,
            quote_day=quote_day,
            promised_day=promised_day,
            reserved_qty=reserve,
            production_needed=need_prod,
            status="Reserved" if need_prod > 0 else "Ready",
        )
        if need_prod == 0:
            order.full_ready_day = quote_day
        else:
            st.session_state.production_queue.append(order.order_id)
            # First customer order that needs production triggers a one-day initial setup.
            if not st.session_state.initial_setup_done and st.session_state.current_order_id is None:
                st.session_state.setup_days_remaining = 1
                st.session_state.initial_setup_done = True
        st.session_state.orders.append(order)
        result_record["order_id"] = order.order_id
        result_record["reserved_qty"] = reserve
        result_record["production_needed"] = need_prod
    st.session_state.quotes[rfq["id"]] = result_record.copy()
    st.session_state.last_quote_results.append(result_record)
    return accepted


def next_prod_draw():
    idx = st.session_state.prod_idx
    value = PRODUCTION_SEQUENCE[idx % len(PRODUCTION_SEQUENCE)]
    st.session_state.prod_idx += 1
    return value


def allocate_production_to_order(order: Order, qty: int, day: int):
    remaining = order.production_needed - order.produced_for_order
    use = min(qty, remaining)
    order.produced_for_order += use
    extra = qty - use

    if order.produced_for_order >= order.production_needed:
        order.full_ready_day = day
        order.status = "Ready"
        st.session_state.current_order_id = None
        # Remove from queue if still present
        if order.order_id in st.session_state.production_queue:
            st.session_state.production_queue.remove(order.order_id)
        # One day changeover before next customer order if one exists.
        if st.session_state.production_queue:
            st.session_state.setup_days_remaining = 1
    return extra


def ship_ready_orders(day: int):
    # Shipping eligibility begins the day after full quantity is ready.
    for o in get_orders():
        if o.status == "Ready" and o.full_ready_day is not None and day >= o.full_ready_day + 1:
            standard_delivery = day + STANDARD_TRANSIT_DAYS
            premium_delivery = day + PREMIUM_TRANSIT_DAYS
            if standard_delivery <= o.promised_day:
                mode = "Standard"
                delivery = standard_delivery
                cost_per = STANDARD_SHIP_COST
            else:
                mode = "Premium"
                delivery = premium_delivery
                cost_per = PREMIUM_SHIP_COST

            o.ship_day = day
            o.delivery_day = delivery
            o.ship_mode = mode
            o.transport_cost = cost_per * o.qty
            st.session_state.transport_cost_total += o.transport_cost
            o.late_days = max(0, delivery - o.promised_day)
            if o.late_days > 0:
                o.late_penalty = o.qty * o.price * LATE_PENALTY_RATE * o.late_days
                st.session_state.late_penalty_total += o.late_penalty
            o.status = "In Transit"


def deliver_orders(day: int):
    for o in get_orders():
        if o.status == "In Transit" and o.delivery_day == day:
            o.status = "Delivered Late" if o.late_days > 0 else "Delivered"
            st.session_state.revenue_total += o.qty * o.price


def accrue_holding_cost():
    # Available + reserved finished goods still physically held.
    held_units = st.session_state.available_inventory + total_reserved_inventory()
    daily_cost = held_units * DAILY_HOLDING_COST
    st.session_state.holding_cost_total += daily_cost
    # Attribute reserved inventory holding cost to open orders proportionally by units held.
    for o in get_orders():
        if o.status not in ("Delivered", "Delivered Late", "In Transit"):
            units = o.reserved_qty + o.produced_for_order
            o.holding_cost += units * DAILY_HOLDING_COST


def run_production_day(day: int):
    # Setup day: no output.
    if st.session_state.setup_days_remaining > 0:
        st.session_state.setup_days_remaining -= 1
        return 0, "Setup"

    # Start next customer order if queued.
    if st.session_state.current_order_id is None and st.session_state.production_queue:
        st.session_state.current_order_id = st.session_state.production_queue[0]

    # Customer backlog has priority.
    if st.session_state.current_order_id:
        o = find_order(st.session_state.current_order_id)
        if o:
            draw = next_prod_draw()
            extra = allocate_production_to_order(o, draw, day)
            # If a day's output exceeds what was needed for the order, excess becomes available stock.
            if extra > 0:
                st.session_state.available_inventory += extra
            return draw, f"Order {o.order_id}"

    # No customer backlog: replenish available inventory to target.
    if st.session_state.available_inventory < BASE_STOCK_TARGET:
        draw = next_prod_draw()
        actual = min(draw, BASE_STOCK_TARGET - st.session_state.available_inventory)
        st.session_state.available_inventory += actual
        return actual, "Inventory"

    return 0, "Stopped"


def advance_one_week():
    wk = st.session_state.week
    start_day = st.session_state.day
    produced = 0
    setup_days = 0
    modes_before = sum(1 for o in get_orders() if o.ship_mode == "Premium")
    late_before = sum(1 for o in get_orders() if o.late_days > 0)
    delivered_before = sum(1 for o in get_orders() if o.status in ("Delivered", "Delivered Late"))

    # Five production days. Monday is start_day.
    for offset in range(5):
        day = start_day + offset
        deliver_orders(day)
        ship_ready_orders(day)
        out, kind = run_production_day(day)
        produced += out
        if kind == "Setup":
            setup_days += 1
        accrue_holding_cost()

    # Weekend: no production, but shipments can arrive and holding costs accrue.
    for offset in (5, 6):
        day = start_day + offset
        deliver_orders(day)
        ship_ready_orders(day)
        accrue_holding_cost()

    st.session_state.week_log.append({
        "week": wk,
        "produced": produced,
        "setup_days": setup_days,
        "premium_shipments": sum(1 for o in get_orders() if o.ship_mode == "Premium") - modes_before,
        "late_shipments": sum(1 for o in get_orders() if o.late_days > 0) - late_before,
        "deliveries": sum(1 for o in get_orders() if o.status in ("Delivered", "Delivered Late")) - delivered_before,
        "ending_available_inventory": st.session_state.available_inventory,
        "profit": current_profit(),
    })

    st.session_state.day += 7
    st.session_state.week += 1
    st.session_state.week_started = False

    if st.session_state.week > DISPLAY_WEEKS:
        # Continue automatically until all accepted orders are delivered.
        safety = 0
        while any(o.status not in ("Delivered", "Delivered Late") for o in get_orders()) and safety < 40:
            start = st.session_state.day
            for offset in range(7):
                day = start + offset
                deliver_orders(day)
                ship_ready_orders(day)
                if offset < 5:
                    run_production_day(day)
                accrue_holding_cost()
            st.session_state.day += 7
            safety += 1
        st.session_state.simulation_finished = True


def current_profit():
    return (
        st.session_state.revenue_total
        - sum(o.qty * UNIT_PRODUCTION_COST for o in get_orders() if o.status in ("Delivered", "Delivered Late"))
        - st.session_state.transport_cost_total
        - st.session_state.holding_cost_total
        - st.session_state.late_penalty_total
    )


def on_time_rate():
    delivered = [o for o in get_orders() if o.status in ("Delivered", "Delivered Late")]
    if not delivered:
        return 0.0
    ontime = sum(1 for o in delivered if o.late_days == 0)
    return ontime / len(delivered)


def rfqs_this_week():
    return [r for r in RFQS if r["week"] == st.session_state.week]


def start_week_if_needed():
    if not st.session_state.week_started and st.session_state.week <= DISPLAY_WEEKS:
        monday_adjustment()
        st.session_state.week_started = True


def final_report():
    st.header("Final Report")
    delivered = [o for o in get_orders() if o.status in ("Delivered", "Delivered Late")]
    accepted = len(get_orders())
    total_quotes = len(st.session_state.quotes)
    units = sum(o.qty for o in get_orders())
    avg_price = sum(o.qty * o.price for o in get_orders()) / units if units else 0
    revenue = st.session_state.revenue_total
    profit = current_profit()
    margin = profit / revenue if revenue else 0

    a,b,c,d = st.columns(4)
    a.metric("Final Profit", f"${profit:,.0f}")
    b.metric("On-Time Delivery", f"{on_time_rate()*100:.1f}%")
    c.metric("Quote Acceptance", f"{(accepted/total_quotes*100 if total_quotes else 0):.1f}%")
    d.metric("Profit Margin", f"{margin*100:.1f}%")

    st.subheader("Commercial Performance")
    st.dataframe([
        {"Metric": "RFQs received", "Value": total_quotes},
        {"Metric": "Quotations accepted", "Value": accepted},
        {"Metric": "Total units sold", "Value": units},
        {"Metric": "Average selling price", "Value": f"${avg_price:,.2f}"},
        {"Metric": "Revenue", "Value": f"${revenue:,.2f}"},
    ], use_container_width=True, hide_index=True)

    st.subheader("Operational Performance")
    late_count = sum(1 for o in delivered if o.late_days > 0)
    avg_late = sum(o.late_days for o in delivered if o.late_days > 0) / max(1, late_count)
    st.dataframe([
        {"Metric": "Orders delivered", "Value": len(delivered)},
        {"Metric": "On-time delivery rate", "Value": f"{on_time_rate()*100:.1f}%"},
        {"Metric": "Standard shipments", "Value": sum(1 for o in delivered if o.ship_mode == "Standard")},
        {"Metric": "Premium shipments", "Value": sum(1 for o in delivered if o.ship_mode == "Premium")},
        {"Metric": "Late deliveries", "Value": late_count},
        {"Metric": "Average days late (late orders only)", "Value": f"{avg_late:.2f}"},
        {"Metric": "Ending available inventory", "Value": st.session_state.available_inventory},
    ], use_container_width=True, hide_index=True)

    st.subheader("Financial Performance")
    st.dataframe([
        {"Metric": "Revenue", "Value": f"${st.session_state.revenue_total:,.2f}"},
        {"Metric": "Production cost", "Value": f"${sum(o.qty * UNIT_PRODUCTION_COST for o in delivered):,.2f}"},
        {"Metric": "Transportation cost", "Value": f"${st.session_state.transport_cost_total:,.2f}"},
        {"Metric": "Inventory holding cost", "Value": f"${st.session_state.holding_cost_total:,.2f}"},
        {"Metric": "Late penalties", "Value": f"${st.session_state.late_penalty_total:,.2f}"},
        {"Metric": "Final profit", "Value": f"${profit:,.2f}"},
    ], use_container_width=True, hide_index=True)

    st.subheader("Order-Level Results")
    rows = []
    for o in get_orders():
        order_profit = o.qty*o.price - o.qty*UNIT_PRODUCTION_COST - o.transport_cost - o.holding_cost - o.late_penalty
        rows.append({
            "Order": o.order_id,
            "Customer": o.customer,
            "Type": o.ctype,
            "Qty": o.qty,
            "Price": o.price,
            "Promised Day": o.promised_day,
            "Actual Delivery": o.delivery_day,
            "Mode": o.ship_mode,
            "Late Days": o.late_days,
            "Order Profit": round(order_profit, 2),
        })
    st.dataframe(rows, use_container_width=True)

    with st.expander("Instructor reflection view: hidden customer thresholds"):
        st.dataframe([{**r, "target": r["target"] if r["target"] is not None else "Not disclosed"} for r in RFQS], use_container_width=True)


def reset_game():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# -----------------------------
# UI
# -----------------------------
init_state()

st.title("Sooner Bicycle Order Management Simulation")
st.caption("Boomer Bike — Order Management Team")

with st.sidebar:
    st.subheader("Operating Rules")
    st.write(f"Initial inventory: **{INITIAL_INVENTORY} bikes**")
    st.write(f"Production: **mean {MEAN_DAILY_PRODUCTION}/day**, typical range **{PRODUCTION_RANGE[0]}–{PRODUCTION_RANGE[1]}**")
    st.write("Production scheduling: **FCFS**")
    st.write("Setup/changeover: **1 production day**")
    st.write(f"Inventory target when no backlog: **{BASE_STOCK_TARGET} bikes**")
    st.write(f"Standard shipping: **{STANDARD_TRANSIT_DAYS} days, ${STANDARD_SHIP_COST:.0f}/bike**")
    st.write(f"Premium shipping: **{PREMIUM_TRANSIT_DAYS} days, ${PREMIUM_SHIP_COST:.0f}/bike**")
    st.write(f"Late penalty: **{LATE_PENALTY_RATE*100:.0f}% of order revenue per late day**")
    st.write(f"Holding rate: **{ANNUAL_HOLDING_RATE*100:.0f}% annually**")
    if st.button("Reset Simulation"):
        reset_game()

if st.session_state.simulation_finished:
    final_report()
    st.stop()

start_week_if_needed()

st.header(f"Week {st.session_state.week}")
st.write(f"Monday morning — Simulation Day {st.session_state.day}")

# Dashboard
c1, c2, c3, c4 = st.columns(4)
c1.metric("Available Inventory", f"{st.session_state.available_inventory} bikes")
c2.metric("Reserved Inventory", f"{total_reserved_inventory()} bikes")
c3.metric("Production Backlog", f"{production_backlog()} bikes")
c4.metric("Production Status", current_production_status())

p1, p2, p3, p4 = st.columns(4)
p1.metric("Orders Won", len(get_orders()))
p2.metric("Revenue", f"${st.session_state.revenue_total:,.0f}")
p3.metric("Profit", f"${current_profit():,.0f}")
p4.metric("On-Time Delivery", f"{on_time_rate()*100:.1f}%")

st.subheader("Open Orders")
open_rows = []
for o in get_orders():
    if o.status not in ("Delivered", "Delivered Late"):
        open_rows.append({
            "Order": o.order_id,
            "Customer": o.customer,
            "Qty": o.qty,
            "Price": o.price,
            "Promised Day": o.promised_day,
            "Status": o.status,
            "Reserved/Produced": o.reserved_qty + o.produced_for_order,
            "Remaining Production": max(0, o.production_needed - o.produced_for_order),
        })
if open_rows:
    st.dataframe(open_rows, use_container_width=True, hide_index=True)
else:
    st.info("No open customer orders.")

st.subheader("This Week's RFQs")
rfqs = rfqs_this_week()
week_result_ids = {r["id"] for r in rfqs}
recent_results = [x for x in st.session_state.last_quote_results if x["rfq_id"] in week_result_ids]
for res in recent_results:
    if res["accepted"]:
        st.success(
            f"✅ ORDER WON — {res['customer']} accepted your quote. "
            f"Confirmed order {res['order_id']}: {res['qty']} bikes at ${res['price']:.0f}/bike, "
            f"promised in {res['promised_lt']} days. "
            f"Reserved inventory: {res['reserved_qty']} bikes; production needed: {res['production_needed']} bikes."
        )
    else:
        st.error(
            f"❌ ORDER LOST — {res['customer']} declined your quote of "
            f"${res['price']:.0f}/bike with delivery in {res['promised_lt']} days."
        )
unprocessed = [r for r in rfqs if r["id"] not in st.session_state.processed_rfq_ids]
if not rfqs:
    st.info("No new RFQs this week.")
elif not unprocessed:
    st.success("All RFQs for this week have been quoted.")
else:
    for r in unprocessed:
        with st.container(border=True):
            st.markdown(f"### {r['customer']} — {r['ctype']} Customer")
            q1, q2, q3 = st.columns(3)
            q1.metric("Requested Quantity", f"{r['qty']} bikes")
            q2.metric("Target Price", "Not disclosed" if r["target"] is None else f"${r['target']}/bike")
            q3.metric("Requested Delivery", f"Within {r['req_lt']} days")
            price = st.number_input(f"Quoted price per bike — {r['id']}", min_value=90.0, max_value=180.0, value=float((r['target'] or 120)), step=1.0, key=f"price_{r['id']}")
            promised_lt = st.number_input(f"Promised delivery time in days — {r['id']}", min_value=1, max_value=60, value=int(r['req_lt']), step=1, key=f"lt_{r['id']}")
            if st.button(f"Submit Quote — {r['id']}", key=f"submit_{r['id']}"):
                accepted = accept_quote(r, price, promised_lt)
                if accepted:
                    st.success("Quotation accepted. The order has been confirmed and resources reserved.")
                else:
                    st.error("Quotation declined.")
                st.rerun()

# Weekly results to date
if st.session_state.week_log:
    last = st.session_state.week_log[-1]
    with st.expander(f"Previous Week Summary — Week {last['week']}"):
        st.write(last)

# Advance only after all current RFQs processed.
all_done = all(r["id"] in st.session_state.processed_rfq_ids for r in rfqs)
if all_done:
    if st.session_state.week <= DISPLAY_WEEKS:
        if st.button("Run This Week and Continue", type="primary"):
            advance_one_week()
            st.rerun()
