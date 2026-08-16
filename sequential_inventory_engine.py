from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, List, Any
import copy
import numpy as np

ENGINE_VERSION = "2026-08-16-scenario-bank-v2"

PRODUCTS = ("blue_jeans", "whats_next", "high_fashion")
PRODUCT_LABELS = {
    "blue_jeans": "Blue Jeans",
    "whats_next": "What's Next",
    "high_fashion": "High Fashion",
}

@dataclass(frozen=True)
class ProductSpec:
    forecast_mean: int
    forecast_sd: int
    china_cost: float
    tjl_cost: float
    regular_revenue: float
    salvage_value: float
    china_max: int

PRODUCT_SPECS: Dict[str, ProductSpec] = {
    "blue_jeans": ProductSpec(10000, 1000, 12.25, 16.75, 33.00, 13.40, 12000),
    "whats_next": ProductSpec(7500, 1250, 14.10, 18.75, 30.00, 10.55, 10000),
    "high_fashion": ProductSpec(5000, 1600, 17.20, 19.75, 37.50, 3.50, 8200),
}

MONTHLY_PROFILES = {
    "blue_jeans": (0.22, 0.25, 0.27, 0.26),
    "whats_next": (0.35, 0.30, 0.22, 0.13),
    "high_fashion": (0.50, 0.30, 0.15, 0.05),
}

# Curated, instructor-approved scenarios. Attempt 1 always uses 3203.
# Seasonal totals were selected to create different learning conditions.
SCENARIO_BANK: Dict[int, Dict[str, Any]] = {
    3203: {
        "description": "Baseline: trend upside, fashion downside",
        "seasonal_demand": {"blue_jeans": 9969, "whats_next": 8226, "high_fashion": 4316},
        "tjl_lead_weeks": {1: 2, 2: 2, 3: 1},
    },
    1009: {
        "description": "Broad upside",
        "seasonal_demand": {"blue_jeans": 10728, "whats_next": 8996, "high_fashion": 6143},
        "tjl_lead_weeks": {1: 3, 2: 2, 3: 1},
    },
    1007: {
        "description": "Broad downside",
        "seasonal_demand": {"blue_jeans": 8925, "whats_next": 5909, "high_fashion": 5205},
        "tjl_lead_weeks": {1: 1, 2: 2, 3: 2},
    },
    1047: {
        "description": "Fashion surprise",
        "seasonal_demand": {"blue_jeans": 9758, "whats_next": 7851, "high_fashion": 6455},
        "tjl_lead_weeks": {1: 2, 2: 1, 3: 3},
    },
    1075: {
        "description": "Trend disappointment",
        "seasonal_demand": {"blue_jeans": 10354, "whats_next": 6475, "high_fashion": 4924},
        "tjl_lead_weeks": {1: 1, 2: 3, 3: 2},
    },
    1068: {
        "description": "Stable-product surprise",
        "seasonal_demand": {"blue_jeans": 11172, "whats_next": 7735, "high_fashion": 5208},
        "tjl_lead_weeks": {1: 3, 2: 1, 3: 2},
    },
    1016: {
        "description": "Mixed difficult",
        "seasonal_demand": {"blue_jeans": 8934, "whats_next": 8667, "high_fashion": 6680},
        "tjl_lead_weeks": {1: 2, 2: 3, 3: 1},
    },
}
APPROVED_SCENARIO_IDS = tuple(SCENARIO_BANK.keys())
DEFAULT_SCENARIO_ID = 3203
MAX_ATTEMPTS = 5

# Preserve the exact baseline weekly path already approved in prior discussion.
BASELINE_WEEKLY_DEMAND: Dict[str, List[int]] = {
    "blue_jeans": [571,566,560,496,684,556,585,667,731,669,641,651,553,632,698,709],
    "whats_next": [663,775,721,720,624,585,573,686,474,498,433,405,247,261,312,249],
    "high_fashion": [617,480,514,547,301,348,303,343,160,165,164,158,54,58,50,54],
}

ANNUAL_HOLDING_RATE = 0.10
WEEKLY_HOLDING_RATE = ANNUAL_HOLDING_RATE / 52.0
CHINA_MOQ = 2000
TJL_MOQ = 500
ORDER_INCREMENT = 100
TJL_MONTHLY_CAPACITY = 2500


class DecisionValidationError(ValueError):
    pass


def _clean_order_dict(order: Optional[Mapping[str, int]]) -> Dict[str, int]:
    order = order or {}
    unknown = set(order) - set(PRODUCTS)
    if unknown:
        raise DecisionValidationError(f"Unknown product key(s): {sorted(unknown)}")
    out = {}
    for p in PRODUCTS:
        q = int(order.get(p, 0))
        if q < 0:
            raise DecisionValidationError(f"{PRODUCT_LABELS[p]} quantity cannot be negative.")
        out[p] = q
    return out


def _validate_increment(qty: int, product: str, source: str) -> None:
    if qty % ORDER_INCREMENT != 0:
        raise DecisionValidationError(
            f"{source} order for {PRODUCT_LABELS[product]} must be in {ORDER_INCREMENT}-unit increments."
        )


def validate_initial_orders(china_orders, tjl_initial_orders=None) -> None:
    china = _clean_order_dict(china_orders)
    tjl = _clean_order_dict(tjl_initial_orders)
    for p in PRODUCTS:
        q = china[p]
        _validate_increment(q, p, "China")
        if q != 0 and q < CHINA_MOQ:
            raise DecisionValidationError(f"China order for {PRODUCT_LABELS[p]} must be 0 or at least {CHINA_MOQ:,}.")
        if q > PRODUCT_SPECS[p].china_max:
            raise DecisionValidationError(
                f"China order for {PRODUCT_LABELS[p]} cannot exceed {PRODUCT_SPECS[p].china_max:,}."
            )
        tq = tjl[p]
        _validate_increment(tq, p, "TJL")
        if tq != 0 and tq < TJL_MOQ:
            raise DecisionValidationError(f"Initial TJL order for {PRODUCT_LABELS[p]} must be 0 or at least {TJL_MOQ:,}.")


def validate_replenishment(month: int, order) -> Dict[str, int]:
    if month not in (1, 2, 3):
        raise DecisionValidationError("TJL replenishment month must be 1, 2, or 3.")
    cleaned = _clean_order_dict(order)
    if sum(cleaned.values()) > TJL_MONTHLY_CAPACITY:
        raise DecisionValidationError(
            f"Month {month} TJL replenishment exceeds the shared {TJL_MONTHLY_CAPACITY:,}-unit capacity."
        )
    for p, q in cleaned.items():
        _validate_increment(q, p, "TJL")
        if q != 0 and q < TJL_MOQ:
            raise DecisionValidationError(
                f"Month {month} TJL order for {PRODUCT_LABELS[p]} must be 0 or at least {TJL_MOQ:,}."
            )
    return cleaned


def _add_inventory(qty_on_hand: float, avg_cost: float, arrival_qty: int, arrival_unit_cost: float):
    if arrival_qty <= 0:
        return qty_on_hand, avg_cost
    total_value = qty_on_hand * avg_cost + arrival_qty * arrival_unit_cost
    new_qty = qty_on_hand + arrival_qty
    return new_qty, total_value / new_qty


def _allocate_total(total: int, shares: List[float]) -> List[int]:
    raw = np.array(shares, dtype=float) * total
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:remainder]:
            base[idx] += 1
    return base.tolist()


def build_weekly_demand(scenario_id: int) -> Dict[str, List[int]]:
    if scenario_id not in SCENARIO_BANK:
        raise ValueError(f"Unknown scenario_id {scenario_id}")
    if scenario_id == DEFAULT_SCENARIO_ID:
        return copy.deepcopy(BASELINE_WEEKLY_DEMAND)

    rng = np.random.default_rng(scenario_id)
    weekly = {}
    for p in PRODUCTS:
        seasonal_total = SCENARIO_BANK[scenario_id]["seasonal_demand"][p]
        month_totals = _allocate_total(seasonal_total, list(MONTHLY_PROFILES[p]))
        product_weeks: List[int] = []
        for m_total in month_totals:
            # Small weekly variation around equal quarter shares; normalized to sum to 1.
            weights = rng.uniform(0.88, 1.12, size=4)
            weights = weights / weights.sum()
            product_weeks.extend(_allocate_total(m_total, weights.tolist()))
        weekly[p] = product_weeks
    return weekly


@dataclass
class SimulationState:
    scenario_id: int = DEFAULT_SCENARIO_ID
    current_week: int = 0
    current_month: int = 0
    on_hand: Dict[str, float] = field(default_factory=dict)
    avg_cost: Dict[str, float] = field(default_factory=dict)
    revenue: float = 0.0
    sourcing_cost_committed: float = 0.0
    holding_cost: float = 0.0
    cumulative_sales: Dict[str, int] = field(default_factory=dict)
    cumulative_lost_sales: Dict[str, int] = field(default_factory=dict)
    weekly_records: List[Dict[str, Any]] = field(default_factory=list)
    month_end_summaries: List[Dict[str, Any]] = field(default_factory=list)
    order_ledger: List[Dict[str, Any]] = field(default_factory=list)
    pending_arrivals: Dict[int, Dict[str, int]] = field(default_factory=dict)
    completed: bool = False


class InventorySimulation:
    """Sequential engine intended for Streamlit session_state."""

    def __init__(self, china_initial, tjl_initial=None, scenario_id: int = DEFAULT_SCENARIO_ID):
        if scenario_id not in SCENARIO_BANK:
            raise DecisionValidationError(f"Scenario {scenario_id} is not approved.")
        china = _clean_order_dict(china_initial)
        tjl0 = _clean_order_dict(tjl_initial)
        validate_initial_orders(china, tjl0)

        self.scenario_id = scenario_id
        self.weekly_demand = build_weekly_demand(scenario_id)
        self.seasonal_demand = {p: sum(self.weekly_demand[p]) for p in PRODUCTS}
        self.tjl_lead_weeks = copy.deepcopy(SCENARIO_BANK[scenario_id]["tjl_lead_weeks"])

        self.state = SimulationState(
            scenario_id=scenario_id,
            on_hand={p: 0.0 for p in PRODUCTS},
            avg_cost={p: 0.0 for p in PRODUCTS},
            cumulative_sales={p: 0 for p in PRODUCTS},
            cumulative_lost_sales={p: 0 for p in PRODUCTS},
            pending_arrivals={w: {p: 0 for p in PRODUCTS} for w in range(1, 17)},
        )

        for p in PRODUCTS:
            self.state.on_hand[p], self.state.avg_cost[p] = _add_inventory(
                self.state.on_hand[p], self.state.avg_cost[p], china[p], PRODUCT_SPECS[p].china_cost
            )
            self.state.on_hand[p], self.state.avg_cost[p] = _add_inventory(
                self.state.on_hand[p], self.state.avg_cost[p], tjl0[p], PRODUCT_SPECS[p].tjl_cost
            )
            if china[p]:
                self.state.order_ledger.append({
                    "stage": "preseason", "source": "China", "product": p,
                    "quantity": china[p], "unit_cost": PRODUCT_SPECS[p].china_cost,
                    "placement_week": 0, "arrival_week": 1,
                })
            if tjl0[p]:
                self.state.order_ledger.append({
                    "stage": "preseason", "source": "TJL", "product": p,
                    "quantity": tjl0[p], "unit_cost": PRODUCT_SPECS[p].tjl_cost,
                    "placement_week": 0, "arrival_week": 1,
                })

        self.state.sourcing_cost_committed = (
            sum(china[p] * PRODUCT_SPECS[p].china_cost for p in PRODUCTS)
            + sum(tjl0[p] * PRODUCT_SPECS[p].tjl_cost for p in PRODUCTS)
        )

    def lead_time_for_decision(self, month: int) -> int:
        if month not in (1, 2, 3):
            raise DecisionValidationError("Lead time is available only for Months 1-3 replenishment decisions.")
        return int(self.tjl_lead_weeks[month])

    def place_replenishment(self, month: int, order) -> Dict[str, int]:
        if self.state.completed:
            raise DecisionValidationError("Simulation is already complete.")
        if month != self.state.current_month:
            raise DecisionValidationError(
                f"Replenishment must be placed after Month {self.state.current_month}, not Month {month}."
            )
        if month not in (1, 2, 3):
            raise DecisionValidationError("No replenishment decision is allowed after Month 4.")
        if any(o.get("decision_month") == month for o in self.state.order_ledger):
            raise DecisionValidationError(f"Month {month} replenishment has already been submitted.")

        cleaned = validate_replenishment(month, order)
        lead_weeks = self.lead_time_for_decision(month)
        # Order is placed at the end of week month*4. A 1-week lead time means
        # one full waiting week, then inventory is available at the start of the next week.
        arrival_week = month * 4 + lead_weeks + 1

        for p, q in cleaned.items():
            if q:
                self.state.pending_arrivals[arrival_week][p] += q
                cost = q * PRODUCT_SPECS[p].tjl_cost
                self.state.sourcing_cost_committed += cost
                self.state.order_ledger.append({
                    "stage": f"after_month_{month}", "decision_month": month,
                    "source": "TJL", "product": p, "quantity": q,
                    "unit_cost": PRODUCT_SPECS[p].tjl_cost,
                    "placement_week": month * 4,
                    "lead_time_weeks": lead_weeks,
                    "arrival_week": arrival_week,
                })
        return cleaned

    def advance_one_month(self) -> Dict[str, Any]:
        if self.state.completed:
            raise DecisionValidationError("Simulation is already complete.")

        target_month = self.state.current_month + 1
        start_week = self.state.current_week + 1
        end_week = target_month * 4

        for week in range(start_week, end_week + 1):
            for p in PRODUCTS:
                arrival_qty = self.state.pending_arrivals[week][p]
                if arrival_qty:
                    self.state.on_hand[p], self.state.avg_cost[p] = _add_inventory(
                        self.state.on_hand[p], self.state.avg_cost[p], arrival_qty, PRODUCT_SPECS[p].tjl_cost
                    )

                beginning_inventory = self.state.on_hand[p]
                demand = self.weekly_demand[p][week - 1]
                sales = int(min(demand, beginning_inventory))
                lost_sales = int(demand - sales)
                ending_inventory = beginning_inventory - sales
                average_weekly_inventory = (beginning_inventory + ending_inventory) / 2.0
                week_holding_cost = average_weekly_inventory * self.state.avg_cost[p] * WEEKLY_HOLDING_RATE
                week_revenue = sales * PRODUCT_SPECS[p].regular_revenue

                self.state.revenue += week_revenue
                self.state.holding_cost += week_holding_cost
                self.state.cumulative_sales[p] += sales
                self.state.cumulative_lost_sales[p] += lost_sales
                self.state.on_hand[p] = ending_inventory

                self.state.weekly_records.append({
                    "week": week, "month": target_month, "product": p,
                    "product_label": PRODUCT_LABELS[p], "arrival_qty": arrival_qty,
                    "beginning_inventory": beginning_inventory, "true_demand": demand,
                    "sales": sales, "lost_sales": lost_sales,
                    "ending_inventory": ending_inventory,
                    "inventory_avg_unit_cost": self.state.avg_cost[p],
                    "average_weekly_inventory": average_weekly_inventory,
                    "holding_cost": week_holding_cost, "sales_revenue": week_revenue,
                })

        self.state.current_week = end_week
        self.state.current_month = target_month

        month_records = [r for r in self.state.weekly_records if r["month"] == target_month]
        per_product = {}
        for p in PRODUCTS:
            pr = [r for r in month_records if r["product"] == p]
            per_product[p] = {
                "sales": sum(r["sales"] for r in pr),
                "true_demand": sum(r["true_demand"] for r in pr),
                "lost_sales": sum(r["lost_sales"] for r in pr),
                "ending_inventory": self.state.on_hand[p],
            }

        summary = {
            "month": target_month,
            "per_product": copy.deepcopy(per_product),
            "sales_revenue_to_date": self.state.revenue,
            "sourcing_cost_committed": self.state.sourcing_cost_committed,
            "holding_cost_to_date": self.state.holding_cost,
            "interim_profit": self.state.revenue - self.state.sourcing_cost_committed - self.state.holding_cost,
        }
        self.state.month_end_summaries.append(summary)

        if target_month == 4:
            self.state.completed = True

        return self.student_month_view(target_month)

    def student_month_view(self, month: Optional[int] = None) -> Dict[str, Any]:
        if not self.state.month_end_summaries:
            raise DecisionValidationError("No month has been simulated yet.")
        month = self.state.current_month if month is None else month
        if month < 1 or month > self.state.current_month:
            raise DecisionValidationError("That month is not yet available.")

        summary = self.state.month_end_summaries[month - 1]
        visible_products = {
            p: {
                "sales": summary["per_product"][p]["sales"],
                "ending_inventory": summary["per_product"][p]["ending_inventory"],
            }
            for p in PRODUCTS
        }
        end_week = month * 4
        outstanding = [
            copy.deepcopy(o) for o in self.state.order_ledger
            if o["source"] == "TJL"
            and o.get("placement_week", 0) <= end_week
            and o.get("arrival_week", 1) > end_week
        ]
        view = {
            "month": month,
            "products": visible_products,
            "outstanding_tjl_orders": outstanding,
            "sales_revenue_to_date": summary["sales_revenue_to_date"],
            "sourcing_cost_committed": summary["sourcing_cost_committed"],
            "holding_cost_to_date": summary["holding_cost_to_date"],
            "interim_profit": summary["interim_profit"],
            "note": "Interim profit excludes the end-of-season value of remaining inventory. Final profit is calculated after Month 4.",
        }
        if month in (1, 2, 3):
            view["next_tjl_lead_time_weeks"] = self.lead_time_for_decision(month)
        return view

    def final_results(self) -> Dict[str, Any]:
        if not self.state.completed:
            raise DecisionValidationError("Final results are available only after Month 4.")
        salvage_revenue = sum(
            self.state.on_hand[p] * PRODUCT_SPECS[p].salvage_value for p in PRODUCTS
        )
        final_profit = (
            self.state.revenue + salvage_revenue
            - self.state.sourcing_cost_committed - self.state.holding_cost
        )
        return {
            "scenario_id": self.scenario_id,
            "sales_revenue": self.state.revenue,
            "sourcing_cost": self.state.sourcing_cost_committed,
            "holding_cost": self.state.holding_cost,
            "salvage_revenue": salvage_revenue,
            "final_profit": final_profit,
            "ending_inventory": copy.deepcopy(self.state.on_hand),
            "units_sold": copy.deepcopy(self.state.cumulative_sales),
            "lost_sales": copy.deepcopy(self.state.cumulative_lost_sales),
            "stockout_occurred": {p: self.state.cumulative_lost_sales[p] > 0 for p in PRODUCTS},
            "seasonal_true_demand": copy.deepcopy(self.seasonal_demand),
        }

    def instructor_state(self) -> Dict[str, Any]:
        out = copy.deepcopy(self.state.__dict__)
        out["weekly_demand"] = copy.deepcopy(self.weekly_demand)
        out["seasonal_demand"] = copy.deepcopy(self.seasonal_demand)
        out["tjl_lead_weeks"] = copy.deepcopy(self.tjl_lead_weeks)
        return out
