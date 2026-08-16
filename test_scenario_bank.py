from sequential_inventory_engine import (
    SCENARIO_BANK, PRODUCTS, InventorySimulation
)


def test_scenario_bank_structure():
    assert len(SCENARIO_BANK) == 7
    for scenario in SCENARIO_BANK.values():
        assert all(len(scenario.weekly_demand[p]) == 16 for p in PRODUCTS)
        assert all(scenario.tjl_lead_time_weeks[m] in (1, 2, 3) for m in (1, 2, 3))


def test_common_first_attempt_reference_path():
    sim = InventorySimulation(
        china_initial={"blue_jeans":10000,"whats_next":7500,"high_fashion":5000},
        scenario_id=3203,
    )
    sim.advance_one_month()
    assert sim.replenishment_lead_time_weeks(1) == 2
    assert sim.expected_arrival_week(1) == 7
    sim.place_replenishment(1, {"whats_next":500})
    sim.advance_one_month()
    sim.place_replenishment(2, {})
    sim.advance_one_month()
    sim.place_replenishment(3, {})
    sim.advance_one_month()
    final = sim.final_results()
    assert final["ending_inventory"] == {
        "blue_jeans":31.0, "whats_next":0.0, "high_fashion":684.0
    }
    assert final["lost_sales"]["whats_next"] == 226
    assert round(final["final_profit"], 2) == 405685.07


def test_stockout_before_arrival_is_lost():
    sim = InventorySimulation(
        china_initial={"blue_jeans":2000,"whats_next":2000,"high_fashion":2000},
        scenario_id=1009,
    )
    sim.advance_one_month()
    assert sim.replenishment_lead_time_weeks(1) == 3
    assert sim.expected_arrival_week(1) == 8
    sim.place_replenishment(1, {"blue_jeans":500,"whats_next":1000,"high_fashion":1000})
    sim.advance_one_month()
    records = sim.instructor_state()["weekly_records"]
    waiting = [r for r in records if r["week"] in (5,6,7)]
    assert any(r["lost_sales"] > 0 for r in waiting)
    assert all(r["arrival_qty"] == 0 for r in waiting)
    arrival = [r for r in records if r["week"] == 8]
    assert any(r["arrival_qty"] > 0 for r in arrival)


if __name__ == "__main__":
    test_scenario_bank_structure()
    test_common_first_attempt_reference_path()
    test_stockout_before_arrival_is_lost()
    print("All scenario-bank tests passed.")
