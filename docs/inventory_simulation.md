# Beautiful Bags Inventory Simulation

## Current scenario design

Students may complete up to five attempts in one browser session.

- Attempt 1 always uses common scenario **3203** for class comparability.
- Attempts 2–5 draw without replacement from the remaining approved scenarios.
- The approved bank contains seven scenarios total, so a student will not repeat a scenario within five attempts.
- All scenarios retain the same forecast information, costs, lifecycle profiles, sourcing rules, and capacity rules; only realized demand paths and TJL lead times differ.

### Approved scenario bank (instructor reference)

| Scenario | Instructor description | Seasonal demand: Blue Jeans / What's Next / High Fashion | TJL lead times after Months 1 / 2 / 3 |
|---|---|---|---|
| 3203 | Baseline: trend upside, fashion downside | 9,969 / 8,226 / 4,316 | 2 / 2 / 1 weeks |
| 1009 | Broad upside | 10,728 / 8,996 / 6,143 | 3 / 2 / 1 weeks |
| 1007 | Broad downside | 8,925 / 5,909 / 5,205 | 1 / 2 / 2 weeks |
| 1047 | Fashion surprise | 9,758 / 7,851 / 6,455 | 2 / 1 / 3 weeks |
| 1075 | Trend disappointment | 10,354 / 6,475 / 4,924 | 1 / 3 / 2 weeks |
| 1068 | Stable-product surprise | 11,172 / 7,735 / 5,208 | 3 / 1 / 2 weeks |
| 1016 | Mixed difficult | 8,934 / 8,667 / 6,680 | 2 / 3 / 1 weeks |

The scenario descriptions and IDs are instructor-facing and should not be shown to students during active play.

## TJL lead-time logic

TJL in-season lead time is expressed in weekly buckets: **1, 2, or 3 weeks**.

Before each Month 1–3 replenishment decision, the student sees the exact lead time that will apply to an order placed at that decision point. The actual lead time equals the displayed lead time.

Arrival convention:

- Order at the end of Week 4 with a 1-week lead time → available at the beginning of Week 6.
- Order at the end of Week 4 with a 2-week lead time → available at the beginning of Week 7.
- Order at the end of Week 4 with a 3-week lead time → available at the beginning of Week 8.

The same convention applies after Months 2 and 3.

Demand during the waiting weeks must be served from inventory already on hand. If inventory reaches zero before the TJL order arrives, unmet demand is lost and is not recovered after replenishment arrives.

## Student information rule

During active play, students see:

- monthly sales,
- ending inventory,
- outstanding TJL orders and expected arrival week,
- revenue to date,
- sourcing cost committed,
- holding cost to date,
- interim profit,
- the TJL lead time for the next decision.

Students do not see true demand or lost demand until the Month 4 debrief.
