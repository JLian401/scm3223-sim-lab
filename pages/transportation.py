"""Retail Delivery Challenge v2. Run: python -m streamlit run transportation_game.py
Only streamlit and pandas are required. All rates and events are fictional.
"""
import json
import random
from collections import defaultdict

import pandas as pd
import streamlit as st

DAYS = 12
CAPACITY = 10
TRUCK_COST = 900
LTL_FIXED = 160
LTL_PALLET = 110
EXPRESS_PALLET = 300
LATE_PALLET_DAY = 120
UNSHIPPED_PALLET = 1800
ROUTES = {
    'South': ['Moore', 'Norman', 'Purcell'],
    'North': ['Edmond', 'Guthrie', 'Stillwater'],
    'West': ['Yukon', 'El Reno', 'Weatherford'],
}
OPTIONS = ['Hold', 'Express', 'LTL', 'Truck 1', 'Truck 2']


def make_orders():
    rng = random.Random(3223)
    stores = [(route, store) for route, names in ROUTES.items() for store in names]
    orders = []
    for day in range(1, 11):
        count = 7 if day <= 3 else rng.choice([6, 7, 8])
        for i in range(count):
            route, store = stores[(day * 5 + i * 4) % 9]
            pallets = rng.choices([1, 2, 3, 4], weights=[3, 4, 3, 1])[0]
            slack = rng.choice([4, 5]) if day <= 3 else rng.choices([2, 3, 4, 5], [1, 3, 4, 2])[0]
            orders.append(dict(Order=f'O{len(orders)+1:03d}', Store=store, Route=route,
                               Released=day, Pallets=pallets, Deadline=day+slack))
    return orders


ORDERS = make_orders()
BY_ID = {o['Order']: o for o in ORDERS}


def conditions(day):
    """Known conditions at the start of this dispatch day."""
    total = {5: 2, 8: 1, 11: 2}.get(day, 6)
    transit = 3 if day in (7, 8) else 2
    notices = []
    if total < 6:
        notices.append(f'Truck shortage: only {total} dedicated truck(s) across ALL routes today.')
    if transit == 3:
        notices.append('LTL congestion: shipments dispatched today take 3 days instead of 2.')
    if day == 4:
        notices.append('Advance notice: tomorrow only 2 dedicated trucks will be available across all routes.')
    if day == 6:
        notices.append('Advance notice: LTL dispatched on Days 7 and 8 will take 3 days. Day 8 will have only 1 truck across all routes.')
    if day == 10:
        notices.append('Advance notice: tomorrow only 2 dedicated trucks will be available across all routes. All game orders have now been released.')
    return dict(trucks=total, ltl_days=transit, notices=notices)


def backlog(day, history):
    sent = {s['Order'] for s in history}
    return [dict(o) for o in ORDERS if o['Released'] <= day and o['Order'] not in sent]


def build_plan(day, assignments, history):
    """Validate atomically, consolidate by store for LTL and route/slot for trucks.
    Return order-level allocations and shipment-level quotes; never mutate state.
    """
    if not isinstance(day, int) or not 1 <= day <= DAYS:
        return [], [], ['Dispatch day must be between 1 and 12.']
    active = {o['Order']: o for o in backlog(day, history)}
    groups = defaultdict(list)
    errors = []
    for oid, mode in assignments.items():
        if oid not in active:
            errors.append(f'{oid} is unavailable or has already shipped.'); continue
        if mode not in OPTIONS:
            errors.append(f'{oid} has an invalid shipping choice.'); continue
        if mode == 'Hold':
            continue
        o = active[oid]
        # LTL combines orders only for the SAME store on the SAME day.
        group = (o['Route'], mode, o['Store'] if mode in ('LTL', 'Express') else '')
        groups[group].append(o)
    cfg = conditions(day)
    truck_groups = [k for k in groups if k[1].startswith('Truck')]
    if len(truck_groups) > cfg['trucks']:
        errors.append(f"Today's network limit is {cfg['trucks']} trucks; your plan uses {len(truck_groups)}.")
    allocations, quotes = [], []
    for (route, mode, store), orders in sorted(groups.items()):
        load = sum(o['Pallets'] for o in orders)
        truck = mode.startswith('Truck')
        if truck and load > CAPACITY:
            errors.append(f'{route} {mode}: {load} pallets exceeds the 10-pallet limit.')
        transit = cfg['ltl_days'] if mode == 'LTL' else 1
        cost = TRUCK_COST if truck else (LTL_FIXED+LTL_PALLET*load if mode == 'LTL' else EXPRESS_PALLET*load)
        arrival = day+transit
        sid = f'D{day:02d}-{route}-{mode.replace(" ", "")}-{store or "route"}'
        penalty = sum(max(0, arrival-o['Deadline'])*LATE_PALLET_DAY*o['Pallets'] for o in orders)
        quotes.append(dict(Shipment=sid, Route=route, Mode=mode,
                           Stores=', '.join(sorted({o['Store'] for o in orders})),
                           Orders=', '.join(o['Order'] for o in orders), Pallets=load,
                           Arrival=arrival, Freight=cost, **{'Late penalty': penalty,
                           'Truck fill %': load/CAPACITY*100 if truck else None}))
        for o in orders:
            late = max(0, arrival-o['Deadline'])
            allocations.append(dict(**o, Shipment=sid, Mode=mode, Dispatch=day,
                                    Arrival=arrival, Freight=cost*o['Pallets']/load,
                                    Penalty=late*LATE_PALLET_DAY*o['Pallets'],
                                    Late_days=late, On_time=late==0))
    return allocations, quotes, errors


def score(history, day):
    finished = day > DAYS
    missing = backlog(DAYS, history) if finished else []
    freight = sum(s['Freight'] for s in history)
    penalty = sum(s['Penalty'] for s in history)+sum(o['Pallets'] for o in missing)*UNSHIPPED_PALLET
    trucks = {s['Shipment'] for s in history if s['Mode'].startswith('Truck')}
    load = sum(s['Pallets'] for s in history if s['Mode'].startswith('Truck'))
    total_pallets = sum(o['Pallets'] for o in ORDERS)
    on_time = sum(s['Pallets'] for s in history if s['On_time'])
    delivered = sum(s['Pallets'] for s in history if finished or s['Arrival'] <= day)
    released = sum(o['Pallets'] for o in ORDERS if o['Released'] <= min(day, DAYS))
    waiting = sum(o['Pallets'] for o in backlog(min(day, DAYS), history))
    shipped = sum(s['Pallets'] for s in history)
    wait = sum((s['Dispatch']-s['Released'])*s['Pallets'] for s in history)/shipped if shipped else 0
    return dict(freight=freight, penalty=penalty, total=freight+penalty,
                service=100*on_time/total_pallets, trucks=len(trucks),
                utilization=100*load/(CAPACITY*len(trucks)) if trucks else 0,
                waiting=waiting, delivered=delivered, released=released,
                shipped=shipped, on_time=on_time, average_wait=wait,
                total_pallets=total_pallets, unshipped=sum(o['Pallets'] for o in missing))


def initialize():
    st.session_state.setdefault('day', 1)
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('daily', [])
    st.session_state.setdefault('team', '')


def order_table(orders, day):
    cfg = conditions(day)
    return pd.DataFrame([dict(**o, **{'Days left': o['Deadline']-day,
                          'Latest LTL dispatch': o['Deadline']-cfg['ltl_days'],
                          'Latest truck/express dispatch': o['Deadline']-1,
                          'Choice': 'Hold'}) for o in orders])


def main():
    st.set_page_config(page_title='Retail Delivery Challenge', page_icon='🚚', layout='wide')
    initialize()
    day = st.session_state.day
    finished = day > DAYS
    history = st.session_state.history
    st.title('🚚 Retail Delivery Challenge')
    st.caption('12 business days · 9 stores · 3 routes · Consolidate across orders and destinations')
    with st.sidebar:
        st.text_input('Team name', key='team')
        st.markdown('**Goal:** meet the 95% on-time pallet target, then minimize freight plus penalties.')
        st.caption('About 20 minutes of play. Advance at your own pace; no timer. Use one browser tab per team.')
        if st.button('New game / reset'):
            st.session_state['confirm_reset'] = True
        if st.session_state.get('confirm_reset'):
            st.warning('This will erase your current game.')
            if st.button('Confirm reset'):
                st.session_state.clear()
                st.rerun()
            if st.button('Keep playing'):
                st.session_state['confirm_reset'] = False
                st.rerun()
        st.markdown('**Routes from the distribution center**')
        for route, stores in ROUTES.items():
            st.write(f"**{route}:** {' → '.join(stores)}")
        st.caption('Predefined service corridors; no route sequencing required. Business days only.')
    with st.expander('How to play and shipping rates', expanded=day == 1):
        st.markdown('Each day, assign orders in the route tables. **The same truck number on the same route creates one truck load.** Truck 1 on South and Truck 1 on North are two separate trucks. Each order stays intact. Unassigned orders are held automatically. Orders cannot be changed after dispatch.')
        st.table(pd.DataFrame([
            ['Express', '$300 per pallet', '1 day', 'Urgent small orders; no capacity limit'],
            ['LTL', '$160 per store shipment + $110 per pallet', 'Normally 2 days', 'Same-store orders dispatched together share ONE $160 charge'],
            ['Dedicated truck', '$900 per truck', '1 day', '10 pallets; mix stores on one route; normally 2 trucks per route'],
            ['Hold', 'No storage fee in this exercise', 'Dispatch later', 'Wait to consolidate; watch deadlines and truck availability'],
        ], columns=['Choice', 'Freight', 'Transit', 'Consolidation rule']))
        st.markdown('Dispatch Day 2 + two-day transit = arrival Day 4. Late deliveries cost **$120 per pallet per business day**. Unshipped pallets after Day 12 cost **$1,800 per pallet** and fail service. Dispatched shipments are followed automatically to arrival after Day 12. All orders arrive by Day 10; deadlines run up to Day 15. All prices and disruptions are fictional teaching assumptions.')
        st.caption('Latest dispatch columns use TODAY’S transit times; announced future congestion can change these dates. Future order details and unannounced disruptions stay hidden. Typical arrivals: 6–8 orders/day, mostly 1–4 pallets.')
    m = score(history, day)
    cols = st.columns(4)
    cols[0].metric('Committed total cost', f"${m['total']:,.0f}")
    cols[1].metric('Truck utilization', f"{m['utilization']:.1f}%" if m['trucks'] else '—')
    cols[2].metric('Truck departures', m['trucks'])
    cols[3].metric('Average wait before dispatch', f"{m['average_wait']:.1f} days")
    if not finished:
        st.progress((day-1)/DAYS, text=f'Dispatch day {day} of {DAYS}')
        cfg = conditions(day)
        st.info(f"Today: {cfg['trucks']} truck(s) across the network, at most 2 per route; LTL transit {cfg['ltl_days']} days. Truck and express transit 1 day.")
        for notice in cfg['notices']:
            st.warning(notice)
        if day >= 11:
            st.info('No more new orders. Use the remaining dispatch days to clear your backlog.')
        active = backlog(day, history)
        st.write(f"**{len(active)} available orders / {m['waiting']} pallets** · {m['delivered']} pallets delivered so far · {m['shipped']-m['delivered']} pallets in transit")
        if st.session_state.daily:
            last = st.session_state.daily[-1]
            st.caption(f"Previous dispatch: ${last['Freight']:,.0f} freight, ${last['Late penalties']:,.0f} eventual late penalties, {last['Trucks']} trucks, {last['Held pallets']} pallets held.")
        st.caption('Edit only the Choice column. Use Truck 1/Truck 2 to group orders. To apply one choice to several rows quickly, copy a Choice cell and paste it into selected Choice cells. Preview updates when the edit is committed.')
        assignments = {}
        for tab, route in zip(st.tabs(list(ROUTES)), ROUTES):
            with tab:
                route_orders = sorted([o for o in active if o['Route'] == route], key=lambda o: (o['Deadline'], o['Store'], o['Order']))
                if not route_orders:
                    st.write('No waiting orders on this route.')
                    continue
                frame = order_table(route_orders, day)
                columns = ['Order', 'Store', 'Pallets', 'Released', 'Deadline', 'Days left', 'Latest LTL dispatch', 'Latest truck/express dispatch', 'Choice']
                edited = st.data_editor(frame[columns], key=f'editor_{day}_{route}', hide_index=True,
                                        disabled=[c for c in columns if c != 'Choice'],
                                        column_config={'Choice': st.column_config.SelectboxColumn('Choice', options=OPTIONS, required=True)},
                                        use_container_width=True)
                assignments.update(dict(zip(edited['Order'], edited['Choice'])))
        rows, quotes, errors = build_plan(day, assignments, history)
        st.subheader('Today’s shipment preview')
        if quotes:
            st.dataframe(pd.DataFrame(quotes).style.format({'Freight': '${:,.0f}', 'Late penalty': '${:,.0f}', 'Truck fill %': '{:.0f}'}, na_rep='—'), hide_index=True, use_container_width=True)
            st.write(f"**Freight: ${sum(q['Freight'] for q in quotes):,.0f} · Late penalties: ${sum(q['Late penalty'] for q in quotes):,.0f}**")
        else:
            st.write('No shipments planned. All available orders will be held.')
        held = [o for o in active if assignments.get(o['Order'], 'Hold') == 'Hold']
        held_qty = sum(o['Pallets'] for o in held)
        st.caption(f'{len(held)} orders / {held_qty} pallets will remain in the distribution center.')
        urgent = [o['Order'] for o in held if o['Deadline'] <= day+1]
        if urgent:
            st.warning('Holding these orders makes an on-time delivery impossible after today: '+', '.join(urgent))
        if day == DAYS and held:
            st.error(f'Final-day hold: {held_qty} pallets will be unshipped, adding ${held_qty*UNSHIPPED_PALLET:,.0f} to penalties and counting as not on time.')
        for error in errors:
            st.error(error)
        if st.button('Confirm dispatch and finish game' if day == DAYS else 'Confirm dispatch and advance day', type='primary', disabled=bool(errors)):
            trucks = [q for q in quotes if q['Mode'].startswith('Truck')]
            st.session_state.daily.append({'Day': day, 'Freight': sum(q['Freight'] for q in quotes),
                'Late penalties': sum(q['Late penalty'] for q in quotes), 'Trucks': len(trucks),
                'Truck utilization %': sum(q['Pallets'] for q in trucks)/(CAPACITY*len(trucks))*100 if trucks else None,
                'Held pallets': held_qty})
            st.session_state.history.extend(rows)
            st.session_state.day += 1
            # Old editor widget state is no longer needed.
            for key in list(st.session_state):
                if key.startswith(f'editor_{day}_'):
                    del st.session_state[key]
            st.rerun()
    else:
        st.subheader(f"Final results — {st.session_state.team or 'Team'}")
        st.metric('On-time pallets / all game pallets', f"{m['service']:.1f}%")
        if m['service'] >= 95:
            st.success('95% service target met. Compare total cost with other teams that met the target.')
        else:
            st.warning('Service target missed. Rank below teams meeting 95%; among below-target teams, rank by service first, then total cost.')
        st.write(f"Freight **${m['freight']:,.0f}** + penalties **${m['penalty']:,.0f}** = **${m['total']:,.0f}**. Unshipped pallets: **{m['unshipped']}**.")
        st.caption('Final service counts every game pallet, including unshipped pallets. All dispatched shipments have been followed to arrival; costs include every late day.')
        st.text_area('Reflection: Describe one decision in which waiting to consolidate helped—or hurt—your team. Why?', key='reflection')
        summary = dict(Team=st.session_state.team or 'Team', **m, Reflection=st.session_state.get('reflection', ''))
        st.download_button('Download team summary', json.dumps(summary, indent=2), 'team_summary.json', 'application/json')
        report = list(history)
        for o in backlog(DAYS, history):
            report.append(dict(**o, Mode='Unshipped', Freight=0, Penalty=o['Pallets']*UNSHIPPED_PALLET, On_time=False))
        st.download_button('Download order results (CSV)', pd.DataFrame(report).to_csv(index=False), 'order_results.csv', 'text/csv')
    with st.expander('Daily performance and shipment history', expanded=finished):
        if st.session_state.daily:
            daily = pd.DataFrame(st.session_state.daily)
            st.dataframe(daily, hide_index=True, use_container_width=True)
            st.bar_chart(daily.set_index('Day')[['Freight', 'Late penalties']])
        if history:
            frame = pd.DataFrame(history)
            frame['Status'] = ['Delivered' if finished or a <= day else 'In transit' for a in frame['Arrival']]
            st.dataframe(frame, hide_index=True, use_container_width=True)
        if not history:
            st.write('No shipments dispatched yet.')
    st.caption('Keep this tab open. Progress is held in this Streamlit session; a browser reload or server restart may reset it. There is no shared leaderboard. Download results before closing.')


if __name__ == '__main__':
    main()
