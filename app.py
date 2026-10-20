import math
import joblib
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Delivery time estimator", layout="wide")
INK, GREEN, RED, LINE = "#101820", "#0B6E4F", "#B5452F", "#D5DAD9"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NICE = {"electric_scooter": "Electric scooter", "motorcycle": "Motorcycle", "scooter": "Scooter",
        "bicycle": "Bicycle", "Metropolitian": "Metropolitan", "Semi-Urban": "Semi-urban"}
nice = lambda v: NICE.get(v, str(v))

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, label, button {{ font-family: 'IBM Plex Sans', sans-serif; }}
h1, h2, h3, .eta-number {{ font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.02em; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 1150px; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
.eta {{ border-left: 6px solid {GREEN}; padding: 0.3rem 0 0.3rem 1.4rem; margin-bottom: 1.2rem; }}
.eta-number {{ font-size: 6rem; font-weight: 700; line-height: 1; color: {INK}; }}
.eta-unit {{ font-size: 1.5rem; color: #55615f; margin-left: 0.4rem; }}
.eta-sub {{ font-size: 1.05rem; color: #3c4846; margin-top: 0.5rem; }}
div[data-testid="stMetric"] {{ border-top: 2px solid {INK}; padding-top: 0.5rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return joblib.load("delivery_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("delivery_clean.csv")


S = load()
model, low, high, cols, opt, rng = S["model"], S["low"], S["high"], S["columns"], S["options"], S["ranges"]
order = [t for t in ["Low", "Medium", "High", "Jam"] if t in opt["traffic"]]

# ---------- defaults and quick scenarios ----------
pick = lambda key, pref: pref if pref in opt[key] else opt[key][0]
DEFAULT = {"distance": 4.0, "hour": 13, "weekday": DAYS[2], "traffic": order[1] if len(order) > 1 else order[0],
           "weather": pick("weather", "Sunny"), "order_type": opt["order_type"][0], "vehicle": pick("vehicle", "motorcycle"),
           "age": int(sum(rng["rider_age"]) / 2), "rating": 4.6, "multi": 0, "festival": False,
           "city": opt["city"][0], "promise": 30}
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)

SCENARIOS = {
    "Custom": {},
    "Lunch rush, busy roads": {"hour": 13, "traffic": "High", "distance": 5.0, "multi": 1},
    "Stormy evening, long ride": {"hour": 20, "weather": "Stormy", "traffic": "Medium", "distance": 9.0},
    "Quiet afternoon, short ride": {"hour": 16, "traffic": "Low", "distance": 2.0, "multi": 0},
    "Festival night": {"hour": 21, "festival": True, "traffic": "Jam", "distance": 6.0},
}


def apply_scenario():
    for k, v in SCENARIOS[st.session_state.scn].items():
        if k in ("traffic", "weather") and v not in opt[k]:
            continue
        st.session_state[k] = v


with st.sidebar:
    st.header("Order details")
    st.selectbox("Quick scenario", list(SCENARIOS), key="scn", on_change=apply_scenario,
                 help="Fills the form with a typical situation. You can still change anything.")
    st.slider("Distance (km)", 0.5, 20.0, key="distance", step=0.5, help="Restaurant to customer, in a straight line.")
    st.slider("Order placed at (hour, 24h)", 0, 23, key="hour")
    st.select_slider("Traffic", order, key="traffic")
    st.selectbox("Weather", opt["weather"], key="weather")
    st.selectbox("What was ordered", opt["order_type"], key="order_type")
    with st.expander("Rider and extra details"):
        st.selectbox("Vehicle", opt["vehicle"], key="vehicle", format_func=nice)
        st.slider("Rider age", int(rng["rider_age"][0]), int(rng["rider_age"][1]), key="age")
        st.slider("Rider rating", 1.0, 5.0, key="rating", step=0.1)
        st.slider("Other orders the rider is carrying", 0, 3, key="multi")
        st.selectbox("Day", DAYS, key="weekday")
        st.selectbox("City type", opt["city"], key="city", format_func=nice)
        st.toggle("Festival day", key="festival")
    st.number_input("Promised delivery time (min)", 10, 90, key="promise",
                    help="The time you told the customer. Used for the late-risk check.")
    st.caption(S["source"])

if "history" not in st.session_state:
    st.session_state.history = []


def row(**over):
    s = {**{k: st.session_state[k] for k in DEFAULT}, **over}
    return pd.DataFrame([{
        "distance_km": s["distance"], "rider_age": s["age"], "rider_rating": s["rating"],
        "multi_orders": s["multi"], "hour": s["hour"], "weekday": DAYS.index(s["weekday"]),
        "traffic": s["traffic"], "weather": s["weather"], "order_type": s["order_type"],
        "vehicle": s["vehicle"], "festival": "Yes" if s["festival"] else "No", "city": s["city"]}])[cols]


def estimate(**over):
    x = row(**over)
    p = float(model.predict(x)[0])
    return p, min(float(low.predict(x)[0]), p), max(float(high.predict(x)[0]), p)


def clock(minutes):
    t = int(st.session_state.hour * 60 + minutes)
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


pt, lo, hi = estimate()
promise = st.session_state.promise

st.title("Delivery time estimator")
t1, t2, t3, t4, t5 = st.tabs(["Estimate", "What-if", "Insights from data", "Batch upload", "Model"])

# ---------- Estimate ----------
with t1:
    st.markdown(f"""<div class="eta"><span class="eta-number">{round(pt)}</span><span class="eta-unit">minutes</span>
    <div class="eta-sub">Arrives around {clock(pt)}. Most orders like this take {round(lo)} to {round(hi)} minutes.</div></div>""",
                unsafe_allow_html=True)

    if promise >= hi:
        st.success(f"A {promise}-minute promise is safe. Nearly all similar orders arrive in time.")
    elif promise >= pt:
        st.info(f"A {promise}-minute promise is likely to hold, but a slow order could miss it. Consider {math.ceil(hi)} minutes to be safe.")
    elif promise >= lo:
        st.warning(f"A {promise}-minute promise is at risk. Quote {math.ceil(hi)} minutes or assign a faster rider.")
    else:
        st.error(f"A {promise}-minute promise will probably be missed. Quote {math.ceil(hi)} minutes instead.")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("What is adding time")
        fixes = {"Weather": {"weather": pick("weather", "Sunny")}, "Traffic": {"traffic": order[0]},
                 "Busy hour": {"hour": 16}, "Festival": {"festival": False}, "Rider carrying other orders": {"multi": 0}}
        rows = [{"factor": n, "minutes": round(pt - estimate(**c)[0], 1)} for n, c in fixes.items()
                if pt - estimate(**c)[0] > 0.3]
        if rows:
            st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar(color=RED, size=22).encode(
                x=alt.X("minutes:Q", title="Extra minutes"), y=alt.Y("factor:N", sort="-x", title=None)
            ).properties(height=40 * len(rows) + 40), width="stretch")
        else:
            st.write("Conditions are good. Nothing is slowing this order down.")
    with right:
        st.subheader("Message for the customer")
        st.code(f"Your order is on its way and should reach you by {clock(pt)} "
                f"(about {round(lo)}-{round(hi)} minutes).", language=None)
        if st.button("Save this estimate", width="stretch"):
            st.session_state.history.append({"distance_km": st.session_state.distance, "hour": st.session_state.hour,
                                             "traffic": st.session_state.traffic, "weather": st.session_state.weather,
                                             "estimate_min": round(pt), "low": round(lo), "high": round(hi)})
    if st.session_state.history:
        st.subheader("Saved estimates")
        h = pd.DataFrame(st.session_state.history)
        st.dataframe(h, width="stretch", hide_index=True)
        a, b = st.columns([1, 5])
        a.download_button("Download CSV", h.to_csv(index=False), "estimates.csv", "text/csv")
        if b.button("Clear list"):
            st.session_state.history = []
            st.rerun()

# ---------- What-if ----------
with t2:
    st.write("See how the estimate changes as one thing varies. Everything else follows the sidebar.")
    a, b = st.columns(2)
    d = pd.DataFrame([{"distance": k / 2, "traffic": t, "minutes": estimate(distance=k / 2, traffic=t)[0]}
                      for k in range(1, 41, 2) for t in order])
    a.subheader("Distance and traffic")
    a.altair_chart(alt.Chart(d).mark_line(strokeWidth=3).encode(
        x=alt.X("distance:Q", title="Distance (km)"), y=alt.Y("minutes:Q", title="Minutes", scale=alt.Scale(zero=False)),
        color=alt.Color("traffic:N", title="Traffic", sort=order)), width="stretch")
    h = pd.DataFrame([{"hour": x, "minutes": estimate(hour=x)[0]} for x in range(0, 24)])
    b.subheader("Time of day")
    b.altair_chart(alt.Chart(h).mark_bar(color=GREEN).encode(
        x=alt.X("hour:O", title="Hour of day"), y=alt.Y("minutes:Q", title="Minutes", scale=alt.Scale(zero=False))),
        width="stretch")
    st.subheader("Which vehicle is fastest here?")
    v = pd.DataFrame([{"vehicle": nice(x), "minutes": estimate(vehicle=x)[0]} for x in opt["vehicle"]])
    st.altair_chart(alt.Chart(v).mark_bar(color=INK, size=22).encode(
        x=alt.X("minutes:Q", title="Minutes"), y=alt.Y("vehicle:N", sort="x", title=None)
    ).properties(height=40 * len(v) + 30), width="stretch")

# ---------- Insights ----------
with t3:
    data = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Orders analysed", f"{len(data):,}")
    c2.metric("Average delivery", f"{data.minutes.mean():.0f} min")
    c3.metric("Orders over 40 min", f"{(data.minutes > 40).mean():.0%}")
    by = st.selectbox("Compare average delivery time by", ["traffic", "weather", "order_type", "vehicle", "city",
                                                           "festival", "hour", "weekday"],
                      format_func=lambda x: {"order_type": "order type", "weekday": "day of week"}.get(x, x))
    g = data.groupby(by).minutes.agg(["mean", "count"]).reset_index()
    if by == "weekday":
        g[by] = g[by].map(lambda i: DAYS[int(i)])
    g[by] = g[by].map(nice) if by in ("vehicle", "city") else g[by]
    st.altair_chart(alt.Chart(g).mark_bar(color=GREEN).encode(
        x=alt.X(f"{by}:N", sort=(order if by == "traffic" else None), title=None),
        y=alt.Y("mean:Q", title="Average minutes"),
        tooltip=[by, alt.Tooltip("mean:Q", format=".1f"), "count"]), width="stretch")

# ---------- Batch ----------
with t4:
    st.write("Upload a CSV of orders and get an estimate for each one.")
    tpl = row()
    st.download_button("Download template", tpl.to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("Orders file", type="csv")
    if up:
        df = pd.read_csv(up)
        miss = [c for c in cols if c not in df.columns]
        if miss:
            st.error(f"Missing columns: {', '.join(miss)}. Start from the template to get the right format.")
        else:
            df["estimate_min"] = model.predict(df[cols]).round(1)
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("Download results", df.to_csv(index=False), "predictions.csv", "text/csv")

# ---------- Model ----------
with t5:
    m = S["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average error", f"{m['mae']:.1f} min", f"{m['base_mae'] - m['mae']:.1f} better than linear model")
    c2.metric("Within 5 minutes", f"{m['within5']:.0%}")
    c3.metric("R-squared", f"{m['r2']:.2f}")
    c4.metric("Range accuracy", f"{m['coverage']:.0%}", "target 80%", delta_color="off")
    imp = pd.DataFrame({"feature": list(S["importance"]), "importance": list(S["importance"].values())})
    st.subheader("What the model relies on")
    st.altair_chart(alt.Chart(imp).mark_bar(color=INK).encode(
        x=alt.X("importance:Q", title="Minutes of error added if this input is scrambled"),
        y=alt.Y("feature:N", sort="-x", title=None)).properties(height=300), width="stretch")
    st.caption(f"{S['source']}. Trained on {m['rows']:,} orders.")