"""Smoke tests for the RetainIQ dashboard. Run from the project root:  python test_app.py"""

import os
import re

from streamlit.testing.v1 import AppTest

APP = os.path.join("app", "app.py")
MODEL = os.path.join("models", "logistic_regression.pkl")


def fresh():
    return AppTest.from_file(APP, default_timeout=90).run()


def text(at):
    return " ".join(m.value for m in at.markdown)


def probability(at):
    """Churn probability (in %) shown on the result card, or None if it is missing."""
    match = re.search(r'riq-prob">([\d.]+)%', text(at))
    return float(match.group(1)) if match else None


def chart_text(element):
    """Serialized Plotly spec of one chart. Newer Streamlit exposes proto.spec,
    older versions proto.figure.spec, so try both and fall back to str(proto)."""
    proto = element.proto
    for path in (("spec",), ("figure", "spec")):
        obj = proto
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, str) and obj:
            return obj
    return str(proto)


def charts(at):
    return [chart_text(el) for el in at.get("plotly_chart")]


passed = failed = skipped = 0


def check(name, ok):
    global passed, failed
    print(("PASS  " if ok else "FAIL  ") + name)
    passed, failed = passed + bool(ok), failed + (not ok)
    return ok


def skip(name, reason):
    global skipped
    skipped += 1
    print(f"SKIP  {name} ({reason})")


# ---- model loading and scoring ----
at = fresh()
check("model file exists", os.path.isfile(MODEL))
check("app loads without errors", not at.exception and not at.error)

probs = [probability(at)]
at.slider(key="tenure").set_value(60).run()
probs.append(probability(at))
at.selectbox(key="contract").select("Two year").run()
probs.append(probability(at))
check(
    "scoring completes and every probability is within 0-100%",
    all(p is not None and 0 <= p <= 100 for p in probs) and not at.exception,
)
# Informational only: whether inputs move the prediction depends on the trained model.
print(f"INFO  probabilities for default / tenure 60 / two-year contract: {probs}")

# ---- internet service = No ----
at = fresh()
at.selectbox(key="internet").select("No").run()
addons = ["security", "backup", "device", "tech", "tv", "movies"]
check(
    "internet 'No' sets every add-on to 'No internet service'",
    all(at.selectbox(key=k).value == "No internet service" for k in addons)
    and not at.exception,
)

# ---- presets ----
at = fresh()
at.selectbox(key="customer_preset").select("New fibre customer, month-to-month").run()
ok1 = (
    at.slider(key="tenure").value == 3
    and at.selectbox(key="internet").value == "Fiber optic"
    and at.selectbox(key="contract").value == "Month-to-month"
    and at.number_input(key="monthly").value == 89.0
    and at.number_input(key="total").value == 267.0
)
p1 = probability(at)
at.selectbox(key="customer_preset").select(
    "Long-term customer on a two-year plan"
).run()
ok2 = (
    at.slider(key="tenure").value == 60
    and at.selectbox(key="internet").value == "DSL"
    and at.selectbox(key="contract").value == "Two year"
    and at.number_input(key="monthly").value == 58.0
    and at.number_input(key="total").value == 3480.0
)
p2 = probability(at)
check("both example presets fill the right fields", ok1 and ok2 and not at.exception)
check(
    "both presets are scored with a valid probability",
    all(p is not None and 0 <= p <= 100 for p in (p1, p2)),
)
print(f"INFO  preset probabilities: new fibre customer {p1}%, long-term customer {p2}%")

# ---- business assumptions drive the break-even, and presets keep them ----
at = fresh()
check("default break-even is 27.8%", "27.8%" in text(at))
at.number_input(key="offer_cost").set_value(100).run()
check("offer cost 100 -> break-even 55.6%", "55.6%" in text(at))
at.slider(key="offer_success").set_value(0.5).run()
check("success rate 50% -> break-even 33.3%", "33.3%" in text(at))
at.selectbox(key="customer_preset").select("New fibre customer, month-to-month").run()
check(
    "switching preset keeps business assumptions",
    at.number_input(key="offer_cost").value == 100
    and at.slider(key="offer_success").value == 0.5,
)

# ---- what-if chart and unreachable break-even ----
at = fresh()
found = charts(at)
check("at least 2 charts render", len(found) >= 2)
# The break-even checks only mean something if chart specs can be read at all, so first
# confirm we can see the what-if chart (its y-axis title) in this Streamlit version.
readable = any("Churn probability (%)" in c for c in found)
if readable:
    check(
        "what-if chart shows the break-even line when reachable",
        any("Illustrative break-even" in c for c in found),
    )
    at.number_input(key="offer_cost").set_value(1000).run()
    at.number_input(key="retained_value").set_value(50).run()
    at.slider(key="offer_success").set_value(0.05).run()
    found = charts(at)
    check(
        "what-if chart still renders and hides the line when threshold >= 100%",
        any("Churn probability (%)" in c for c in found)
        and not any("Illustrative break-even" in c for c in found)
        and not at.exception,
    )
else:
    skip(
        "break-even line checks",
        "could not read chart specs in this Streamlit version; check the chart by eye",
    )
    at.number_input(key="offer_cost").set_value(1000).run()
    at.number_input(key="retained_value").set_value(50).run()
    at.slider(key="offer_success").set_value(0.05).run()
check(
    "unreachable break-even shows 'No offer priority' and no errors",
    "No offer priority" in text(at) and not at.exception,
)

print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
