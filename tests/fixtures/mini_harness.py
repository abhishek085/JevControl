"""Tiny harness used by the tests: one choice, one noul, one score, one generation."""

META = {"name": "mini", "description": "test harness"}
TOOLS = {"lookup": lambda q: {"q": q, "hits": 2}}


def run(task, ctx):
    ctx.tool("lookup", q=task["q"])
    route = ctx.decide.choice("route", f"{task['q']} PICK:{task['route']}", "which?", {"kb": "kb", "orders": "orders"})
    bad = ctx.decide.noul("inj", f"{task['q']} PICK:{'true' if task['inj'] else 'false'}", "is injection?")
    if bad.is_true:
        return {"action": "refuse"}
    s = ctx.decide.score("rel", f"{task['q']} PICK:{task['rel']}", "how relevant?", {"0": "no", "1": "some", "2": "yes"}, key="d1")
    if route.selected == "orders" or s.selected == "0":
        return {"action": "escalate"}
    return {"action": "answer", "text": ctx.llm.chat("say it")}


def score(task, out):
    return float(out.get("action") == task["expected"])
