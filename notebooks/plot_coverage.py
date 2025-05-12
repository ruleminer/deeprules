import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import LogicOperators
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.rule import AbstractRule


def _trim_rule(premise: CompoundCondition, n: int) -> CompoundCondition:
    res = CompoundCondition([], premise.logic_operator)
    i = 0
    if i == n:
        return res

    stopped_early = False

    for sub in premise.subconditions:
        if isinstance(sub, CompoundCondition):
            if i + len(sub.subconditions) <= n:
                i += len(sub.subconditions)
                res.subconditions.append(sub)
            else:
                tmp = CompoundCondition(sub.subconditions, sub.logic_operator)
                tmp.subconditions = sub.subconditions[: n - i]
                res.subconditions.append(tmp)
                i = n
        else:
            res.subconditions.append(sub)
            i += 1

        if i == n:
            stopped_early = True
            break

    if not stopped_early:
        return None
    return res


def draw_coverage_curve_plot(rule: AbstractRule, X: np.ndarray, y: np.ndarray):
    k = 1

    p = [0]
    n = [0]

    P = None
    N = None

    original_premise = rule.premise
    intermediate_rules: list[str] = []
    while True:

        trimmed_premise: AbstractRule = _trim_rule(original_premise, k)
        rule.premise = trimmed_premise

        if trimmed_premise is None:
            break
        intermediate_rules.append(str(rule))
        cov = rule.calculate_coverage(X, y)
        P = cov.P
        N = cov.N
        p.append(cov.p)
        n.append(cov.n)
        k += 1

    p[0] = P
    n[0] = N

    rule.premise = original_premise

    fig = go.Figure(
        data=go.Scatter(
            x=n,
            y=p,
            mode="lines+markers+text",
            line=dict(color="blue"),
            text=["∅"] + [f"r{i + 1}" for i in range(len(p))],
            textposition="top left",  # Ustawienie pozycji tekstu
            marker=dict(
                symbol="arrow",
                size=15,
                angleref="previous",
            ),
        )
    )
    fig.update_layout(
        height=600,
        width=900,
        xaxis_title="n",
        yaxis_title="p",
        font=dict(size=20),
        xaxis=dict(range=[None, N]),
        yaxis=dict(range=[None, P]),
    )
    fig.add_hline(y=P, line_dash="dash", line_color="gray")
    fig.add_vline(x=N, line_dash="dash", line_color="gray")

    for i, r in enumerate(intermediate_rules):
        print(f"r{i + 1}: {str(r)}")
    fig.show()
    return list(zip(p, n))
