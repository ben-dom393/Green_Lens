import plotly.graph_objects as go

from src.analysis import AnalysisResult
from src.config import METRIC_SCORE_MAX, METRIC_SCORE_MIN


def build_radar_chart(result: AnalysisResult, selected_metric: str = "") -> go.Figure:
    metrics = [metric.name for metric in result.metrics]
    scores = [metric.score for metric in result.metrics]
    display_metrics = [_wrap_metric_label(metric) for metric in metrics]

    polar_metrics = display_metrics + [display_metrics[0]]
    polar_scores = scores + [scores[0]]
    is_highlighted = bool(selected_metric)

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=polar_scores,
            theta=polar_metrics,
            fill="toself",
            name="Risk Score",
            hovertemplate="%{theta}: %{r:.0f}<extra></extra>",
            line={"color": "#1d6f5f", "width": 3},
            fillcolor="rgba(29, 111, 95, 0.18)" if is_highlighted else "rgba(29, 111, 95, 0.28)",
            marker={"size": 8, "color": "#0f5132"},
            opacity=0.45 if is_highlighted else 1.0,
        )
    )

    if selected_metric and selected_metric in metrics:
        selected_index = metrics.index(selected_metric)
        selected_display_metric = display_metrics[selected_index]
        selected_score = scores[selected_index]

        figure.add_trace(
            go.Scatterpolar(
                r=[0, selected_score],
                theta=[selected_display_metric, selected_display_metric],
                mode="lines+markers",
                name="Selected Sin",
                hovertemplate=f"{selected_metric}: {selected_score:.0f}<extra></extra>",
                line={"color": "#d97706", "width": 5},
                marker={
                    "size": [0, 18],
                    "color": ["rgba(0,0,0,0)", "#d97706"],
                    "line": {"color": "#fff7ed", "width": 2},
                },
            )
        )

        figure.add_trace(
            go.Scatterpolar(
                r=[selected_score],
                theta=[selected_display_metric],
                mode="markers",
                name="Selected Point",
                hovertemplate=f"{selected_metric}: {selected_score:.0f}<extra></extra>",
                marker={
                    "size": 26,
                    "color": "rgba(217, 119, 6, 0.20)",
                    "line": {"color": "#d97706", "width": 3},
                    "symbol": "circle",
                },
            )
        )

    figure.update_layout(
        margin={"l": 60, "r": 60, "t": 40, "b": 60},
        paper_bgcolor="white",
        font={"color": "#10261d"},
        polar={
            "radialaxis": {
                "visible": True,
                "range": [METRIC_SCORE_MIN, METRIC_SCORE_MAX],
                "tickvals": [0, 20, 40, 60, 80, 100],
                "gridcolor": "#c7d9d0",
                "linecolor": "#8ba79b",
                "tickfont": {"size": 12, "color": "#244338"},
            },
            "angularaxis": {
                "gridcolor": "#e6efeb",
                "linecolor": "#8ba79b",
                "tickfont": {"size": 13, "color": "#0f241c", "family": "Arial, sans-serif"},
            },
            "bgcolor": "#f7fbf9",
        },
        showlegend=False,
    )
    return figure


# Label mapping for radar chart — short enough to display without truncation
_RADAR_LABELS = {
    "Hidden Trade-Off": "Hidden<br>Trade-Off",
    "No Proof": "No Proof",
    "Vagueness": "Vagueness",
    "False Labels": "False Labels",
    "Irrelevance": "Irrelevance",
    "Lesser of Two Evils": "Lesser of<br>Two Evils",
    "Fibbing": "Fibbing",
}


def _wrap_metric_label(metric_name: str) -> str:
    return _RADAR_LABELS.get(metric_name, metric_name)
