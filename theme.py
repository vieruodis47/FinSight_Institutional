"""Shared institutional Plotly theme for FinSight."""
import plotly.io as pio

NAVY = "#0B1220"
PANEL = "#111C2E"
PANEL_ALT = "#162338"
BORDER = "#26364D"
TEXT = "#E7EDF5"
MUTED = "#91A1B7"
BLUE = "#5B9BD5"
TEAL = "#42C7B7"
GREEN = "#39B980"
RED = "#E16B75"
AMBER = "#E6B566"

def apply_plotly_theme(fig, height=390):
    """Apply a consistent dark institutional style without changing chart data."""
    if fig is None:
        return None
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color=TEXT, size=12),
        title=dict(font=dict(size=15, color=TEXT), x=0.02, xanchor="left"),
        height=height,
        margin=dict(l=48, r=28, t=62, b=48),
        hoverlabel=dict(
            bgcolor=PANEL_ALT, bordercolor=BORDER,
            font=dict(color=TEXT, family="Inter, Arial, sans-serif")
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(color=MUTED, size=11)
        ),
        xaxis=dict(
            gridcolor="rgba(145,161,183,0.12)",
            zerolinecolor="rgba(145,161,183,0.18)",
            linecolor=BORDER, tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
        ),
        yaxis=dict(
            gridcolor="rgba(145,161,183,0.12)",
            zerolinecolor="rgba(145,161,183,0.18)",
            linecolor=BORDER, tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
        ),
    )
    return fig
