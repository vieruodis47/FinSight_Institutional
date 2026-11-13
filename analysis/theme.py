"""Shared institutional Plotly theme for FinSight."""
import plotly.io as pio

"""Shared institutional Plotly theme for FinSight (Light Financial Edition)."""

BG = "#F8F9FB"
PANEL = "#FFFFFF"
PANEL_ALT = "#F1F4F9"
BORDER = "#E2E8F0"
TEXT = "#0F172A"
MUTED = "#475569"
ORANGE = "#EA580C"
AMBER = "#D97706"
BLUE = "#1E3A8A"
TEAL = "#0F766E"
GREEN = "#15803D"
RED = "#DC2626"

# Keep legacy aliases so existing imports never break
NAVY = BG


def apply_plotly_theme(fig, height=390):
    """Apply a crisp white & financial-orange institutional style."""
    if fig is None:
        return None
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Sans, Arial, sans-serif", color=TEXT, size=12),
        title=dict(font=dict(size=15, color=TEXT), x=0.02, xanchor="left"),
        height=height,
        margin=dict(l=48, r=28, t=62, b=48),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=ORANGE,
            font=dict(color=TEXT, family="DM Sans, Arial, sans-serif"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=0,
            font=dict(color=MUTED, size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(15, 23, 42, 0.06)",
            zerolinecolor="rgba(15, 23, 42, 0.14)",
            linecolor=BORDER,
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
        ),
        yaxis=dict(
            gridcolor="rgba(15, 23, 42, 0.06)",
            zerolinecolor="rgba(15, 23, 42, 0.14)",
            linecolor=BORDER,
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED),
        ),
    )
    return fig
