# =============================================================
# app/team_branding.py
# NFL Team Colors, Logos & Branding
# =============================================================
# Logos via ESPN CDN — no API key needed, always up-to-date.
# Primary/secondary colors are official NFL team brand colors.
# =============================================================

# ESPN logo CDN base
_ESPN = "https://a.espncdn.com/i/teamlogos/nfl/500"

# ESPN abbreviation overrides (a few differ from nflreadpy)
_ESPN_ABBR = {
    "JAX": "jax", "LAC": "lac", "LAR": "lar",
    "LV":  "lv",  "NE":  "ne",  "SF":  "sf",
    "TB":  "tb",  "WAS": "wsh",
}

def logo_url(team: str) -> str:
    """Return ESPN CDN logo URL for a team abbreviation."""
    abbr = _ESPN_ABBR.get(team, team).lower()
    return f"{_ESPN}/{abbr}.png"


# fmt: off
TEAM_COLORS: dict[str, dict] = {
    # ── AFC East ──────────────────────────────────────────────
    "BUF": {"primary": "#00338D", "secondary": "#C60C30", "name": "Buffalo Bills",       "conf": "AFC", "div": "East"},
    "MIA": {"primary": "#008E97", "secondary": "#FC4C02", "name": "Miami Dolphins",      "conf": "AFC", "div": "East"},
    "NE":  {"primary": "#002244", "secondary": "#C60C30", "name": "New England Patriots","conf": "AFC", "div": "East"},
    "NYJ": {"primary": "#125740", "secondary": "#FFFFFF", "name": "New York Jets",       "conf": "AFC", "div": "East"},
    # ── AFC North ─────────────────────────────────────────────
    "BAL": {"primary": "#241773", "secondary": "#9E7C0C", "name": "Baltimore Ravens",    "conf": "AFC", "div": "North"},
    "CIN": {"primary": "#FB4F14", "secondary": "#000000", "name": "Cincinnati Bengals",  "conf": "AFC", "div": "North"},
    "CLE": {"primary": "#FF3C00", "secondary": "#311D00", "name": "Cleveland Browns",    "conf": "AFC", "div": "North"},
    "PIT": {"primary": "#101820", "secondary": "#FFB612", "name": "Pittsburgh Steelers", "conf": "AFC", "div": "North"},
    # ── AFC South ─────────────────────────────────────────────
    "HOU": {"primary": "#03202F", "secondary": "#A71930", "name": "Houston Texans",      "conf": "AFC", "div": "South"},
    "IND": {"primary": "#002C5F", "secondary": "#A2AAAD", "name": "Indianapolis Colts",  "conf": "AFC", "div": "South"},
    "JAX": {"primary": "#006778", "secondary": "#9F792C", "name": "Jacksonville Jaguars","conf": "AFC", "div": "South"},
    "TEN": {"primary": "#0C2340", "secondary": "#4B92DB", "name": "Tennessee Titans",    "conf": "AFC", "div": "South"},
    # ── AFC West ──────────────────────────────────────────────
    "DEN": {"primary": "#FB4F14", "secondary": "#002244", "name": "Denver Broncos",      "conf": "AFC", "div": "West"},
    "KC":  {"primary": "#E31837", "secondary": "#FFB81C", "name": "Kansas City Chiefs",  "conf": "AFC", "div": "West"},
    "LV":  {"primary": "#000000", "secondary": "#A5ACAF", "name": "Las Vegas Raiders",   "conf": "AFC", "div": "West"},
    "LAC": {"primary": "#0080C6", "secondary": "#FFC20E", "name": "LA Chargers",         "conf": "AFC", "div": "West"},
    # ── NFC East ──────────────────────────────────────────────
    "DAL": {"primary": "#003594", "secondary": "#869397", "name": "Dallas Cowboys",      "conf": "NFC", "div": "East"},
    "NYG": {"primary": "#0B2265", "secondary": "#A71930", "name": "New York Giants",     "conf": "NFC", "div": "East"},
    "PHI": {"primary": "#004C54", "secondary": "#A5ACAF", "name": "Philadelphia Eagles", "conf": "NFC", "div": "East"},
    "WAS": {"primary": "#5A1414", "secondary": "#FFB612", "name": "Washington Commanders","conf":"NFC", "div": "East"},
    # ── NFC North ─────────────────────────────────────────────
    "CHI": {"primary": "#0B162A", "secondary": "#C83803", "name": "Chicago Bears",       "conf": "NFC", "div": "North"},
    "DET": {"primary": "#0076B6", "secondary": "#B0B7BC", "name": "Detroit Lions",       "conf": "NFC", "div": "North"},
    "GB":  {"primary": "#203731", "secondary": "#FFB612", "name": "Green Bay Packers",   "conf": "NFC", "div": "North"},
    "MIN": {"primary": "#4F2683", "secondary": "#FFC62F", "name": "Minnesota Vikings",   "conf": "NFC", "div": "North"},
    # ── NFC South ─────────────────────────────────────────────
    "ATL": {"primary": "#A71930", "secondary": "#000000", "name": "Atlanta Falcons",     "conf": "NFC", "div": "South"},
    "CAR": {"primary": "#0085CA", "secondary": "#101820", "name": "Carolina Panthers",   "conf": "NFC", "div": "South"},
    "NO":  {"primary": "#D3BC8D", "secondary": "#101820", "name": "New Orleans Saints",  "conf": "NFC", "div": "South"},
    "TB":  {"primary": "#D50A0A", "secondary": "#FF7900", "name": "Tampa Bay Buccaneers","conf": "NFC", "div": "South"},
    # ── NFC West ──────────────────────────────────────────────
    "ARI": {"primary": "#97233F", "secondary": "#000000", "name": "Arizona Cardinals",   "conf": "NFC", "div": "West"},
    "LAR": {"primary": "#003594", "secondary": "#FFA300", "name": "LA Rams",             "conf": "NFC", "div": "West"},
    "SF":  {"primary": "#AA0000", "secondary": "#B3995D", "name": "San Francisco 49ers", "conf": "NFC", "div": "West"},
    "SEA": {"primary": "#002244", "secondary": "#69BE28", "name": "Seattle Seahawks",    "conf": "NFC", "div": "West"},
}
# fmt: on

def get_team(team: str) -> dict:
    """Return branding dict for a team, with fallback defaults."""
    return TEAM_COLORS.get(team, {
        "primary": "#1a1a2e", "secondary": "#e94560",
        "name": team, "conf": "NFL", "div": "—",
    })

def primary(team: str) -> str:
    return get_team(team)["primary"]

def secondary(team: str) -> str:
    return get_team(team)["secondary"]

def full_name(team: str) -> str:
    return get_team(team)["name"]

def conference(team: str) -> str:
    return get_team(team)["conf"]


# ── Global Streamlit CSS injector ─────────────────────────────
def inject_global_css() -> str:
    """Return <style> block for the whole app."""
    return """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Remove default Streamlit padding ── */
.block-container { padding-top: 1.5rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0f1117;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 0.875rem 1rem !important;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8899aa !important;
}
[data-testid="stMetricValue"] { font-weight: 700 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #070b14 !important;
    border-right: 1px solid #1a2234 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #c8d6e8 !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #0f1117 !important;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1e2a3a !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* ── Divider ── */
hr { border-color: #1a2234 !important; }

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
"""


def team_card_css(team: str) -> str:
    """Return CSS overrides scoped to a team's colors."""
    p = primary(team)
    s = secondary(team)
    return f"""
<style>
.team-header {{
    background: linear-gradient(135deg, {p} 0%, {p}cc 60%, {s}33 100%);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
    border: 1px solid {s}44;
}}
.team-header img {{
    width: 72px;
    height: 72px;
    object-fit: contain;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.5));
}}
.team-header h1 {{
    margin: 0;
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
}}
.team-header .sub {{
    font-size: 13px;
    color: {s};
    font-weight: 500;
    margin-top: 2px;
}}
.metric-pill {{
    display: inline-block;
    background: {p}22;
    border: 1px solid {p}55;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    color: {s};
    margin: 2px;
}}
</style>
"""