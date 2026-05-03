# =============================================================
# app/pages/5_Leaderboard.py
# NFL Super Bowl Champion Predictor
# =============================================================
# PURPOSE:
#   Community leaderboard — register, pick a Super Bowl winner,
#   track Brier score week by week vs other users.
#
# THREE SECTIONS:
#   1. Auth        — register / login / logout
#   2. My Pick     — make or view SB pick + confidence level
#   3. Leaderboard — all users ranked by Brier score
#
# BRIER SCORE = (probability - outcome)^2  (lower = better)
#   outcome = 1 if your team won, 0 if not
#   Rewards honest calibration, not just picking the favourite.
# =============================================================

import hashlib
import sqlite3
import sys,os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.team_branding import (
    inject_global_css, TEAM_COLORS, logo_url,
    primary, secondary, full_name,
)

st.set_page_config(page_title="Leaderboard · NFL",
                   page_icon="🏆", layout="wide")
st.markdown(inject_global_css(), unsafe_allow_html=True)
st.markdown("""
<style>
body,.stApp{background:#07090f;color:#e2e8f0}

/* Podium */
.podium{display:flex;align-items:flex-end;justify-content:center;
        gap:12px;margin:1.25rem 0}
.pod-card{border-radius:14px;padding:1rem .875rem;
          text-align:center;border:1px solid #1a2234;
          transition:transform .2s}
.pod-card:hover{transform:translateY(-3px)}
.pod-rank{font-size:26px;margin-bottom:.35rem}
.pod-name{font-size:13px;font-weight:700;color:#fff;margin-bottom:.2rem}
.pod-score{font-size:11px;margin-top:.35rem}

/* Leaderboard rows */
.lb-row{
    display:flex;align-items:center;gap:.875rem;
    padding:.7rem 1rem;
    background:#0f1520;border:1px solid #1a2234;
    border-radius:12px;margin-bottom:6px;
    transition:border-color .2s;
}
.lb-row:hover{border-color:#2a4a6a}
.lb-rank{font-size:14px;font-weight:700;color:#4a5568;min-width:28px;text-align:right}
.lb-name{flex:1;font-size:13px;font-weight:600;color:#e2e8f0}
.lb-team{display:flex;align-items:center;gap:6px;font-size:12px;color:#8899aa}
.lb-score{font-size:15px;font-weight:800;min-width:56px;text-align:right}
.lb-bar-wrap{width:80px;height:7px;background:#1a2234;border-radius:3px;overflow:hidden}
.lb-bar{height:100%;border-radius:3px}

/* Auth form */
.auth-card{
    background:#0f1520;border:1px solid #1a2234;
    border-radius:16px;padding:1.5rem 1.75rem;
    max-width:400px;margin:0 auto;
}
.auth-card h3{font-size:18px;font-weight:700;color:#fff;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)


# ── DB helpers ───────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "users.db")

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username: str, password: str) -> tuple[bool, str]:
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.strip(), hash_pw(password)),
        )
        conn.commit(); conn.close()
        return True, "Account created!"
    except sqlite3.IntegrityError:
        return False, "Username already taken."

def login_user(username: str, password: str) -> tuple[bool, str]:
    conn  = get_conn()
    row   = conn.execute(
        "SELECT id FROM users WHERE username=? AND password_hash=?",
        (username.strip(), hash_pw(password)),
    ).fetchone()
    conn.close()
    if row:
        return True, row[0]
    return False, "Invalid username or password."

def submit_pick(user_id: int, team: str, confidence: float):
    conn = get_conn()
    conn.execute(
        "INSERT INTO picks (user_id, team, confidence) VALUES (?,?,?)",
        (user_id, team, confidence),
    )
    conn.commit(); conn.close()

def load_leaderboard() -> pd.DataFrame:
    conn = get_conn()
    df   = pd.read_sql_query("""
        SELECT u.username, p.team, p.confidence, p.brier_score, p.submitted_at
        FROM picks p JOIN users u ON p.user_id = u.id
        ORDER BY p.submitted_at DESC
    """, conn)
    conn.close()

    # Season complete — SEA won. Compute Brier if not set.
    SEA_WON = 1
    if len(df):
        df["brier_score"] = df.apply(
            lambda r: (r["confidence"] / 100 - (1 if r["team"] == "SEA" else 0)) ** 2
            if pd.isna(r["brier_score"]) else r["brier_score"],
            axis=1,
        )
        df = df.sort_values("brier_score")
    return df


# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.25rem'>
    <h1 style='font-size:26px;font-weight:800;color:#fff;margin:0'>
        🏆 Community Leaderboard
    </h1>
    <p style='color:#8899aa;font-size:13px;margin:.25rem 0 0'>
        Pick your Super Bowl champion · track your Brier score · season complete
    </p>
</div>
""", unsafe_allow_html=True)

# SB result banner
st.markdown(f"""
<div style="background:linear-gradient(135deg,#002244 0%,#001833 70%,#69BE2818 100%);
            border:1px solid #69BE2844;border-radius:14px;
            padding:1rem 1.5rem;margin-bottom:1.25rem;
            display:flex;align-items:center;gap:1rem">
    <img src="{logo_url('SEA')}"
         style="width:52px;height:52px;object-fit:contain">
    <div style="flex:1">
        <div style="font-size:16px;font-weight:800;color:#fff">
            Seattle Seahawks won Super Bowl LX ✅
        </div>
        <div style="font-size:12px;color:#69BE28;margin-top:2px">
            SEA 29 – NE 13 · Feb 8 2026 · Brier scoring complete
        </div>
    </div>
    <div style="text-align:right">
        <div style="font-size:11px;color:#8899aa">Lower Brier = better</div>
        <div style="font-size:11px;color:#8899aa">SEA @ 80% → 0.04 🏆</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_lb, tab_pick, tab_auth = st.tabs(
    ["🏅 Leaderboard", "🎯 Submit Pick", "🔐 Account"])

# ════ TAB 1 — LEADERBOARD ════════════════════════════════════
with tab_lb:
    lb_df = load_leaderboard()

    if lb_df.empty:
        st.info("No picks yet — be the first to submit!")
    else:
        # ── Podium (top 3) ───────────────────────────────────
        medals = ["🥇", "🥈", "🥉"]
        pod_heights = ["160px", "130px", "110px"]
        pod_colors  = ["#FFD700", "#C0C0C0", "#CD7F32"]
        pod_bg      = ["#1a1500", "#141414", "#120c00"]

        top3 = lb_df.head(3)
        if len(top3) >= 1:
            st.markdown('<div class="podium">', unsafe_allow_html=True)
            # Reorder: 2nd, 1st, 3rd for podium look
            order = [1, 0, 2] if len(top3) >= 3 else list(range(len(top3)))
            cols  = st.columns(len(order))
            for col_idx, rank_idx in enumerate(order):
                if rank_idx >= len(top3):
                    continue
                row        = top3.iloc[rank_idx]
                medal      = medals[rank_idx]
                clr        = pod_colors[rank_idx]
                bg         = pod_bg[rank_idx]
                team       = row["team"]
                conf_color = clr
                brier_val  = f"{row['brier_score']:.4f}"

                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="pod-card"
                         style="background:{bg};border-color:{clr}44;
                                height:{pod_heights[rank_idx]}">
                        <div class="pod-rank">{medal}</div>
                        <img src="{logo_url(team)}"
                             style="width:36px;height:36px;object-fit:contain">
                        <div class="pod-name">{row['username']}</div>
                        <div style="font-size:11px;color:#8899aa">
                            Picked {full_name(team)}
                        </div>
                        <div class="pod-score" style="color:{clr}">
                            Brier {brier_val}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### All Picks")

        min_b = lb_df["brier_score"].min() if len(lb_df) else 1
        max_b = lb_df["brier_score"].max() if len(lb_df) else 1

        for i, row in lb_df.reset_index(drop=True).iterrows():
            rank      = i + 1
            team      = row["team"]
            p_c       = primary(team)
            s_c       = secondary(team)
            brier     = row["brier_score"]
            # bar: lower brier = longer green bar
            bar_pct   = max(0, 1 - (brier - min_b) / (max_b - min_b + 1e-9))
            bar_color = ("#1D9E75" if brier < 0.10 else
                         "#EF9F27" if brier < 0.25 else "#E24B4A")
            medal_icon = (medals[rank - 1] if rank <= 3 else
                          f"<span style='color:#4a5568'>#{rank}</span>")

            st.markdown(f"""
            <div class="lb-row">
                <div class="lb-rank">{medal_icon}</div>
                <img src="{logo_url(team)}"
                     style="width:30px;height:30px;object-fit:contain">
                <div class="lb-name">{row['username']}</div>
                <div class="lb-team">
                    <span style="color:{p_c};font-weight:600">{team}</span>
                    @ {row['confidence']:.0f}%
                </div>
                <div class="lb-bar-wrap">
                    <div class="lb-bar"
                         style="width:{bar_pct*100:.1f}%;background:{bar_color}">
                    </div>
                </div>
                <div class="lb-score" style="color:{bar_color}">
                    {brier:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Pick distribution chart
        st.markdown("---")
        st.markdown("#### Pick Distribution")
        pick_counts = lb_df["team"].value_counts().head(12)
        fig_picks = go.Figure()
        for team_abbr, count in pick_counts.items():
            fig_picks.add_trace(go.Bar(
                x=[team_abbr], y=[count],
                name=team_abbr,
                marker_color=primary(team_abbr),
                marker_line_color=secondary(team_abbr),
                marker_line_width=1.5,
                hovertemplate=(
                    f"<b>{full_name(team_abbr)}</b><br>"
                    f"{count} picks<extra></extra>"
                ),
            ))
        fig_picks.update_layout(
            showlegend=False,
            paper_bgcolor="#07090f", plot_bgcolor="#07090f",
            font_color="#e2e8f0", height=240, bargap=0.25,
            margin=dict(l=20,r=20,t=10,b=20),
            xaxis=dict(gridcolor="#1a2234"),
            yaxis=dict(gridcolor="#1a2234", title="# of Picks"),
        )
        st.plotly_chart(fig_picks, use_container_width=True)

# ════ TAB 2 — SUBMIT PICK ════════════════════════════════════
with tab_pick:
    if "user_id" not in st.session_state:
        st.warning("Please log in first (Account tab) to submit a pick.")
    else:
        uid      = st.session_state["user_id"]
        uname    = st.session_state.get("username", "User")
        existing = load_leaderboard()
        user_row = existing[existing["username"] == uname] \
                   if len(existing) else pd.DataFrame()

        if len(user_row):
            pu = user_row.iloc[0]
            st.markdown(f"""
            <div style="background:#0f1520;border:1px solid #1a2234;
                        border-radius:14px;padding:1.25rem;margin-bottom:1rem;
                        display:flex;align-items:center;gap:1rem">
                <img src="{logo_url(pu['team'])}"
                     style="width:48px;height:48px;object-fit:contain">
                <div>
                    <div style="font-size:14px;font-weight:700;color:#fff">
                        Your current pick: {full_name(pu['team'])}
                    </div>
                    <div style="font-size:12px;color:#8899aa;margin-top:2px">
                        Confidence: {pu['confidence']:.0f}% ·
                        Brier Score: {pu['brier_score']:.4f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Submit or Update Your Pick")

        # Team picker with logo preview
        pick_team = st.selectbox(
            "Select your Super Bowl Champion",
            sorted(TEAM_COLORS.keys()),
            format_func=full_name,
            key="pick_team_sel",
        )

        # Preview selected team card
        pt_primary   = primary(pick_team)
        pt_secondary = secondary(pick_team)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{pt_primary}44 0%,#07090f 100%);
                    border:1px solid {pt_primary}55;border-radius:14px;
                    padding:1rem 1.25rem;margin:.75rem 0;
                    display:flex;align-items:center;gap:1rem">
            <img src="{logo_url(pick_team)}"
                 style="width:56px;height:56px;object-fit:contain">
            <div>
                <div style="font-size:18px;font-weight:800;color:#fff">
                    {full_name(pick_team)}
                </div>
                <div style="font-size:12px;color:{pt_secondary};margin-top:2px">
                    {TEAM_COLORS[pick_team]['conf']} ·
                    {TEAM_COLORS[pick_team]['div']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        confidence = st.slider(
            "Confidence level (%)",
            min_value=10, max_value=90, value=60, step=10,
            help="How confident are you in this pick? "
                 "Honest calibration beats overconfidence.",
        )

        # Live Brier preview
        brier_win  = (confidence / 100 - 1) ** 2
        brier_loss = (confidence / 100 - 0) ** 2
        b1, b2     = st.columns(2)
        b1.markdown(f"""
        <div style="background:#0a1a0a;border:1px solid #1D9E75;
                    border-radius:10px;padding:.75rem;text-align:center">
            <div style="font-size:11px;color:#8899aa">If {pick_team} wins</div>
            <div style="font-size:24px;font-weight:800;color:#1D9E75">
                {brier_win:.4f}
            </div>
            <div style="font-size:10px;color:#69BE28">Lower is better 🏆</div>
        </div>
        """, unsafe_allow_html=True)
        b2.markdown(f"""
        <div style="background:#1a0a0a;border:1px solid #E24B4A;
                    border-radius:10px;padding:.75rem;text-align:center">
            <div style="font-size:11px;color:#8899aa">If {pick_team} loses</div>
            <div style="font-size:24px;font-weight:800;color:#E24B4A">
                {brier_loss:.4f}
            </div>
            <div style="font-size:10px;color:#ff6b6b">
                Overconfident picks hurt
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if st.button("✅  Submit Pick", type="primary",
                     use_container_width=True):
            submit_pick(uid, pick_team, confidence)
            st.success(f"Pick submitted! {full_name(pick_team)} "
                       f"@ {confidence}% confidence.")
            st.rerun()

# ════ TAB 3 — ACCOUNT ════════════════════════════════════════
with tab_auth:
    if "user_id" in st.session_state:
        st.success(f"Logged in as **{st.session_state['username']}**")
        if st.button("Log out"):
            del st.session_state["user_id"]
            del st.session_state["username"]
            st.rerun()
    else:
        auth_mode = st.radio("", ["Login", "Register"],
                             horizontal=True, key="auth_mode")

        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown(f'<h3>{"🔑 Login" if auth_mode=="Login" else "✨ Create Account"}</h3>',
                    unsafe_allow_html=True)

        username_inp = st.text_input("Username", key="auth_user",
                                     placeholder="your_username")
        password_inp = st.text_input("Password", type="password",
                                     key="auth_pass",
                                     placeholder="••••••••")

        if auth_mode == "Login":
            if st.button("Login", type="primary", use_container_width=True):
                ok, result = login_user(username_inp, password_inp)
                if ok:
                    st.session_state["user_id"]   = result
                    st.session_state["username"]  = username_inp.strip()
                    st.success("Logged in! 🎉")
                    st.rerun()
                else:
                    st.error(result)
        else:
            if st.button("Create Account", type="primary",
                         use_container_width=True):
                if len(username_inp.strip()) < 3:
                    st.error("Username must be at least 3 characters.")
                elif len(password_inp) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, msg = register_user(username_inp, password_inp)
                    if ok:
                        st.success(msg + " Please log in.")
                    else:
                        st.error(msg)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="background:#0f1520;border:1px solid #1a2234;
                    border-radius:12px;padding:.875rem 1rem">
            <div style="font-size:13px;font-weight:700;color:#e2e8f0;
                        margin-bottom:.35rem">How Brier Scoring Works</div>
            <div style="font-size:12px;color:#8899aa;line-height:1.6">
                Brier = (confidence/100 − outcome)² · Lower is better<br>
                Pick SEA @ 80% → SEA wins → Brier = 0.04 🏆<br>
                Pick KC @ 70% → SEA wins → Brier = 0.49 ❌<br>
                Honest calibration always beats overconfidence.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Known Super Bowl winners by season
KNOWN_WINNERS = {
    2025: "SEA",  # Super Bowl LX
    2024: "PHI",
    2023: "KC",
    2022: "KC",
    2021: "LAR",
    2020: "TB",
    2019: "KC",
    2018: "NE",
}


# =============================================================
# DATABASE HELPERS
# =============================================================

def get_conn():
    db_path = CFG.APP["db_path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO users (username, pw_hash) VALUES (?,?)",
            (username.strip(), hash_pw(password)),
        )
        conn.commit()
        conn.close()
        return True, f"Account created! Welcome, {username}."
    except sqlite3.IntegrityError:
        return False, "Username already taken."
    except Exception as e:
        return False, f"Registration failed: {e}"


def login_user(username, password):
    try:
        conn = get_conn()
        cur  = conn.execute(
            "SELECT user_id FROM users WHERE username=? AND pw_hash=?",
            (username.strip(), hash_pw(password)),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return True, f"Welcome back, {username}!", row[0]
        return False, "Invalid username or password.", None
    except Exception as e:
        return False, f"Login failed: {e}", None


def get_user_pick(user_id, season):
    try:
        conn = get_conn()
        cur  = conn.execute(
            "SELECT team, confidence, created_at FROM picks "
            "WHERE user_id=? AND season=?",
            (user_id, season),
        )
        row = cur.fetchone()
        conn.close()
        return {"team": row[0], "confidence": row[1], "created_at": row[2]} \
               if row else None
    except Exception:
        return None


def save_pick(user_id, season, team, confidence):
    try:
        conn = get_conn()
        conn.execute(
            """
            INSERT INTO picks (user_id, season, team, confidence)
            VALUES (?,?,?,?)
            ON CONFLICT(user_id, season)
            DO UPDATE SET team=excluded.team,
                          confidence=excluded.confidence,
                          created_at=datetime('now')
            """,
            (user_id, season, team, confidence),
        )
        conn.commit()
        conn.close()
        return True, f"Pick saved — {team} at {confidence}% confidence."
    except Exception as e:
        return False, f"Could not save pick: {e}"


def get_all_picks(season):
    try:
        conn = get_conn()
        df   = pd.read_sql_query(
            """
            SELECT u.username, p.team, p.confidence, p.created_at
            FROM picks p JOIN users u ON p.user_id = u.user_id
            WHERE p.season=?
            ORDER BY p.created_at ASC
            """,
            conn, params=(season,),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_leaderboard(season):
    picks = get_all_picks(season)
    if len(picks) == 0:
        return pd.DataFrame()

    actual = KNOWN_WINNERS.get(season)
    if actual:
        picks["outcome"]= (picks["team"] == actual).astype(int)
        picks["prob"]   = picks["confidence"] / 100.0
        picks["brier"]  = (picks["prob"] - picks["outcome"]) ** 2
        picks["correct"]= picks["team"] == actual
        picks = picks.sort_values("brier").reset_index(drop=True)
    else:
        picks["brier"]   = None
        picks["correct"] = None

    picks.insert(0, "rank", range(1, len(picks)+1))
    return picks


def total_users():
    try:
        conn = get_conn()
        n    = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


# =============================================================
# CHART HELPERS
# =============================================================

def pick_dist_chart(picks_df):
    if len(picks_df) == 0:
        return None
    tc = picks_df["team"].value_counts().reset_index()
    tc.columns = ["team","picks"]
    tc = tc.sort_values("picks", ascending=True)
    fig = go.Figure(go.Bar(
        x=tc["picks"], y=tc["team"], orientation="h",
        marker_color="#3498db",
        text=tc["picks"], textposition="outside",
    ))
    fig.update_layout(
        title="SB Pick Distribution",
        xaxis_title="Users", yaxis_title="",
        height=max(260, len(tc)*28+80),
        margin=dict(l=60,r=60,t=60,b=40), showlegend=False,
    )
    return fig


def confidence_hist(picks_df):
    if len(picks_df) == 0:
        return None
    fig = go.Figure(go.Histogram(
        x=picks_df["confidence"], nbinsx=9,
        marker_color="#9b59b6", opacity=0.8,
    ))
    fig.update_layout(
        title="Confidence Level Distribution",
        xaxis_title="Confidence %", yaxis_title="Users",
        height=260, margin=dict(l=60,r=40,t=60,b=40), showlegend=False,
    )
    return fig


# =============================================================
# PICK FORM
# =============================================================

def render_pick_form(user_id, season, update=False):
    """Render team picker + confidence slider."""
    try:
        from app.main import load_everything
        bundle   = load_everything()
        sb_probs = bundle.get("sb_probs", pd.DataFrame())
        if len(sb_probs) > 0:
            teams = sb_probs.sort_values("sb_prob",ascending=False)["team"].tolist()
            extra = [t for t in CFG.ALL_TEAMS if t not in teams]
            teams = teams + extra
        else:
            teams    = CFG.ALL_TEAMS
            sb_probs = pd.DataFrame()
    except Exception:
        teams    = CFG.ALL_TEAMS
        sb_probs = pd.DataFrame()

    st.markdown("Teams sorted by current Super Bowl probability (highest first).")

    fk = f"form_{'upd' if update else 'new'}"
    c1, c2 = st.columns(2)

    with c1:
        picked = st.selectbox(
            "🏈 Pick your Super Bowl champion",
            options=teams, key=f"pt_{fk}",
        )
    with c2:
        conf = st.slider(
            "💯 Confidence level",
            min_value=10, max_value=90, value=50, step=10,
            key=f"pc_{fk}",
            help="Be honest — Brier score rewards calibration.",
        )

    # Model probability context
    if len(sb_probs) > 0:
        row = sb_probs[sb_probs["team"] == picked]
        if len(row) > 0:
            mp = float(row.iloc[0]["sb_prob"])
            diff = abs(conf - mp)
            if diff > 30:
                st.info(
                    f"💡 Model gives {picked} {mp:.1f}% SB probability. "
                    f"Your confidence is {conf}% — big gap!"
                )
            else:
                st.success(
                    f"✓ Model probability: {mp:.1f}% — your pick aligns."
                )

    # Live Brier preview
    p = conf / 100.0
    st.markdown(
        f"**If {picked} wins:** Brier = `{(p-1)**2:.4f}`  "
        f"| **If they lose:** Brier = `{(p-0)**2:.4f}`"
    )

    label = "💾 Update Pick" if update else "✅ Submit Pick"
    if st.button(label, type="primary", key=f"sub_{fk}"):
        ok, msg = save_pick(user_id, season, picked, conf)
        if ok:
            st.success(msg)
            st.balloons()
            st.rerun()
        else:
            st.error(msg)


# =============================================================
# MAIN PAGE
# =============================================================

def main():

    st.title("🏆 Community Leaderboard")
    st.markdown(
        "Register, make your Super Bowl pick, and track your "
        "prediction accuracy with Brier scoring against other users."
    )
    st.divider()

    # Season selector
    sel_season = st.selectbox(
        "Season",
        options=[2025, 2024, 2023, 2022],
        index=0,
        key="lb_season",
    )
    st.divider()

    # Auth state
    logged_in = st.session_state.get("logged_in_user") is not None
    user_id   = st.session_state.get("user_id")
    username  = st.session_state.get("logged_in_user")

    # ===========================================================
    # AUTH
    # ===========================================================

    if not logged_in:
        st.subheader("🔐 Register or Login")
        tab_login, tab_reg = st.tabs(["Login", "Register"])

        with tab_login:
            u = st.text_input("Username", key="li_user")
            p = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login", type="primary", key="li_btn"):
                if u and p:
                    ok, msg, uid = login_user(u, p)
                    if ok:
                        st.session_state["logged_in_user"] = u
                        st.session_state["user_id"]        = uid
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Enter username and password.")

        with tab_reg:
            ru   = st.text_input("Choose a username", key="ru")
            rp   = st.text_input("Password (min 6 chars)", type="password", key="rp")
            rpc  = st.text_input("Confirm password",    type="password", key="rpc")
            if st.button("Create Account", type="primary", key="rb"):
                if not ru or not rp:
                    st.warning("Fill in all fields.")
                elif rp != rpc:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(ru, rp)
                    if ok:
                        _, _, uid = login_user(ru, rp)
                        st.session_state["logged_in_user"] = ru
                        st.session_state["user_id"]        = uid
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.divider()

    else:
        # Logged-in header
        hc1, hc2 = st.columns([4,1])
        with hc1:
            st.subheader(f"👤 {username}")
        with hc2:
            if st.button("Logout", key="lo_btn"):
                st.session_state["logged_in_user"] = None
                st.session_state["user_id"]        = None
                st.rerun()

        st.divider()

        # ===========================================================
        # MY PICK
        # ===========================================================

        st.subheader(f"🎯 My Pick — {sel_season}")
        existing = get_user_pick(user_id, sel_season)

        if existing:
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("My Pick",      existing["team"])
            pc2.metric("Confidence",   f"{existing['confidence']}%")
            pc3.metric("Submitted",    existing["created_at"][:10])

            actual = KNOWN_WINNERS.get(sel_season)
            if actual:
                outcome = 1 if existing["team"] == actual else 0
                prob    = existing["confidence"] / 100.0
                brier   = (prob - outcome) ** 2

                bc1, bc2 = st.columns(2)
                bc1.metric("My Brier Score", f"{brier:.4f}",
                           help="Lower = better. Random baseline = 0.25")
                with bc2:
                    if existing["team"] == actual:
                        st.success(f"✅ Correct! {actual} won!")
                    else:
                        st.error(f"❌ {actual} won. Better luck next year!")

            # Allow update for current season
            CURRENT = CFG.ADDITIONAL_TEST
            if sel_season == CURRENT:
                with st.expander("✏️ Change my pick"):
                    render_pick_form(user_id, sel_season, update=True)
        else:
            CURRENT = CFG.ADDITIONAL_TEST
            if sel_season == CURRENT:
                st.info(f"No pick yet for {sel_season}.")
                render_pick_form(user_id, sel_season, update=False)
            else:
                st.info(f"No pick recorded for {sel_season} (season over).")

        st.divider()

    # ===========================================================
    # LEADERBOARD
    # ===========================================================

    st.subheader(f"🏆 Community Leaderboard — {sel_season}")
    st.caption(f"Registered users: {total_users()}")

    lb_df = get_leaderboard(sel_season)

    if len(lb_df) == 0:
        st.info(
            "No picks yet for this season. "
            "Register and be the first to make a pick!"
        )
    else:
        # Summary row
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Total Picks", len(lb_df))

        popular     = lb_df["team"].value_counts().idxmax()
        popular_pct = lb_df["team"].value_counts().max()/len(lb_df)*100
        s2.metric("Most Picked", f"{popular} ({popular_pct:.0f}%)")
        s3.metric("Avg Confidence", f"{lb_df['confidence'].mean():.0f}%")

        if "correct" in lb_df.columns and lb_df["correct"].notna().any():
            s4.metric("Picked Correctly",
                      f"{lb_df['correct'].mean()*100:.0f}%")
        else:
            s4.metric("Season", "In progress")

        st.divider()

        # Table
        cols = ["rank","username","team","confidence"]
        if "brier" in lb_df.columns and lb_df["brier"].notna().any():
            cols += ["brier","correct"]

        disp = lb_df[cols].copy().rename(columns={
            "rank":"Rank","username":"User","team":"SB Pick",
            "confidence":"Confidence %","brier":"Brier Score",
            "correct":"Correct",
        })
        if "Correct" in disp.columns:
            disp["Correct"] = disp["Correct"].map(
                {True:"✅", False:"❌", None:"—"}
            )

        def hl(row):
            r = row.get("Rank",99)
            if r == 1: return ["background-color:#ffd700"] * len(row)
            if r == 2: return ["background-color:#c0c0c0"] * len(row)
            if r == 3: return ["background-color:#cd7f32"] * len(row)
            return [""] * len(row)

        fmt = {}
        if "Brier Score"   in disp.columns: fmt["Brier Score"]   = "{:.4f}"
        if "Confidence %" in disp.columns: fmt["Confidence %"] = "{:d}%"

        st.dataframe(
            disp.style.apply(hl, axis=1).format(fmt, na_rep="—"),
            use_container_width=True,
            height=min(600, 45+len(disp)*36),
        )
        st.caption(
            "🥇Gold=1st | 🥈Silver=2nd | 🥉Bronze=3rd | "
            "Lower Brier = better calibrated prediction."
        )

        st.divider()

        # Analysis charts
        st.subheader("📊 Community Pick Analysis")
        ch1, ch2 = st.columns(2)
        with ch1:
            f1 = pick_dist_chart(lb_df)
            if f1: st.plotly_chart(f1, use_container_width=True)
        with ch2:
            f2 = confidence_hist(lb_df)
            if f2: st.plotly_chart(f2, use_container_width=True)

    st.divider()

    with st.expander("ℹ️ How Brier Scoring Works"):
        st.markdown("""
**Brier Score = (your probability − actual outcome)²**

- outcome = **1** if your team won, **0** if not
- probability = your confidence / 100

**2025 examples (SEA won Super Bowl LX):**

| User | Pick | Confidence | Brier |
|------|------|-----------|-------|
| Alice | SEA | 80% | (0.80−1)² = **0.04** 🏆 |
| Bob   | SEA | 40% | (0.40−1)² = **0.36** |
| Carol | KC  | 70% | (0.70−0)² = **0.49** |
| Dave  | KC  | 30% | (0.30−0)² = **0.09** ✓ |

Dave picked wrong but was honest about his uncertainty (30%).
Carol was overconfident in the wrong team (70%).
Dave beats Carol. **Calibration beats boldness.**

Random always-50% baseline = **0.25**. Beat that to prove skill.
        """)


if __name__ == "__main__" or True:
    main()