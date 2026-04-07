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
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from config import CFG

st.set_page_config(
    page_title = "Leaderboard — NFL SB Predictor",
    page_icon  = "🏆",
    layout     = "wide",
)

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