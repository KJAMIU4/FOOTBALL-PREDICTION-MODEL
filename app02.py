import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# 🚀 Load model and data (cached for performance)
# ============================================
@st.cache_data
def load_resources():
    model = joblib.load("football_model.pkl")
    data = pd.read_csv("data_processed.csv")
    team_encoder = joblib.load("team_encoder.pkl")
    return model, data, team_encoder

model, data, team_encoder = load_resources()

# ============================================
# ⚙️ Helper Functions
# ============================================

def recent_form_weighted(data, team_col, goals_for, goals_against, window=5):
    """Compute weighted recent form (last N matches)."""
    data = data.copy()
    data["Points"] = np.where(
        data[goals_for] > data[goals_against],
        3,
        np.where(data[goals_for] == data[goals_against], 1, 0),
    )
    form = (
        data.groupby(team_col)["Points"]
        .apply(lambda x: x.rolling(window, min_periods=1)
               .apply(lambda s: np.average(s, weights=np.arange(1, len(s) + 1)), raw=False))
        .reset_index(level=0, drop=True)
    )
    return form


def head_to_head(data):
    """Compute head-to-head average stats between teams."""
    h2h = (
        data.groupby(["HomeTeam", "AwayTeam"])
        .agg({
            "Home Goals": "mean",
            "Away Goals": "mean",
            "HomeWin": "mean"
        })
        .rename(columns={
            "Home Goals": "AvgHomeGoals",
            "Away Goals": "AvgAwayGoals",
            "HomeWin": "HeadToHeadWinRate"
        })
        .reset_index()
    )
    return h2h


def calculate_relative_strength(data, home_team, away_team):
    """Fetch relative team strength for a given pair."""
    row = data.loc[
        (data["HomeTeam"] == home_team) & (data["AwayTeam"] == away_team),
        "RelativeTeamStrength"
    ]
    return row.mean() if not row.empty else 0.0


def get_recent_form(data, team, home=True):
    """Return average recent form for a team (home or away)."""
    if home:
        col = "homeform"
        mask = data["HomeTeam"] == team
    else:
        col = "awayform"
        mask = data["AwayTeam"] == team

    values = data.loc[mask, col]
    return values.mean() if not values.empty else 0.5  # neutral default


def get_head_to_head_winrate(data, home_team, away_team):
    """Return historical head-to-head win rate (safe version)."""
    if "HeadToHeadWinRate" not in data.columns:
        return 0.5  # neutral fallback

    mask = (data["HomeTeam"] == home_team) & (data["AwayTeam"] == away_team)
    values = data.loc[mask, "HeadToHeadWinRate"]
    return values.mean() if not values.empty else 0.5


# ============================================
# 🧮 Feature Engineering on Dataset
# ============================================

# Compute home/away form
if "homeform" not in data.columns:
    data["homeform"] = recent_form_weighted(data, "HomeTeam", "Home Goals", "Away Goals")
if "awayform" not in data.columns:
    data["awayform"] = recent_form_weighted(data, "AwayTeam", "Away Goals", "Home Goals")

# Relative team strength
if "RelativeTeamStrength" not in data.columns:
    data["RelativeTeamStrength"] = data["homeform"] - data["awayform"]

# Merge head-to-head stats
h2h_stats = head_to_head(data)
data = data.merge(h2h_stats, on=["HomeTeam", "AwayTeam"], how="left")

# ============================================
# 🎛️ Streamlit User Interface
# ============================================

st.title("⚽ Football Match Outcome Predictor")
st.write("Predicts whether the **Home Team** will win or not.")

teams = sorted(data["HomeTeam"].unique())

home_team = st.selectbox("🏠 Select Home Team", teams)
away_team = st.selectbox("🛫 Select Away Team", teams)

if home_team == away_team:
    st.warning("Home and Away teams cannot be the same ❌")
    st.stop()

hour = st.slider("⏰ Kick-off Hour", 0, 23, 15)

# ============================================
# 🧠 Feature Calculation
# ============================================

home_id = team_encoder.transform([home_team])[0]
away_id = team_encoder.transform([away_team])[0]

relative_strength = calculate_relative_strength(data, home_team, away_team)
home_form = get_recent_form(data, home_team, home=True)
away_form = get_recent_form(data, away_team, home=False)
h2h_winrate = get_head_to_head_winrate(data, home_team, away_team)
weighted_perf = 0.6 * home_form + 0.4 * (1 - away_form)

# Display features
st.subheader("📊 Feature Summary")
st.write(pd.DataFrame({
    "Home Team": [home_team],
    "Away Team": [away_team],
    "Home ID": [home_id],
    "Away ID": [away_id],
    "Kick-off Hour": [hour],
    "Home Form": [home_form],
    "Away Form": [away_form],
    "Weighted Performance": [weighted_perf],
    "Relative Strength": [relative_strength],
    "Head-to-Head Win Rate": [h2h_winrate],
}))

# ============================================
# 🧩 Model Prediction
# ============================================

if st.button("🔮 Predict Outcome"):
    features = np.array([[home_id, away_id, hour,
                          weighted_perf, relative_strength, h2h_winrate]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ **Prediction:** Home Team Wins! (Confidence: {proba:.2%})")
    else:
        st.error(f"❌ **Prediction:** Away Win or Draw (Confidence: {(1 - proba):.2%})")
