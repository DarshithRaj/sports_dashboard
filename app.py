import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sports Analytics Dashboard", layout="wide")

# Load data
matches = pd.read_csv("data/matches.csv")
players = pd.read_csv("data/players.csv")

st.title("⚽ Sports Analytics Dashboard")

# ---- OVERVIEW ----
st.header("📊 League Overview")
col1, col2 = st.columns(2)

# Top scorers
top_scorers = players.sort_values("goals", ascending=False).head(5)
fig1 = px.bar(top_scorers, x="player", y="goals", color="team", title="Top 5 Scorers")
col1.plotly_chart(fig1, use_container_width=True)

# Team goals
team_goals = matches.groupby("home_team")[["home_goals"]].sum().reset_index()
team_goals.rename(columns={"home_team": "team", "home_goals": "goals"}, inplace=True)
fig2 = px.pie(team_goals, names="team", values="goals", title="Goals Distribution by Teams")
col2.plotly_chart(fig2, use_container_width=True)

# ---- PLAYER COMPARISON ----
st.header("🆚 Player Comparison")
p1, p2 = st.columns(2)
player1 = p1.selectbox("Select Player 1", players["player"].unique())
player2 = p2.selectbox("Select Player 2", players["player"].unique())

def radar_chart(player_df, name):
    categories = ["goals","assists","shots","dribbles","tackles","interceptions","rating"]
    values = [player_df[categories].values[0][i] for i in range(len(categories))]
    return values, categories

p1_data = players[players["player"]==player1]
p2_data = players[players["player"]==player2]

p1_values, categories = radar_chart(p1_data, player1)
p2_values, _ = radar_chart(p2_data, player2)

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=p1_values, theta=categories, fill='toself', name=player1))
fig_radar.add_trace(go.Scatterpolar(r=p2_values, theta=categories, fill='toself', name=player2))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Player Performance Radar")
st.plotly_chart(fig_radar, use_container_width=True)

# ---- TEAM LEADERBOARD ----
st.header("🏆 Team Leaderboard")
matches["points_home"] = matches.apply(lambda x: 3 if x["home_win"]==1 else (1 if x["draw"]==1 else 0), axis=1)
matches["points_away"] = matches.apply(lambda x: 3 if (x["home_win"]==0 and x["draw"]==0) else (1 if x["draw"]==1 else 0), axis=1)

home_table = matches.groupby("home_team").agg({"home_goals":"sum","away_goals":"sum","points_home":"sum"}).reset_index()
away_table = matches.groupby("away_team").agg({"away_goals":"sum","home_goals":"sum","points_away":"sum"}).reset_index()

home_table.rename(columns={"home_team":"team","home_goals":"GF","away_goals":"GA","points_home":"points"}, inplace=True)
away_table.rename(columns={"away_team":"team","away_goals":"GF","home_goals":"GA","points_away":"points"}, inplace=True)

league_table = pd.concat([home_table, away_table]).groupby("team").sum().reset_index()
league_table["GD"] = league_table["GF"] - league_table["GA"]
league_table = league_table.sort_values(["points","GD"], ascending=[False,False])

st.dataframe(league_table)

fig3 = px.bar(league_table, x="team", y="points", color="GD", title="Points by Team")
st.plotly_chart(fig3, use_container_width=True)

# ---- WIN PREDICTOR ----
st.header("🔮 Match Outcome Predictor")

X = matches[["home_shots","away_shots","home_possession","away_possession","home_corners","away_corners"]]
y = matches["home_win"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {acc*100:.2f}%")

colA, colB = st.columns(2)
home_shots = colA.slider("Home Shots", 0, 30, 10)
away_shots = colB.slider("Away Shots", 0, 30, 10)
home_possession = colA.slider("Home Possession %", 0, 100, 50)
away_possession = colB.slider("Away Possession %", 0, 100, 50)
home_corners = colA.slider("Home Corners", 0, 15, 5)
away_corners = colB.slider("Away Corners", 0, 15, 5)

input_data = [[home_shots, away_shots, home_possession, away_possession, home_corners, away_corners]]
prediction = model.predict_proba(input_data)[0][1]

st.metric("Home Win Probability", f"{prediction*100:.1f}%")
