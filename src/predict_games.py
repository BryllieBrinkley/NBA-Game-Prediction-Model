import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import scoreboardv2
from train_model import build_feature_table

# load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nba_games_multi_season.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "winner_model.pkl")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
model = joblib.load(MODEL_PATH)

# Mapping of team IDs to names for display purposes
team_names = {
    1610612737: "Hawks",
    1610612738: "Celtics",
    1610612739: "Cavaliers",
    1610612740: "Pelicans",
    1610612741: "Bulls",
    1610612742: "Mavericks",
    1610612743: "Nuggets",
    1610612744: "Warriors",
    1610612745: "Rockets",
    1610612746: "Clippers",
    1610612747: "Lakers",
    1610612748: "Heat",
    1610612749: "Bucks",
    1610612750: "Timberwolves",
    1610612751: "Nets",
    1610612752: "Knicks",
    1610612753: "Magic",
    1610612754: "Pacers",
    1610612755: "76ers",
    1610612756: "Suns",
    1610612757: "Trail Blazers",
    1610612758: "Kings",
    1610612759: "Spurs",
    1610612760: "Thunder",
    1610612761: "Raptors",
    1610612762: "Jazz",
    1610612763: "Grizzlies",
    1610612764: "Wizards",
    1610612765: "Pistons",
    1610612766: "Hornets",
}

vegas_team_names = {
    "Hawks": "Atlanta Hawks",
    "Celtics": "Boston Celtics",
    "Cavaliers": "Cleveland Cavaliers",
    "Pelicans": "New Orleans Pelicans",
    "Bulls": "Chicago Bulls",
    "Mavericks": "Dallas Mavericks",
    "Nuggets": "Denver Nuggets",
    "Warriors": "Golden State Warriors",
    "Rockets": "Houston Rockets",
    "Clippers": "Los Angeles Clippers",
    "Lakers": "Los Angeles Lakers",
    "Heat": "Miami Heat",
    "Bucks": "Milwaukee Bucks",
    "Timberwolves": "Minnesota Timberwolves",
    "Nets": "Brooklyn Nets",
    "Knicks": "New York Knicks",
    "Magic": "Orlando Magic",
    "Pacers": "Indiana Pacers",
    "76ers": "Philadelphia 76ers",
    "Suns": "Phoenix Suns",
    "Trail Blazers": "Portland Trail Blazers",
    "Kings": "Sacramento Kings",
    "Spurs": "San Antonio Spurs",
    "Thunder": "Oklahoma City Thunder",
    "Raptors": "Toronto Raptors",
    "Jazz": "Utah Jazz",
    "Grizzlies": "Memphis Grizzlies",
    "Wizards": "Washington Wizards",
    "Pistons": "Detroit Pistons",
    "Hornets": "Charlotte Hornets",
}

# get games for today from NBA API
def fetch_today_games() -> pd.DataFrame:
    today = datetime.today().strftime("%m/%d/%Y")
    games = scoreboardv2.ScoreboardV2(game_date=today)
    games_df = games.game_header.get_data_frame()
    return games_df[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]]

# get vegas odds for today's games, with caching and error handling
def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def normalize_prob(home_ml: float, away_ml: float) -> tuple[float, float]:
    home_prob = american_to_implied_prob(home_ml)
    away_prob = american_to_implied_prob(away_ml)
    if pd.isna(home_prob) or pd.isna(away_prob):
        return np.nan, np.nan
    total = home_prob + away_prob
    if total == 0:
        return np.nan, np.nan
    return home_prob / total, away_prob / total

# get vegas odds
def fetch_vegas_odds() -> pd.DataFrame:
    """Fetch Vegas odds for today's NBA games, with caching."""
    cache_file = "data/vegas_cache.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    if not ODDS_API_KEY:
        return pd.DataFrame()
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "bookmakers": "fanduel",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    rows = []
    for game in data:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        home_ml = None
        away_ml = None
        home_spread = None
        away_spread = None
        for book in bookmakers:
            markets = {m["key"]: m for m in book.get("markets", [])}
            if "h2h" not in markets or "spreads" not in markets:
                continue
            for outcome in markets["h2h"].get("outcomes", []):
                if outcome["name"] == home_team:
                    home_ml = outcome.get("price")
                elif outcome["name"] == away_team:
                    away_ml = outcome.get("price")
            for outcome in markets["spreads"].get("outcomes", []):
                if outcome["name"] == home_team:
                    home_spread = outcome.get("point")
                elif outcome["name"] == away_team:
                    away_spread = outcome.get("point")
            break
        if home_ml is None or away_ml is None:
            continue
        vegas_home_prob, vegas_away_prob = normalize_prob(home_ml, away_ml)
        rows.append(
            {
                "home_team_name": home_team,
                "away_team_name": away_team,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "home_spread": home_spread,
                "away_spread": away_spread,
                "vegas_home_prob": vegas_home_prob,
                "vegas_away_prob": vegas_away_prob,
            }
        )
    odds_df = pd.DataFrame(rows)
    odds_df.to_csv(cache_file, index=False)
    return odds_df


# --- Fetch NBA Injury Report ---
def fetch_injuries() -> pd.DataFrame:
    """Fetch NBA injury report from Rotowire."""
    url = "https://www.rotowire.com/basketball/tables/injury-report.php"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df.columns = ["Player", "Team", "Status", "Injury"]
        return df
    except Exception:
        return pd.DataFrame()

# --- Compute Team Injury Score ---
def compute_team_injury_score(injuries_df: pd.DataFrame) -> dict:
    """Compute injury score by team."""

    if injuries_df.empty:
        return {}

    team_map = {
        "ATL": "Hawks",
        "BOS": "Celtics",
        "CLE": "Cavaliers",
        "NOP": "Pelicans",
        "CHI": "Bulls",
        "DAL": "Mavericks",
        "DEN": "Nuggets",
        "GSW": "Warriors",
        "HOU": "Rockets",
        "LAC": "Clippers",
        "LAL": "Lakers",
        "MIA": "Heat",
        "MIL": "Bucks",
        "MIN": "Timberwolves",
        "BKN": "Nets",
        "NYK": "Knicks",
        "ORL": "Magic",
        "IND": "Pacers",
        "PHI": "76ers",
        "PHX": "Suns",
        "POR": "Trail Blazers",
        "SAC": "Kings",
        "SAS": "Spurs",
        "OKC": "Thunder",
        "TOR": "Raptors",
        "UTA": "Jazz",
        "MEM": "Grizzlies",
        "WAS": "Wizards",
        "DET": "Pistons",
        "CHA": "Hornets",
    }

    weights = {
        "Out": 1.0,
        "Doubtful": 0.8,
        "Questionable": 0.5,
        "Probable": 0.1,
    }

    injuries_df = injuries_df.copy()

    injuries_df["Team"] = injuries_df["Team"].map(team_map)

    injuries_df["weight"] = injuries_df["Status"].map(weights).fillna(0)

    return injuries_df.groupby("Team")["weight"].sum().to_dict()


# --- Main Prediction Logic ---
def predict_today() -> None:
    """Predict today's NBA game winners and print results."""
    df = build_feature_table(DATA_PATH)
    features_win = [
        "elo_diff",
        "points_form_diff",
        "reb_form_diff",
        "ast_form_diff",
        "rest_diff",
        "schedule_diff",
        "expected_pace",
        "momentum_diff",
        "home_advantage_diff",
        "off_form_diff",
        "def_form_diff",
        "official_net_rating_diff",
    ]
    missing = [f for f in features_win if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required winner features: {missing}")
    today_games = fetch_today_games()
    odds_df = fetch_vegas_odds()
    injuries = fetch_injuries()
    print(injuries.head())
    injury_scores = compute_team_injury_score(injuries)
    print("\nToday's Games Predictions\n")
    for _, game in today_games.iterrows():
        home_id = game["HOME_TEAM_ID"]
        away_id = game["VISITOR_TEAM_ID"]
        home_name = team_names.get(home_id, str(home_id))
        away_name = team_names.get(away_id, str(away_id))
        vegas_home = vegas_team_names.get(home_name, home_name)
        vegas_away = vegas_team_names.get(away_name, away_name)
        home_rows = df[
            (df["TEAM_ID_home"] == home_id) | (df["TEAM_ID_away"] == home_id)
        ].sort_values("GAME_DATE_home")
        away_rows = df[
            (df["TEAM_ID_home"] == away_id) | (df["TEAM_ID_away"] == away_id)
        ].sort_values("GAME_DATE_home")
        if home_rows.empty or away_rows.empty:
            print(f"{home_name} vs {away_name} | Not enough feature history")
            continue
        home_team = home_rows.iloc[-1]
        away_team = away_rows.iloc[-1]
        row = home_team.copy()
        row["elo_diff"] = home_team["home_elo"] - away_team["away_elo"]
        row["off_form_diff"] = (
            home_team["home_off_last10"] - away_team["away_off_last10"]
        )
        row["def_form_diff"] = (
            home_team["home_def_last10"] - away_team["away_def_last10"]
        )
        row["points_form_diff"] = (
            home_team["home_pts_last5"] - away_team["away_pts_last5"]
        )
        row["reb_form_diff"] = home_team["home_reb_last5"] - away_team["away_reb_last5"]
        row["ast_form_diff"] = home_team["home_ast_last5"] - away_team["away_ast_last5"]
        row["official_net_rating_diff"] = (
            home_team["NET_RATING_home"] - away_team["NET_RATING_away"]
            if "NET_RATING_home" in home_team and "NET_RATING_away" in away_team
            else 0
        )
        home_injury = injury_scores.get(home_name, 0)
        away_injury = injury_scores.get(away_name, 0)
        X_game = pd.DataFrame([row[features_win]], columns=features_win)
        model_prob = model.predict_proba(X_game)[0][1]
        winner_id = home_id if model_prob > 0.5 else away_id
        winner_name = team_names.get(winner_id, str(winner_id))
        odds_match = pd.DataFrame()
        if not odds_df.empty:
            odds_match = odds_df[
                odds_df["home_team_name"].str.contains(vegas_home, case=False, na=False)
                & odds_df["away_team_name"].str.contains(
                    vegas_away, case=False, na=False
                )
            ]
        if odds_match.empty:
            print(
                f"{home_name} vs {away_name} | "
                f"Model: {model_prob:.2f} | "
                f"Prediction: {winner_name} | "
                f"Home injuries: {home_injury:.1f} | Away injuries: {away_injury:.1f} | "
                f"Vegas: not found"
            )
            continue
        odds_row = odds_match.iloc[0]
        vegas_prob = odds_row["vegas_home_prob"]
        edge = model_prob - vegas_prob
        if edge >= 0.05:
            signal = "BET HOME"
        elif edge <= -0.05:
            signal = "BET AWAY"
        else:
            signal = "NO BET"
        spread = odds_row["home_spread"]
        spread_str = f"{spread:+}" if pd.notna(spread) else "N/A"
        print(
            f"{home_name} vs {away_name} | "
            f"Model: {model_prob:.2f} | "
            f"Vegas: {vegas_prob:.2f} | "
            f"Spread: {spread_str} | "
            f"Edge: {edge:+.2f} | "
            f"Prediction: {winner_name} | "
            f"Home injuries: {home_injury:.1f} | Away injuries: {away_injury:.1f} | "
            f"{signal}"
        )


if __name__ == "__main__":
    predict_today()
