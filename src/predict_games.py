import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import scoreboardv2
from train_model import build_feature_table, get_feature_list

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
    games_df = games_df.drop_duplicates(subset=["GAME_ID"])
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

    cache_file = os.path.join(BASE_DIR, "data", "vegas_cache.csv")

    # ✅ Use cache if exists
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        print("\nLoaded odds from cache")
        return df

    # ❌ No API key
    if not ODDS_API_KEY:
        print("\n❌ ODDS API KEY MISSING")
        return pd.DataFrame()

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",  # 🔥 simplify (important)
        "oddsFormat": "american",
    }

    response = requests.get(url, params=params, timeout=20)

    print("\nAPI STATUS:", response.status_code)

    response.raise_for_status()
    data = response.json()

    print("\nRAW ODDS SAMPLE:")
    print(data[:1])  # 🔥 debug

    rows = []

    for game in data:
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        home_ml = None
        away_ml = None

        for book in bookmakers:
            markets = {m["key"]: m for m in book.get("markets", [])}

            if "h2h" not in markets:
                continue

            for outcome in markets["h2h"].get("outcomes", []):
                if outcome["name"] == home_team:
                    home_ml = outcome.get("price")
                elif outcome["name"] == away_team:
                    away_ml = outcome.get("price")

            break

        if home_ml is None or away_ml is None:
            continue

        vegas_home_prob, vegas_away_prob = normalize_prob(home_ml, away_ml)

        rows.append({
            "home_team_name": home_team,
            "away_team_name": away_team,
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "vegas_home_prob": vegas_home_prob,
            "vegas_away_prob": vegas_away_prob,
        })

    odds_df = pd.DataFrame(rows)

    # ✅ SAVE CACHE
    odds_df.to_csv(cache_file, index=False)

    print("\nODDS DEBUG:")
    print("Empty:", odds_df.empty)
    print("Rows:", len(odds_df))
    print(odds_df.head())

    return odds_df


# get injuries for today's games
def fetch_injuries() -> pd.DataFrame:
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df.columns = ["Player", "Team", "Status", "Injury"]
        return df
    except Exception:
        return pd.DataFrame()
        
def team_injury_score(injuries_df: pd.DataFrame) -> dict:
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

def get_team_snapshot(df: pd.DataFrame, team_id: int) -> dict | None:
    team_games = df[
        (df["TEAM_ID_home"] == team_id) | (df["TEAM_ID_away"] == team_id)
    ].sort_values("GAME_DATE_home")

    if team_games.empty:
        return None

    last = team_games.iloc[-1]
    is_home = last["TEAM_ID_home"] == team_id

    if is_home:
        return {
            "team_id": team_id,
            "last_game_date": pd.to_datetime(last["GAME_DATE_home"]),
            "net_rating": last["NET_RATING_home"],
            "off_rating": last["OFF_RATING_home"],
            "def_rating": last["DEF_RATING_home"],
            "pace": last["PACE_home"],
            "pts_last5": last["home_pts_last5"],
            "reb_last5": last["home_reb_last5"],
            "off_last10": last["home_off_last10"],
            "def_last10": last["home_def_last10"],
            "efg_last10": last["home_efg_last10"],
            "tov_last10": last["home_tov_last10"],
            "reb_last10": last["home_reb_last10"],
            "games_last7": last["home_games_last7"],
            "win_last5": last["home_win_last5"],
        }
    else:
        return {
            "team_id": team_id,
            "last_game_date": pd.to_datetime(last["GAME_DATE_home"]),
            "net_rating": last["NET_RATING_away"],
            "off_rating": last["OFF_RATING_away"],
            "def_rating": last["DEF_RATING_away"],
            "pace": last["PACE_away"],
            "pts_last5": last["away_pts_last5"],
            "reb_last5": last["away_reb_last5"],
            "off_last10": last["away_off_last10"],
            "def_last10": last["away_def_last10"],
            "efg_last10": last["away_efg_last10"],
            "tov_last10": last["away_tov_last10"],
            "reb_last10": last["away_reb_last10"],
            "games_last7": last["away_games_last7"],
            "win_last5": last["away_win_last5"],
        }


def build_live_feature_row(hist_df: pd.DataFrame, home_id: int, away_id: int) -> pd.DataFrame | None:
    home = get_team_snapshot(hist_df, home_id)
    away = get_team_snapshot(hist_df, away_id)

    if home is None or away is None:
        return None

    today = pd.Timestamp.today().normalize()
    home_rest_days = max((today - home["last_game_date"].normalize()).days, 0)
    away_rest_days = max((today - away["last_game_date"].normalize()).days, 0)

    row = {
        "power_diff": home["net_rating"] - away["net_rating"],
        "matchup_diff": (home["off_rating"] - away["def_rating"]) - (away["off_rating"] - home["def_rating"]),
        "efg_diff_roll": home["efg_last10"] - away["efg_last10"],
        "tov_diff_roll": away["tov_last10"] - home["tov_last10"],
        "reb_diff_roll": home["reb_last10"] - away["reb_last10"],
        "pace_diff": home["pace"] - away["pace"],
        "rest_diff": home_rest_days - away_rest_days,
        "schedule_diff": home["games_last7"] - away["games_last7"],
        "points_form_diff": home["pts_last5"] - away["pts_last5"],
        "reb_form_diff": home["reb_last5"] - away["reb_last5"],
        "off_form_diff": home["off_last10"] - away["off_last10"],
        "def_form_diff": home["def_last10"] - away["def_last10"],
        "momentum_diff": home["win_last5"] - away["win_last5"],
        "home_team": 1,
    }

    return pd.DataFrame([row])


# predict today's games and print results
def predict_today() -> None:
    hist_df = build_feature_table(DATA_PATH)
    FEATURES = get_feature_list()

    # Validate features exist
    missing = [f for f in FEATURES if f not in hist_df.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    # Fetch live data
    today_games = fetch_today_games()
    odds_df = fetch_vegas_odds()
    injuries = fetch_injuries()
    injury_scores = team_injury_score(injuries)

    print("\nODDS DF DEBUG:")
    print("Empty:", odds_df.empty)
    print(odds_df.head())

    print("\nINJURY DEBUG:")
    print("Empty:", injuries.empty)
    print(injuries.head())

    print("\nToday's Games Predictions\n")

    # Helper for matching names
    def normalize_name(name: str) -> str:
        return name.lower().replace(" ", "").replace(".", "")

    # Preprocess odds names once
    if not odds_df.empty:
        odds_df["home_norm"] = odds_df["home_team_name"].apply(normalize_name)
        odds_df["away_norm"] = odds_df["away_team_name"].apply(normalize_name)

    for _, game in today_games.iterrows():
        home_id = game["HOME_TEAM_ID"]
        away_id = game["VISITOR_TEAM_ID"]

        home_name = team_names.get(home_id, str(home_id))
        away_name = team_names.get(away_id, str(away_id))

        vegas_home = vegas_team_names.get(home_name, home_name)
        vegas_away = vegas_team_names.get(away_name, away_name)

        # Build features
        X_game = build_live_feature_row(hist_df, home_id, away_id)

        if X_game is None:
            print(f"{home_name} vs {away_name} | Not enough feature history")
            continue

        X_game = X_game[FEATURES]
        model_prob = model.predict_proba(X_game)[0][1]

        winner_id = home_id if model_prob > 0.5 else away_id
        winner_name = team_names.get(winner_id, str(winner_id))

        home_injury = injury_scores.get(home_name, 0)
        away_injury = injury_scores.get(away_name, 0)

        # ✅ Improved odds matching
        odds_match = pd.DataFrame()

        if not odds_df.empty:
            vegas_home_norm = normalize_name(vegas_home)
            vegas_away_norm = normalize_name(vegas_away)

            odds_match = odds_df[
                (odds_df["home_norm"] == vegas_home_norm) &
                (odds_df["away_norm"] == vegas_away_norm)
            ]

        # If no odds found
        if odds_match.empty:
            print(
                f"{home_name} vs {away_name} | "
                f"Model: {model_prob:.2f} | "
                f"Prediction: {winner_name} | "
                f"Injuries H:{home_injury:.1f} A:{away_injury:.1f} | "
                f"Vegas: not found"
            )
            continue

        odds_row = odds_match.iloc[0]
        vegas_prob = odds_row["vegas_home_prob"]
        edge = model_prob - vegas_prob

        # Signal logic
        if edge >= 0.05:
            signal = "BET HOME"
        elif edge <= -0.05:
            signal = "BET AWAY"
        else:
            signal = "NO BET"

        print(
            f"{home_name} vs {away_name} | "
            f"Model: {model_prob:.2f} | "
            f"Vegas: {vegas_prob:.2f} | "
            f"Edge: {edge:+.2f} | "
            f"Prediction: {winner_name} | "
            f"Injuries H:{home_injury:.1f} A:{away_injury:.1f} | "
            f"{signal}"
        )

    print("\nInjury dataframe empty:", injuries.empty)


if __name__ == "__main__":
    predict_today()
