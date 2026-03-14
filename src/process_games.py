import pandas as pd

df = pd.read_csv("data/nba_games_multi_season.csv")

df["home"] = df["MATCHUP"].str.contains("vs.")

home_games = df[df["home"] == True]
away_games = df[df["home"] == False]

games = home_games.merge(away_games, on="GAME_ID", suffixes=("_home", "_away"))

games["total_points"] = games["PTS_home"] + games["PTS_away"]
games["home_win"] = (games["PTS_home"] > games["PTS_away"]).astype(int)

games.to_csv("data/model_dataset.csv", index=False)

print(games.columns)
