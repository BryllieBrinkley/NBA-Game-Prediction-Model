from nba_api.stats.endpoints import leaguedashteamstats

stats = leaguedashteamstats.LeagueDashTeamStats(
    season="2024-25",
    measure_type_detailed_defense="Advanced"
)

df = stats.get_data_frames()[0]

print(df.head())