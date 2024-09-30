import pandas as pd

pressure_defense = pd.read_csv("pressure_defense.csv")
pressure_defense_pmp = pd.read_csv("pressure_defense_pmp.csv")
players = pd.read_csv("players.csv")

# Merging the 'predicted_motion_pressure' column into the 'pressure_defense' dataframe
pressure_defense = pd.merge(
    pressure_defense,
    pressure_defense_pmp[['nflId', 'gameId', 'playId', 'frameId', 'predicted_motion_pressure']],
    on=['nflId', 'gameId', 'playId', 'frameId'],
    how='left'
)

# Grouping by 'nflId', 'gameId', and 'playId' to calculate the average of 'predicted_pressure' and 'predicted_motion_pressure'
pmp_per_play = pressure_defense.groupby(['nflId', 'gameId', 'playId']).agg(
    avg_pressure_probability=('pressure_probability', 'mean'),
    avg_predicted_motion_pressure=('predicted_motion_pressure', 'mean')
).reset_index()

# Creating the new column for the difference between the averages
pmp_per_play['pressure_delta'] = pmp_per_play['avg_pressure_probability'] - pmp_per_play['avg_predicted_motion_pressure']

# Adding names to dataframe
pmp_per_play = pd.merge(
    pmp_per_play,
    players[['nflId', 'displayName']],
    on='nflId',
    how='left'
)

# Average the deltas for each player over all their plays
pmp_per_player = pmp_per_play.groupby('nflId').agg(
    pressure_delta=('pressure_delta', 'mean')
).reset_index()

# Adding name to dataframe
pmp_per_player = pd.merge(
    pmp_per_player,
    players[['nflId', 'displayName']],
    on='nflId',
    how='left'
)

# Sorting both PMP dataframes
pmp_per_play = pmp_per_play.sort_values(by='pressure_delta', ascending=False).reset_index(drop=True)
pmp_per_player = pmp_per_player.sort_values(by='pressure_delta', ascending=False).reset_index(drop=True)

pmp_per_play.to_csv('pmp_per_play.csv')
pmp_per_player.to_csv('pmp_per_player.csv')

# Creating a dataframe with only the pass rusher who rushed at least 5 times
# Counting the number of entries for each nflId in the pmp_per_play
nflId_counts = pmp_per_play['nflId'].value_counts().reset_index()
nflId_counts.columns = ['nflId', 'entry_count']

# Merging the count data with the pmp_per_player to filter only those with at least 5 entries
pmp_per_player_min_5 = pd.merge(pmp_per_player, nflId_counts, on='nflId', how='left')

# Filtering to keep only players with at least 5 entries
pmp_per_player_min_5 = pmp_per_player_min_5[pmp_per_player_min_5['entry_count'] >= 5].drop(columns=['entry_count'])

# Sorting the final filtered dataframe by 'pressure_delta' in descending order
pmp_per_player_min_5 = pmp_per_player_min_5.sort_values(by='pressure_delta', ascending=False).reset_index(drop=True)

pmp_per_player_min_5.to_csv('pmp_per_player_min_5.csv')