import pandas as pd
import numpy as np


length = 120  # length of a football field
width = 160/3  # width of a football field
play_identifier = ["gameId", "playId"]  # combining gameId and playId to identify unique play
player_identifier = play_identifier + ["nflId"]     # combining gameId, playId, and nflId to identify unique player on a play
frame_identifier = play_identifier + ["frameId"]    # combining gameId, playId, and frameId to identify unique frame on a play
frame_and_player_identifier = player_identifier + ["frameId"]   # combine gameId, playId, frameId, and playerId to identify unique player and frame on a play
weeks = range(1, 9)

plays = pd.read_csv("plays.csv")  # load plays.csv

# Load pffScoutingData.csv and assign pressure variable as 1 if either a sack, hit, or hurry has occurred for a player
# on a play.
pff = (
    pd.read_csv("pffScoutingData.csv")
    .assign(
        pressure=lambda df_: (
            df_
            [["pff_hit", "pff_hurry", "pff_sack"]]
            .max(axis=1)
            .fillna(0)
            .astype(bool)
        )
    )
    .set_index(player_identifier)
)


def play_direction(df):
    """
    Function to adjust plays going from right to left so that all available plays run from left to right.
    """

    flip_field = df.assign(
        x=length - df.x,
        y=width - df.y,
        o=df.o + 180,
    )
    return df.where(df.playDirection == "right", flip_field)


def adjust_orientation(df):
    """
    Function to correct orientation for the ones where its value went above 360 degrees due to the previous computation.
    """

    df.o = df.o.apply(lambda x: x - 360 if x >= 360 else x)
    return df


def clean_df(df):
    """
    Function to reverse columns to be true ints again.
    """

    df.nflId = df.nflId.apply(lambda x: int(x))
    df.jerseyNumber = df.jerseyNumber.apply(lambda x: int(x))
    return df


def preprocess_week(week):
    """
    Function to preprocess all weeks and join with PFF dataframe based on the previously established player identifier.
    """

    return (
        pd.read_csv(f"week{week}.csv")
        .join(
            pff[["pff_role", "pff_positionLinedUp", "pressure"]],
            on=player_identifier
        )
        .pipe(play_direction)
        .pipe(adjust_orientation)
        .assign(week=week)
    )


all_weeks = pd.concat(
    [preprocess_week(week) for week in weeks],
    ignore_index=True,
)

# Create cleaned dataframe only containing the players important to pass rush, i.e. the Quarterback, the pass rushers,
# and the pass blockers.
pass_rush = all_weeks[all_weeks['pff_role'].isin(['Pass', 'Pass Rush', 'Pass Block'])].reset_index(drop=True)
pass_rush = clean_df(pass_rush)


# Feature computation:


def xyo_offset(df):
    """
    Function to compute the x, y, and orientation offsets of a player to their previous frame. In the event of the first
    frame, it sets the offsets to 0.
    """

    offset_columns = ["x_offset", "y_offset", "o_offset"]
    offsets = (df[["x", "y", "o"]].diff().set_axis(offset_columns, axis=1))
    df = df.assign(**offsets)
    df.loc[df.frameId == 1, offset_columns] = 0
    return df


def compute_player_offsets(df, columns1, columns2):
    """
    Function to compute the x, y, and orientation offsets columns of a player to another player.
    """

    offsets = df[columns1].values - df[columns2].values
    dists = np.linalg.norm(offsets, axis=1)
    dists = np.expand_dims(dists, 1)
    return np.concatenate([offsets, dists], axis=1)


pass_rush_offsets = xyo_offset(pass_rush)   # Add xyo offsets of previous frame to all players


def find_qb_location(df):
    """
    Function to establish the offsets of each player to the Quarterback.
    """

    qbs = (
        df
        .query("pff_role == 'Pass'")
        .set_index(frame_identifier)
        [["x", "y", "o", "x_offset", "y_offset", "s"]]
    )

    df = df.join(qbs, on=frame_identifier, rsuffix="_qb")

    qb_columns = [
        "x_offset_to_qb",
        "y_offset_to_qb",
        "distance_to_qb",
    ]

    df[qb_columns] = compute_player_offsets(
        df,
        ["x_qb", "y_qb"],
        ["x", "y"],
    )
    return df


def join_defense_offense(df):
    """
    Function to add all pass blockers to the pass rushers.
    """

    defense = (
        df
        .query("pff_role == 'Pass Rush'")
        .drop(columns="pff_role")
    )

    offense = (
        df
        .query("pff_role == 'Pass Block'")
        .set_index(frame_identifier)
        .drop(columns=[
            "pff_role",
            "x_qb", "y_qb", "o_qb", "x_offset_qb", "y_offset_qb", "week",
        ])
    )

    return defense.join(offense, on=frame_identifier, rsuffix="_offense")


def find_offense_offsets(df):
    """
    Function to compute the offsets of each pass blocker to each pass rusher.
    """

    offense_columns = [
        "x_offset_to_offense",
        "y_offset_to_offense",
        "distance_to_offense",
    ]

    df[offense_columns] = compute_player_offsets(
        df,
        ["x_offense", "y_offense"],
        ["x", "y"],
    )
    return df


def find_closest_o_to_d(df):
    """
    Function to find the closest pass blocker for each pass rusher on a frame.
    """

    closest_o = df.groupby(frame_and_player_identifier)["distance_to_offense"].min()
    closest_o = df.merge(closest_o, on=frame_and_player_identifier, suffixes=("", "_min"))
    closest_o = closest_o[closest_o.distance_to_offense == closest_o.distance_to_offense_min].drop("distance_to_offense_min", axis=1).reset_index(drop=True)
    closest_o = closest_o.drop_duplicates(subset=frame_and_player_identifier, keep="first").reset_index(drop=True)
    return closest_o


def find_angle_d_qb_o(df):
    """
    Function to compute the angle between Quarterback, pass rusher, and pass blocker, with the Quarterback at the vertex for every frame.
    """

    d_x = df.x - df.x_qb
    d_y = df.y - df.y_qb
    o_x = df.x_offense - df.x_qb
    o_y = df.y_offense - df.y_qb
    angle = (np.arctan2(d_y, d_x) - np.arctan2(o_y, o_x))
    return df.assign(qb_angle=angle).fillna({"qb_angle": 0})


def set_offsets_defense(df):
    """
    Function to inverse the offset values for better understanding.
    """

    df["x_offset_to_offense"] = df["x_offset_to_offense"].apply(lambda x: -x)
    df["y_offset_to_offense"] = df["y_offset_to_offense"].apply(lambda x: -x)
    df["x_offset_to_qb"] = df["x_offset_to_qb"].apply(lambda x: -x)
    df["y_offset_to_qb"] = df["y_offset_to_qb"].apply(lambda x: -x)
    return df


# Create dataframe containing only the pass rushers with all necessary features for each frame
defense_all_features = (
    pass_rush_offsets
    .pipe(find_qb_location)
    .query("pff_role != 'Pass'")
    .pipe(join_defense_offense)
    .pipe(find_offense_offsets)
    .pipe(find_closest_o_to_d)
    .pipe(find_angle_d_qb_o)
    .drop(columns=['x_offset_to_qb_offense', 'y_offset_to_qb_offense', 'distance_to_qb_offense'], axis=1)
    .pipe(set_offsets_defense)
)

# Turn pressure into an int variable being 0 for no pressure having occurred and 1 for pressure having occurred
defense_all_features['pressure'] = defense_all_features['pressure'].astype(int)

defense_all_features.to_csv('defense_all_features.csv')     # CSV file with only pass rushers and their KNN-relevant features
