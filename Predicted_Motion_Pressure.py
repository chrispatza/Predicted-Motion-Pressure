import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

play_identifier = ["gameId", "playId"]  # combining gameId and playId to identify unique play
player_identifier = play_identifier + ["nflId"]     # combining gameId, playId, and nflId to identify unique player on a play
frame_identifier = play_identifier + ["frameId"]    # combining gameId, playId, and frameId to identify unique frame on a play
frame_and_player_identifier = player_identifier + ["frameId"]   # combine gameId, playId, frameId, and playerId to identify unique player and frame on a play

features = ["x_offset", "y_offset", "s", "o", "o_qb", "s_qb", "x_offset_to_qb", "y_offset_to_qb", "o_offense", "s_offense", "x_offset_to_offense", "y_offset_to_offense", "qb_angle"]

pressure_defense = pd.read_csv("pressure_defense.csv")  # load pressure_defense.csv
defense_all_features = pd.read_csv("defense_all_features.csv")  # load defense_all_features.csv
all_weeks = pd.read_csv("all_weeks.csv")  # load all_weeks.csv
plays = pd.read_csv("plays.csv")  # load plays.csv
plays['frames_on_play'] = 0
# Add how many frames there are per play to plays dataframe
for i in range(len(plays)):
    play = all_weeks[(all_weeks.gameId == plays.gameId.iloc[i]) & (all_weeks.playId == plays.playId.iloc[i])]
    max_frame = play.frameId.max()
    plays.loc[i, 'frames_on_play'] = max_frame


def add_xyo_offset_next_frame(df):
    """
    Function for creating new columns for the x - and y-offsets and orientation in the
    next frame.
    Copy the values from the same frame and shift all up by one.
    """

    df["x_offset_next_frame"] = df["x_offset"]
    df["y_offset_next_frame"] = df["y_offset"]
    df["o_next_frame"] = df["o"]
    df['x_offset_next_frame'] = df['x_offset_next_frame'].shift(-1)
    df['y_offset_next_frame'] = df['y_offset_next_frame'].shift(-1)
    df['o_next_frame'] = df['o_next_frame'].shift(-1)
    df = df[:-1]
    return df


def drop_last_frame(df):
    """
    Function to drop the last frame of each play as this one cannot be used, as there are
    no offsets/orientation in the frame following that one.
    Due to the previous shift up the corresponding values in these rows do not
    match anyway, therefore this gets rid of those frames.
    """

    df.drop(df[df["frameId"] == df["frames_on_play"]].index, inplace=True)
    df = df.reset_index(drop=True)
    return df


# Add amount of frames to pass rusher dataframe and shift all outcome variables up to prepare dataframe for KNN
defense_all_features_motion = pd.merge(defense_all_features, plays, on=['gameId', 'playId'], how='left')
defense_all_features_motion = (
    defense_all_features_motion
    .pipe(add_xyo_offset_next_frame)
    .pipe(drop_last_frame))


def motion_knn_week_8(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame using KNN.
    """

    # Split dataframe into training and testing dataframe. Here, training will be weeks 1-7, while testing will be week 8.
    df_knn_pmp_train_8 = df.loc[df["week"] != 8].reset_index(drop=True)
    df_knn_pmp_test_8 = df.loc[df["week"] == 8].reset_index(drop=True)
    X_train_knn_pmp_8 = df_knn_pmp_train_8[features]
    y_train_knn_pmp_x_offset_8 = df_knn_pmp_train_8['x_offset_next_frame']
    y_train_knn_pmp_y_offset_8 = df_knn_pmp_train_8['y_offset_next_frame']
    y_train_knn_pmp_o_8 = df_knn_pmp_train_8['o_next_frame']
    X_test_knn_pmp_8 = df_knn_pmp_test_8[features]

    # Set the StandardScaler, fit and transform dataframe.
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_8)
    scaled_features_train_pmp_8 = scaler.transform(X_train_knn_pmp_8)
    X_train_knn_pmp_8[features] = scaled_features_train_pmp_8
    scaled_features_test_pmp_8 = scaler.transform(X_test_knn_pmp_8)
    X_test_knn_pmp_8[features] = scaled_features_test_pmp_8

    # Train and fit three separate KNNs for week 8 with n=50 and the distance to the neighbors having been given weight.
    # One KNN predicts the x-offset in the next frame, one the y-offset, and one the orientation.
    knn_pmp_8 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_8 = knn_pmp_8.fit(X_train_knn_pmp_8, y_train_knn_pmp_x_offset_8)
    predicted_x_offset_pmp_8 = knn_pmp_x_offset_8.predict(X_test_knn_pmp_8)
    df_knn_pmp_test_8['x_offset_predicted'] = predicted_x_offset_pmp_8
    knn_pmp_8 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_8 = knn_pmp_8.fit(X_train_knn_pmp_8, y_train_knn_pmp_y_offset_8)
    predicted_y_offset_pmp_8 = knn_pmp_y_offset_8.predict(X_test_knn_pmp_8)
    df_knn_pmp_test_8['y_offset_predicted'] = predicted_y_offset_pmp_8
    knn_pmp_8 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_8 = knn_pmp_8.fit(X_train_knn_pmp_8, y_train_knn_pmp_o_8)
    predicted_o_pmp_8 = knn_pmp_o_8.predict(X_test_knn_pmp_8)
    df_knn_pmp_test_8['o_predicted'] = predicted_o_pmp_8

    # Compute the x - and y values for the next frame by adding the predicted offset to the current x - and y-values.
    # Compute the speed in the next frame by calculating the predicted distance traveled and divide by 0.1 seconds, as
    # v = delta d / delta t.
    df_knn_pmp_test_8['x_predicted'] = df_knn_pmp_test_8['x'] + df_knn_pmp_test_8['x_offset_predicted']
    df_knn_pmp_test_8['y_predicted'] = df_knn_pmp_test_8['y'] + df_knn_pmp_test_8['y_offset_predicted']
    df_knn_pmp_test_8['s_predicted'] = (df_knn_pmp_test_8['x_offset_predicted'] ** 2 + df_knn_pmp_test_8[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_8


# Repeat for all other 7 weeks:
def motion_knn_week_7(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame for week 7 using KNN.
    """

    df_knn_pmp_train_7 = df.loc[df["week"] != 7].reset_index(drop=True)
    df_knn_pmp_test_7 = df.loc[df["week"] == 7].reset_index(drop=True)
    X_train_knn_pmp_7 = df_knn_pmp_train_7[features]
    y_train_knn_pmp_x_offset_7 = df_knn_pmp_train_7['x_offset_next_frame']
    y_train_knn_pmp_y_offset_7 = df_knn_pmp_train_7['y_offset_next_frame']
    y_train_knn_pmp_o_7 = df_knn_pmp_train_7['o_next_frame']
    X_test_knn_pmp_7 = df_knn_pmp_test_7[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_7)
    scaled_features_train_pmp_7 = scaler.transform(X_train_knn_pmp_7)
    X_train_knn_pmp_7[features] = scaled_features_train_pmp_7
    scaled_features_test_pmp_7 = scaler.transform(X_test_knn_pmp_7)
    X_test_knn_pmp_7[features] = scaled_features_test_pmp_7
    knn_pmp_7 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_7 = knn_pmp_7.fit(X_train_knn_pmp_7, y_train_knn_pmp_x_offset_7)
    predicted_x_offset_pmp_7 = knn_pmp_x_offset_7.predict(X_test_knn_pmp_7)
    df_knn_pmp_test_7['x_offset_predicted'] = predicted_x_offset_pmp_7
    knn_pmp_7 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_7 = knn_pmp_7.fit(X_train_knn_pmp_7, y_train_knn_pmp_y_offset_7)
    predicted_y_offset_pmp_7 = knn_pmp_y_offset_7.predict(X_test_knn_pmp_7)
    df_knn_pmp_test_7['y_offset_predicted'] = predicted_y_offset_pmp_7
    knn_pmp_7 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_7 = knn_pmp_7.fit(X_train_knn_pmp_7, y_train_knn_pmp_o_7)
    predicted_o_pmp_7 = knn_pmp_o_7.predict(X_test_knn_pmp_7)
    df_knn_pmp_test_7['o_predicted'] = predicted_o_pmp_7
    df_knn_pmp_test_7['x_predicted'] = df_knn_pmp_test_7['x'] + df_knn_pmp_test_7['x_offset_predicted']
    df_knn_pmp_test_7['y_predicted'] = df_knn_pmp_test_7['y'] + df_knn_pmp_test_7['y_offset_predicted']
    df_knn_pmp_test_7['s_predicted'] = (df_knn_pmp_test_7['x_offset_predicted'] ** 2 + df_knn_pmp_test_7[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_7


def motion_knn_week_6(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame for week 6 using KNN.
    """

    df_knn_pmp_train_6 = df.loc[df["week"] != 6].reset_index(drop=True)
    df_knn_pmp_test_6 = df.loc[df["week"] == 6].reset_index(drop=True)
    X_train_knn_pmp_6 = df_knn_pmp_train_6[features]
    y_train_knn_pmp_x_offset_6 = df_knn_pmp_train_6['x_offset_next_frame']
    y_train_knn_pmp_y_offset_6 = df_knn_pmp_train_6['y_offset_next_frame']
    y_train_knn_pmp_o_6 = df_knn_pmp_train_6['o_next_frame']
    X_test_knn_pmp_6 = df_knn_pmp_test_6[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_6)
    scaled_features_train_pmp_6 = scaler.transform(X_train_knn_pmp_6)
    X_train_knn_pmp_6[features] = scaled_features_train_pmp_6
    scaled_features_test_pmp_6 = scaler.transform(X_test_knn_pmp_6)
    X_test_knn_pmp_6[features] = scaled_features_test_pmp_6
    knn_pmp_6 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_6 = knn_pmp_6.fit(X_train_knn_pmp_6, y_train_knn_pmp_x_offset_6)
    predicted_x_offset_pmp_6 = knn_pmp_x_offset_6.predict(X_test_knn_pmp_6)
    df_knn_pmp_test_6['x_offset_predicted'] = predicted_x_offset_pmp_6
    knn_pmp_6 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_6 = knn_pmp_6.fit(X_train_knn_pmp_6, y_train_knn_pmp_y_offset_6)
    predicted_y_offset_pmp_6 = knn_pmp_y_offset_6.predict(X_test_knn_pmp_6)
    df_knn_pmp_test_6['y_offset_predicted'] = predicted_y_offset_pmp_6
    knn_pmp_6 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_6 = knn_pmp_6.fit(X_train_knn_pmp_6, y_train_knn_pmp_o_6)
    predicted_o_pmp_6 = knn_pmp_o_6.predict(X_test_knn_pmp_6)
    df_knn_pmp_test_6['o_predicted'] = predicted_o_pmp_6
    df_knn_pmp_test_6['x_predicted'] = df_knn_pmp_test_6['x'] + df_knn_pmp_test_6['x_offset_predicted']
    df_knn_pmp_test_6['y_predicted'] = df_knn_pmp_test_6['y'] + df_knn_pmp_test_6['y_offset_predicted']
    df_knn_pmp_test_6['s_predicted'] = (df_knn_pmp_test_6['x_offset_predicted'] ** 2 + df_knn_pmp_test_6[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_6


def motion_knn_week_5(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame for week 5 using KNN.
    """

    df_knn_pmp_train_5 = df.loc[df["week"] != 5].reset_index(drop=True)
    df_knn_pmp_test_5 = df.loc[df["week"] == 5].reset_index(drop=True)
    X_train_knn_pmp_5 = df_knn_pmp_train_5[features]
    y_train_knn_pmp_x_offset_5 = df_knn_pmp_train_5['x_offset_next_frame']
    y_train_knn_pmp_y_offset_5 = df_knn_pmp_train_5['y_offset_next_frame']
    y_train_knn_pmp_o_5 = df_knn_pmp_train_5['o_next_frame']
    X_test_knn_pmp_5 = df_knn_pmp_test_5[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_5)
    scaled_features_train_pmp_5 = scaler.transform(X_train_knn_pmp_5)
    X_train_knn_pmp_5[features] = scaled_features_train_pmp_5
    scaled_features_test_pmp_5 = scaler.transform(X_test_knn_pmp_5)
    X_test_knn_pmp_5[features] = scaled_features_test_pmp_5
    knn_pmp_5 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_5 = knn_pmp_5.fit(X_train_knn_pmp_5, y_train_knn_pmp_x_offset_5)
    predicted_x_offset_pmp_5 = knn_pmp_x_offset_5.predict(X_test_knn_pmp_5)
    df_knn_pmp_test_5['x_offset_predicted'] = predicted_x_offset_pmp_5
    knn_pmp_5 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_5 = knn_pmp_5.fit(X_train_knn_pmp_5, y_train_knn_pmp_y_offset_5)
    predicted_y_offset_pmp_5 = knn_pmp_y_offset_5.predict(X_test_knn_pmp_5)
    df_knn_pmp_test_5['y_offset_predicted'] = predicted_y_offset_pmp_5
    knn_pmp_5 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_5 = knn_pmp_5.fit(X_train_knn_pmp_5, y_train_knn_pmp_o_5)
    predicted_o_pmp_5 = knn_pmp_o_5.predict(X_test_knn_pmp_5)
    df_knn_pmp_test_5['o_predicted'] = predicted_o_pmp_5
    df_knn_pmp_test_5['x_predicted'] = df_knn_pmp_test_5['x'] + df_knn_pmp_test_5['x_offset_predicted']
    df_knn_pmp_test_5['y_predicted'] = df_knn_pmp_test_5['y'] + df_knn_pmp_test_5['y_offset_predicted']
    df_knn_pmp_test_5['s_predicted'] = (df_knn_pmp_test_5['x_offset_predicted'] ** 2 + df_knn_pmp_test_5[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_5


def motion_knn_week_4(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame for week 4 using KNN.
    """

    df_knn_pmp_train_4 = df.loc[df["week"] != 4].reset_index(drop=True)
    df_knn_pmp_test_4 = df.loc[df["week"] == 4].reset_index(drop=True)
    X_train_knn_pmp_4 = df_knn_pmp_train_4[features]
    y_train_knn_pmp_x_offset_4 = df_knn_pmp_train_4['x_offset_next_frame']
    y_train_knn_pmp_y_offset_4 = df_knn_pmp_train_4['y_offset_next_frame']
    y_train_knn_pmp_o_4 = df_knn_pmp_train_4['o_next_frame']
    X_test_knn_pmp_4 = df_knn_pmp_test_4[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_4)
    scaled_features_train_pmp_4 = scaler.transform(X_train_knn_pmp_4)
    X_train_knn_pmp_4[features] = scaled_features_train_pmp_4
    scaled_features_test_pmp_4 = scaler.transform(X_test_knn_pmp_4)
    X_test_knn_pmp_4[features] = scaled_features_test_pmp_4
    knn_pmp_4 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_4 = knn_pmp_4.fit(X_train_knn_pmp_4, y_train_knn_pmp_x_offset_4)
    predicted_x_offset_pmp_4 = knn_pmp_x_offset_4.predict(X_test_knn_pmp_4)
    df_knn_pmp_test_4['x_offset_predicted'] = predicted_x_offset_pmp_4
    knn_pmp_4 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_4 = knn_pmp_4.fit(X_train_knn_pmp_4, y_train_knn_pmp_y_offset_4)
    predicted_y_offset_pmp_4 = knn_pmp_y_offset_4.predict(X_test_knn_pmp_4)
    df_knn_pmp_test_4['y_offset_predicted'] = predicted_y_offset_pmp_4
    knn_pmp_4 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_4 = knn_pmp_4.fit(X_train_knn_pmp_4, y_train_knn_pmp_o_4)
    predicted_o_pmp_4 = knn_pmp_o_4.predict(X_test_knn_pmp_4)
    df_knn_pmp_test_4['o_predicted'] = predicted_o_pmp_4
    df_knn_pmp_test_4['x_predicted'] = df_knn_pmp_test_4['x'] + df_knn_pmp_test_4['x_offset_predicted']
    df_knn_pmp_test_4['y_predicted'] = df_knn_pmp_test_4['y'] + df_knn_pmp_test_4['y_offset_predicted']
    df_knn_pmp_test_4['s_predicted'] = (df_knn_pmp_test_4['x_offset_predicted'] ** 2 + df_knn_pmp_test_4[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_4


def motion_knn_week_3(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame for week 3 using KNN.
    """

    df_knn_pmp_train_3 = df.loc[df["week"] != 3].reset_index(drop=True)
    df_knn_pmp_test_3 = df.loc[df["week"] == 3].reset_index(drop=True)
    X_train_knn_pmp_3 = df_knn_pmp_train_3[features]
    y_train_knn_pmp_x_offset_3 = df_knn_pmp_train_3['x_offset_next_frame']
    y_train_knn_pmp_y_offset_3 = df_knn_pmp_train_3['y_offset_next_frame']
    y_train_knn_pmp_o_3 = df_knn_pmp_train_3['o_next_frame']
    X_test_knn_pmp_3 = df_knn_pmp_test_3[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_3)
    scaled_features_train_pmp_3 = scaler.transform(X_train_knn_pmp_3)
    X_train_knn_pmp_3[features] = scaled_features_train_pmp_3
    scaled_features_test_pmp_3 = scaler.transform(X_test_knn_pmp_3)
    X_test_knn_pmp_3[features] = scaled_features_test_pmp_3
    knn_pmp_3 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_3 = knn_pmp_3.fit(X_train_knn_pmp_3, y_train_knn_pmp_x_offset_3)
    predicted_x_offset_pmp_3 = knn_pmp_x_offset_3.predict(X_test_knn_pmp_3)
    df_knn_pmp_test_3['x_offset_predicted'] = predicted_x_offset_pmp_3
    knn_pmp_3 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_3 = knn_pmp_3.fit(X_train_knn_pmp_3, y_train_knn_pmp_y_offset_3)
    predicted_y_offset_pmp_3 = knn_pmp_y_offset_3.predict(X_test_knn_pmp_3)
    df_knn_pmp_test_3['y_offset_predicted'] = predicted_y_offset_pmp_3
    knn_pmp_3 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_3 = knn_pmp_3.fit(X_train_knn_pmp_3, y_train_knn_pmp_o_3)
    predicted_o_pmp_3 = knn_pmp_o_3.predict(X_test_knn_pmp_3)
    df_knn_pmp_test_3['o_predicted'] = predicted_o_pmp_3
    df_knn_pmp_test_3['x_predicted'] = df_knn_pmp_test_3['x'] + df_knn_pmp_test_3['x_offset_predicted']
    df_knn_pmp_test_3['y_predicted'] = df_knn_pmp_test_3['y'] + df_knn_pmp_test_3['y_offset_predicted']
    df_knn_pmp_test_3['s_predicted'] = (df_knn_pmp_test_3['x_offset_predicted'] ** 2 + df_knn_pmp_test_3[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_3


def motion_knn_week_2(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame for week 2 using KNN.
    """

    df_knn_pmp_train_2 = df.loc[df["week"] != 2].reset_index(drop=True)
    df_knn_pmp_test_2 = df.loc[df["week"] == 2].reset_index(drop=True)
    X_train_knn_pmp_2 = df_knn_pmp_train_2[features]
    y_train_knn_pmp_x_offset_2 = df_knn_pmp_train_2['x_offset_next_frame']
    y_train_knn_pmp_y_offset_2 = df_knn_pmp_train_2['y_offset_next_frame']
    y_train_knn_pmp_o_2 = df_knn_pmp_train_2['o_next_frame']
    X_test_knn_pmp_2 = df_knn_pmp_test_2[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_2)
    scaled_features_train_pmp_2 = scaler.transform(X_train_knn_pmp_2)
    X_train_knn_pmp_2[features] = scaled_features_train_pmp_2
    scaled_features_test_pmp_2 = scaler.transform(X_test_knn_pmp_2)
    X_test_knn_pmp_2[features] = scaled_features_test_pmp_2
    knn_pmp_2 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_2 = knn_pmp_2.fit(X_train_knn_pmp_2, y_train_knn_pmp_x_offset_2)
    predicted_x_offset_pmp_2 = knn_pmp_x_offset_2.predict(X_test_knn_pmp_2)
    df_knn_pmp_test_2['x_offset_predicted'] = predicted_x_offset_pmp_2
    knn_pmp_2 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_2 = knn_pmp_2.fit(X_train_knn_pmp_2, y_train_knn_pmp_y_offset_2)
    predicted_y_offset_pmp_2 = knn_pmp_y_offset_2.predict(X_test_knn_pmp_2)
    df_knn_pmp_test_2['y_offset_predicted'] = predicted_y_offset_pmp_2
    knn_pmp_2 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_2 = knn_pmp_2.fit(X_train_knn_pmp_2, y_train_knn_pmp_o_2)
    predicted_o_pmp_2 = knn_pmp_o_2.predict(X_test_knn_pmp_2)
    df_knn_pmp_test_2['o_predicted'] = predicted_o_pmp_2
    df_knn_pmp_test_2['x_predicted'] = df_knn_pmp_test_2['x'] + df_knn_pmp_test_2['x_offset_predicted']
    df_knn_pmp_test_2['y_predicted'] = df_knn_pmp_test_2['y'] + df_knn_pmp_test_2['y_offset_predicted']
    df_knn_pmp_test_2['s_predicted'] = (df_knn_pmp_test_2['x_offset_predicted'] ** 2 + df_knn_pmp_test_2[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_2


def motion_knn_week_1(df):
    """
    Function to obtain x - and y-offsets and orientation of pass rushers in the next frame for week 1 using KNN.
    """

    df_knn_pmp_train_1 = df.loc[df["week"] != 1].reset_index(drop=True)
    df_knn_pmp_test_1 = df.loc[df["week"] == 1].reset_index(drop=True)
    X_train_knn_pmp_1 = df_knn_pmp_train_1[features]
    y_train_knn_pmp_x_offset_1 = df_knn_pmp_train_1['x_offset_next_frame']
    y_train_knn_pmp_y_offset_1 = df_knn_pmp_train_1['y_offset_next_frame']
    y_train_knn_pmp_o_1 = df_knn_pmp_train_1['o_next_frame']
    X_test_knn_pmp_1 = df_knn_pmp_test_1[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_pmp_1)
    scaled_features_train_pmp_1 = scaler.transform(X_train_knn_pmp_1)
    X_train_knn_pmp_1[features] = scaled_features_train_pmp_1
    scaled_features_test_pmp_1 = scaler.transform(X_test_knn_pmp_1)
    X_test_knn_pmp_1[features] = scaled_features_test_pmp_1
    knn_pmp_1 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_x_offset_1 = knn_pmp_1.fit(X_train_knn_pmp_1, y_train_knn_pmp_x_offset_1)
    predicted_x_offset_pmp_1 = knn_pmp_x_offset_1.predict(X_test_knn_pmp_1)
    df_knn_pmp_test_1['x_offset_predicted'] = predicted_x_offset_pmp_1
    knn_pmp_1 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_y_offset_1 = knn_pmp_1.fit(X_train_knn_pmp_1, y_train_knn_pmp_y_offset_1)
    predicted_y_offset_pmp_1 = knn_pmp_y_offset_1.predict(X_test_knn_pmp_1)
    df_knn_pmp_test_1['y_offset_predicted'] = predicted_y_offset_pmp_1
    knn_pmp_1 = KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=-1)
    knn_pmp_o_1 = knn_pmp_1.fit(X_train_knn_pmp_1, y_train_knn_pmp_o_1)
    predicted_o_pmp_1 = knn_pmp_o_1.predict(X_test_knn_pmp_1)
    df_knn_pmp_test_1['o_predicted'] = predicted_o_pmp_1
    df_knn_pmp_test_1['x_predicted'] = df_knn_pmp_test_1['x'] + df_knn_pmp_test_1['x_offset_predicted']
    df_knn_pmp_test_1['y_predicted'] = df_knn_pmp_test_1['y'] + df_knn_pmp_test_1['y_offset_predicted']
    df_knn_pmp_test_1['s_predicted'] = (df_knn_pmp_test_1['x_offset_predicted'] ** 2 + df_knn_pmp_test_1[
        'y_offset_predicted'] ** 2) ** 0.5 / 0.1
    return df_knn_pmp_test_1


week8_motion_knn = motion_knn_week_8(defense_all_features_motion)
week7_motion_knn = motion_knn_week_7(defense_all_features_motion)
week6_motion_knn = motion_knn_week_6(defense_all_features_motion)
week5_motion_knn = motion_knn_week_5(defense_all_features_motion)
week4_motion_knn = motion_knn_week_4(defense_all_features_motion)
week3_motion_knn = motion_knn_week_3(defense_all_features_motion)
week2_motion_knn = motion_knn_week_2(defense_all_features_motion)
week1_motion_knn = motion_knn_week_1(defense_all_features_motion)


# Combine all motion KNN dataframes into one
frames = [week1_motion_knn, week2_motion_knn, week3_motion_knn, week4_motion_knn, week5_motion_knn, week6_motion_knn, week7_motion_knn, week8_motion_knn]
all_weeks_defense_motion_knn = pd.concat(frames)
all_weeks_defense_motion_knn = all_weeks_defense_motion_knn.reset_index(drop=True)


# Next, the same KNN as in Pressure_Metric_KNN.py will have to be used, but this time for the predicted motion.
# Ergo, all computations for the pass rushers will have to be redone with the predicted location and speed.
# As a first step, a copy of the original all_weeks dataframe will be created and all x, y, o, and s values will have to
# be updated for all next frame. The first frame will be kept identical, as this one could logically not be predicted.

all_weeks_pmp = all_weeks.copy()

# Handle NaNs in nflId by filling NaNs with a placeholder (-1) for the merge
all_weeks_pmp['nflId'] = all_weeks_pmp['nflId'].fillna(-1).astype(int)
all_weeks_defense_motion_knn['nflId'] = all_weeks_defense_motion_knn['nflId'].fillna(-1).astype(int)

# Create the shifted version of frameId in weeks_defense_pmp to represent frameId - 1
all_weeks_defense_motion_knn['prevFrameId'] = all_weeks_defense_motion_knn['frameId'] + 1  # Create previous frameId (frameId - 1)

# Before merging, save the original index of all_weeks_pmp for frameId > 1
all_weeks_pmp_filtered = all_weeks_pmp[all_weeks_pmp['frameId'] > 1].copy()
all_weeks_pmp_filtered['original_index'] = all_weeks_pmp_filtered.index  # Store the original index

# Perform a merge to only include rows that exist in both dataframes (inner join) for frameId > 1
all_weeks_pmp_merged = pd.merge(
    all_weeks_pmp_filtered,  # Use filtered rows from all_weeks_pmp where frameId > 1
    all_weeks_defense_motion_knn,  # Merge with weeks_defense_pmp
    left_on=['gameId', 'playId', 'nflId', 'frameId'],  # Match on gameId, playId, nflId, and frameId
    right_on=['gameId', 'playId', 'nflId', 'prevFrameId'],  # Match with the previous frame in weeks_defense_pmp
    how='inner'  # Only keep rows that match in both dataframes
)

# Restore the original index from all_weeks_pmp to all_weeks_pmp_merged
all_weeks_pmp_merged.set_index('original_index', inplace=True)

# Update the relevant rows in all_weeks_pmp based on the merged dataframe
# Use the restored original index to update only the matching rows in all_weeks_pmp
# Update columns 'x', 'y', 's', 'a', 'o' from the corresponding predicted columns in all_weeks_pmp_merged
all_weeks_pmp.loc[all_weeks_pmp_merged.index, 'x'] = all_weeks_pmp_merged['x_predicted']
all_weeks_pmp.loc[all_weeks_pmp_merged.index, 'y'] = all_weeks_pmp_merged['y_predicted']
all_weeks_pmp.loc[all_weeks_pmp_merged.index, 's'] = all_weeks_pmp_merged['s_predicted']
all_weeks_pmp.loc[all_weeks_pmp_merged.index, 'o'] = all_weeks_pmp_merged['o_predicted']

# Update offset columns 'x_offset_predicted' and 'y_offset_predicted'
all_weeks_pmp.loc[all_weeks_pmp_merged.index, 'x_offset_predicted'] = all_weeks_pmp_merged['x_offset_predicted']
all_weeks_pmp.loc[all_weeks_pmp_merged.index, 'y_offset_predicted'] = all_weeks_pmp_merged['y_offset_predicted']

# For frameId == 1, ensure 'x_offset_predicted' and 'y_offset_predicted' are set to 0
all_weeks_pmp.loc[all_weeks_pmp['frameId'] == 1, ['x_offset_predicted', 'y_offset_predicted']] = 0

# Restore original NaNs in nflId (if they were originally NaN)
all_weeks_pmp['nflId'] = all_weeks_pmp['nflId'].replace(-1, np.nan)
all_weeks_defense_motion_knn['nflId'] = all_weeks_defense_motion_knn['nflId'].replace(-1, np.nan)

# Clean up helper columns (like prevFrameId) from weeks_defense_pmp and all_weeks_pmp_merged
all_weeks_pmp.drop(columns=['prevFrameId'], inplace=True, errors='ignore')


# Repeat steps from Data_Preprocessing.py

def clean_df(df):
    """
    Function to reverse columns to be true ints again.
    """

    df.nflId = df.nflId.apply(lambda x: int(x))
    df.jerseyNumber = df.jerseyNumber.apply(lambda x: int(x))
    return df


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


# Dataframe only containing only the players important to pass rush, i.e. the Quarterback, the pass rushers, and the pass blockers
pass_rush_pmp = all_weeks_pmp[all_weeks_pmp['pff_role'].isin(['Pass', 'Pass Rush', 'Pass Block'])].reset_index(drop=True)

# Clean the dataframe
pass_rush_pmp = clean_df(pass_rush_pmp)

# Add xyo offsets of previous frame to all players
pass_rush_offsets_pmp = xyo_offset(pass_rush_pmp)

# Create dataframe containing only the pass rushers with all necessary features for each frame
defense_all_features_pmp = (
    pass_rush_offsets_pmp
    .pipe(find_qb_location)
    .query("pff_role != 'Pass'")
    .pipe(join_defense_offense)
    .pipe(find_offense_offsets)
    .pipe(find_closest_o_to_d)
    .pipe(find_angle_d_qb_o)
    .drop(columns=['x_offset_to_qb_offense', 'y_offset_to_qb_offense', 'distance_to_qb_offense'], axis=1)
    .pipe(set_offsets_defense)
)

# Turn pressure into an int variable being 0 for no pressure having occurred and 1 for pressure having occured
defense_all_features_pmp['pressure'] = defense_all_features_pmp['pressure'].astype(int)


# Repeat steps from Pressure_Metric_KNN.py, keeping the original motion as the training data

def pmp_knn_week_8(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 8 using KNN.
    """

    # Split dataframe into training and testing dataframe. Here, training will be weeks 1-7, while testing will be week 8.
    df_knn_train_8 = df_train.loc[df_train["week"] != 8].reset_index(drop=True)
    df_knn_test_8_pmp = df_test.loc[df_test["week"] == 8].reset_index(drop=True)
    df_knn_test_8_pmp = df_knn_test_8_pmp.drop(columns=['x_offset'])
    df_knn_test_8_pmp = df_knn_test_8_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_8 = df_knn_train_8[features]
    y_train_knn_8 = df_knn_train_8['pressure']
    X_test_knn_8_pmp = df_knn_test_8_pmp[features]  # Features
    y_test_knn_8_pmp = df_knn_test_8_pmp['pressure']  # Outcome variable

    # Set the StandardScaler, fit and transform dataframe.
    scaler = StandardScaler()
    scaler.fit(X_train_knn_8)
    scaled_features_train_8 = scaler.transform(X_train_knn_8)
    X_train_knn_8[features] = scaled_features_train_8
    scaled_features_test_8_pmp = scaler.transform(X_test_knn_8_pmp)
    X_test_knn_8_pmp[features] = scaled_features_test_8_pmp

    # Train and fit KNN for week 8 with n=500 and the distance to the neighbors having been given weight.
    knn_8_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_8_pmp = knn_8_pmp.fit(X_train_knn_8, y_train_knn_8)

    # Add probabilities for each prediction to the dataframe, displaying the pressure probability/pressure amount for each pass rusher for each frame.
    probabilities_8_pmp = knn_8_pmp.predict_proba(X_test_knn_8_pmp)
    pressure_probability_pmp = probabilities_8_pmp[:, 1]
    df_knn_test_8_pmp['predicted_motion_pressure'] = pressure_probability_pmp

    # Populate the full dataframe with all players from week 8 with the predicted pressure probabilities. Players not involved in the pass rush are naturally assigned a pressure amount of 0.
    week8_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 8].reset_index(drop=True)
    week8_pmp = pd.merge(week8_pmp,
                         df_knn_test_8_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week8_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week8_pmp

def pmp_knn_week_7(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 7 using KNN.
    """

    df_knn_train_7 = df_train.loc[df_train["week"] != 7].reset_index(drop=True)
    df_knn_test_7_pmp = df_test.loc[df_test["week"] == 7].reset_index(drop=True)
    df_knn_test_7_pmp = df_knn_test_7_pmp.drop(columns=['x_offset'])
    df_knn_test_7_pmp = df_knn_test_7_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_7 = df_knn_train_7[features]
    y_train_knn_7 = df_knn_train_7['pressure']
    X_test_knn_7_pmp = df_knn_test_7_pmp[features]
    y_test_knn_7_pmp = df_knn_test_7_pmp['pressure']
    scaler = StandardScaler()
    scaler.fit(X_train_knn_7)
    scaled_features_train_7 = scaler.transform(X_train_knn_7)
    X_train_knn_7[features] = scaled_features_train_7
    scaled_features_test_7_pmp = scaler.transform(X_test_knn_7_pmp)
    X_test_knn_7_pmp[features] = scaled_features_test_7_pmp
    knn_7_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_7_pmp = knn_7_pmp.fit(X_train_knn_7, y_train_knn_7)
    probabilities_7_pmp = knn_7_pmp.predict_proba(X_test_knn_7_pmp)
    pressure_probability_pmp = probabilities_7_pmp[:, 1]
    df_knn_test_7_pmp['predicted_motion_pressure'] = pressure_probability_pmp
    week7_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 7].reset_index(drop=True)
    week7_pmp = pd.merge(week7_pmp,
                         df_knn_test_7_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week7_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week7_pmp


def pmp_knn_week_6(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 6 using KNN.
    """

    df_knn_train_6 = df_train.loc[df_train["week"] != 6].reset_index(drop=True)
    df_knn_test_6_pmp = df_test.loc[df_test["week"] == 6].reset_index(drop=True)
    df_knn_test_6_pmp = df_knn_test_6_pmp.drop(columns=['x_offset'])
    df_knn_test_6_pmp = df_knn_test_6_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_6 = df_knn_train_6[features]
    y_train_knn_6 = df_knn_train_6['pressure']
    X_test_knn_6_pmp = df_knn_test_6_pmp[features]
    y_test_knn_6_pmp = df_knn_test_6_pmp['pressure']
    scaler = StandardScaler()
    scaler.fit(X_train_knn_6)
    scaled_features_train_6 = scaler.transform(X_train_knn_6)
    X_train_knn_6[features] = scaled_features_train_6
    scaled_features_test_6_pmp = scaler.transform(X_test_knn_6_pmp)
    X_test_knn_6_pmp[features] = scaled_features_test_6_pmp
    knn_6_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_6_pmp = knn_6_pmp.fit(X_train_knn_6, y_train_knn_6)
    probabilities_6_pmp = knn_6_pmp.predict_proba(X_test_knn_6_pmp)
    pressure_probability_pmp = probabilities_6_pmp[:, 1]
    df_knn_test_6_pmp['predicted_motion_pressure'] = pressure_probability_pmp
    week6_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 6].reset_index(drop=True)
    week6_pmp = pd.merge(week6_pmp,
                         df_knn_test_6_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week6_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week6_pmp


def pmp_knn_week_5(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 5 using KNN.
    """

    df_knn_train_5 = df_train.loc[df_train["week"] != 5].reset_index(drop=True)
    df_knn_test_5_pmp = df_test.loc[df_test["week"] == 5].reset_index(drop=True)
    df_knn_test_5_pmp = df_knn_test_5_pmp.drop(columns=['x_offset'])
    df_knn_test_5_pmp = df_knn_test_5_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_5 = df_knn_train_5[features]
    y_train_knn_5 = df_knn_train_5['pressure']
    X_test_knn_5_pmp = df_knn_test_5_pmp[features]
    y_test_knn_5_pmp = df_knn_test_5_pmp['pressure']
    scaler = StandardScaler()
    scaler.fit(X_train_knn_5)
    scaled_features_train_5 = scaler.transform(X_train_knn_5)
    X_train_knn_5[features] = scaled_features_train_5
    scaled_features_test_5_pmp = scaler.transform(X_test_knn_5_pmp)
    X_test_knn_5_pmp[features] = scaled_features_test_5_pmp
    knn_5_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_5_pmp = knn_5_pmp.fit(X_train_knn_5, y_train_knn_5)
    probabilities_5_pmp = knn_5_pmp.predict_proba(X_test_knn_5_pmp)
    pressure_probability_pmp = probabilities_5_pmp[:, 1]
    df_knn_test_5_pmp['predicted_motion_pressure'] = pressure_probability_pmp
    week5_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 5].reset_index(drop=True)
    week5_pmp = pd.merge(week5_pmp,
                         df_knn_test_5_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week5_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week5_pmp


def pmp_knn_week_4(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 4 using KNN.
    """

    df_knn_train_4 = df_train.loc[df_train["week"] != 4].reset_index(drop=True)
    df_knn_test_4_pmp = df_test.loc[df_test["week"] == 4].reset_index(drop=True)
    df_knn_test_4_pmp = df_knn_test_4_pmp.drop(columns=['x_offset'])
    df_knn_test_4_pmp = df_knn_test_4_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_4 = df_knn_train_4[features]
    y_train_knn_4 = df_knn_train_4['pressure']
    X_test_knn_4_pmp = df_knn_test_4_pmp[features]
    y_test_knn_4_pmp = df_knn_test_4_pmp['pressure']
    scaler = StandardScaler()
    scaler.fit(X_train_knn_4)
    scaled_features_train_4 = scaler.transform(X_train_knn_4)
    X_train_knn_4[features] = scaled_features_train_4
    scaled_features_test_4_pmp = scaler.transform(X_test_knn_4_pmp)
    X_test_knn_4_pmp[features] = scaled_features_test_4_pmp
    knn_4_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_4_pmp = knn_4_pmp.fit(X_train_knn_4, y_train_knn_4)
    probabilities_4_pmp = knn_4_pmp.predict_proba(X_test_knn_4_pmp)
    pressure_probability_pmp = probabilities_4_pmp[:, 1]
    df_knn_test_4_pmp['predicted_motion_pressure'] = pressure_probability_pmp
    week4_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 4].reset_index(drop=True)
    week4_pmp = pd.merge(week4_pmp,
                         df_knn_test_4_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week4_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week4_pmp


def pmp_knn_week_3(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 3 using KNN.
    """
    df_knn_train_3 = df_train.loc[df_train["week"] != 3].reset_index(drop=True)
    df_knn_test_3_pmp = df_test.loc[df_test["week"] == 3].reset_index(drop=True)
    df_knn_test_3_pmp = df_knn_test_3_pmp.drop(columns=['x_offset'])
    df_knn_test_3_pmp = df_knn_test_3_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_3 = df_knn_train_3[features]
    y_train_knn_3 = df_knn_train_3['pressure']
    X_test_knn_3_pmp = df_knn_test_3_pmp[features]
    y_test_knn_3_pmp = df_knn_test_3_pmp['pressure']
    scaler = StandardScaler()
    scaler.fit(X_train_knn_3)
    scaled_features_train_3 = scaler.transform(X_train_knn_3)
    X_train_knn_3[features] = scaled_features_train_3
    scaled_features_test_3_pmp = scaler.transform(X_test_knn_3_pmp)
    X_test_knn_3_pmp[features] = scaled_features_test_3_pmp
    knn_3_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_3_pmp = knn_3_pmp.fit(X_train_knn_3, y_train_knn_3)
    probabilities_3_pmp = knn_3_pmp.predict_proba(X_test_knn_3_pmp)
    pressure_probability_pmp = probabilities_3_pmp[:, 1]
    df_knn_test_3_pmp['predicted_motion_pressure'] = pressure_probability_pmp
    week3_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 3].reset_index(drop=True)
    week3_pmp = pd.merge(week3_pmp,
                         df_knn_test_3_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week3_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week3_pmp


def pmp_knn_week_2(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 2 using KNN.
    """

    df_knn_train_2 = df_train.loc[df_train["week"] != 2].reset_index(drop=True)
    df_knn_test_2_pmp = df_test.loc[df_test["week"] == 2].reset_index(drop=True)
    df_knn_test_2_pmp = df_knn_test_2_pmp.drop(columns=['x_offset'])
    df_knn_test_2_pmp = df_knn_test_2_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_2 = df_knn_train_2[features]
    y_train_knn_2 = df_knn_train_2['pressure']
    X_test_knn_2_pmp = df_knn_test_2_pmp[features]
    y_test_knn_2_pmp = df_knn_test_2_pmp['pressure']
    scaler = StandardScaler()
    scaler.fit(X_train_knn_2)
    scaled_features_train_2 = scaler.transform(X_train_knn_2)
    X_train_knn_2[features] = scaled_features_train_2
    scaled_features_test_2_pmp = scaler.transform(X_test_knn_2_pmp)
    X_test_knn_2_pmp[features] = scaled_features_test_2_pmp
    knn_2_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_2_pmp = knn_2_pmp.fit(X_train_knn_2, y_train_knn_2)
    probabilities_2_pmp = knn_2_pmp.predict_proba(X_test_knn_2_pmp)
    pressure_probability_pmp = probabilities_2_pmp[:, 1]
    df_knn_test_2_pmp['predicted_motion_pressure'] = pressure_probability_pmp
    week2_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 2].reset_index(drop=True)
    week2_pmp = pd.merge(week2_pmp,
                         df_knn_test_2_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week2_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week2_pmp


def pmp_knn_week_1(df_train, df_test):
    """
    Function to obtain pressure values for the predicted motion for week 1 using KNN.
    """

    df_knn_train_1 = df_train.loc[df_train["week"] != 1].reset_index(drop=True)
    df_knn_test_1_pmp = df_test.loc[df_test["week"] == 1].reset_index(drop=True)
    df_knn_test_1_pmp = df_knn_test_1_pmp.drop(columns=['x_offset'])
    df_knn_test_1_pmp = df_knn_test_1_pmp.rename(columns={'x_offset_predicted': 'x_offset'})
    X_train_knn_1 = df_knn_train_1[features]
    y_train_knn_1 = df_knn_train_1['pressure']
    X_test_knn_1_pmp = df_knn_test_1_pmp[features]
    y_test_knn_1_pmp = df_knn_test_1_pmp['pressure']
    scaler = StandardScaler()
    scaler.fit(X_train_knn_1)
    scaled_features_train_1 = scaler.transform(X_train_knn_1)
    X_train_knn_1[features] = scaled_features_train_1
    scaled_features_test_1_pmp = scaler.transform(X_test_knn_1_pmp)
    X_test_knn_1_pmp[features] = scaled_features_test_1_pmp
    knn_1_pmp = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_1_pmp = knn_1_pmp.fit(X_train_knn_1, y_train_knn_1)
    probabilities_1_pmp = knn_1_pmp.predict_proba(X_test_knn_1_pmp)
    pressure_probability_pmp = probabilities_1_pmp[:, 1]
    df_knn_test_1_pmp['predicted_motion_pressure'] = pressure_probability_pmp
    week1_pmp = all_weeks_pmp.loc[all_weeks_pmp["week"] == 1].reset_index(drop=True)
    week1_pmp = pd.merge(week1_pmp,
                         df_knn_test_1_pmp[['gameId', 'playId', 'nflId', 'frameId', 'predicted_motion_pressure']],
                         on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week1_pmp['predicted_motion_pressure'].fillna(0, inplace=True)
    return week1_pmp


week8_knn_pmp = pmp_knn_week_8(defense_all_features, defense_all_features_pmp)
week7_knn_pmp = pmp_knn_week_7(defense_all_features, defense_all_features_pmp)
week6_knn_pmp = pmp_knn_week_6(defense_all_features, defense_all_features_pmp)
week5_knn_pmp = pmp_knn_week_5(defense_all_features, defense_all_features_pmp)
week4_knn_pmp = pmp_knn_week_4(defense_all_features, defense_all_features_pmp)
week3_knn_pmp = pmp_knn_week_3(defense_all_features, defense_all_features_pmp)
week2_knn_pmp = pmp_knn_week_2(defense_all_features, defense_all_features_pmp)
week1_knn_pmp = pmp_knn_week_1(defense_all_features, defense_all_features_pmp)

# Combine all KNN dataframes into one
frames_pmp = [week1_knn_pmp, week2_knn_pmp, week3_knn_pmp, week4_knn_pmp, week5_knn_pmp, week6_knn_pmp, week7_knn_pmp, week8_knn_pmp]
weeks_knn_pmp = pd.concat(frames_pmp)
weeks_knn_pmp = weeks_knn_pmp.reset_index(drop=True)

# Defense only
pressure_defense_pmp = weeks_knn_pmp.loc[weeks_knn_pmp["pff_role"] == "Pass Rush"].reset_index(drop=True)
pressure_defense_pmp['pressure'] = pressure_defense_pmp['pressure'].astype(int)

pressure_defense_pmp.to_csv('pressure_defense_pmp.csv')   # CSV file containing PMP values for all pass rushers
