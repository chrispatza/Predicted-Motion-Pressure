import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


defense_all_features = pd.read_csv("defense_all_features.csv")  # load defense_all_features.csv
all_weeks = pd.read_csv("all_weeks.csv")  # load defense_all_features.csv

# List with all features needed for the KNN
features = ["x_offset", "y_offset", "s", "o", "o_qb", "s_qb", "x_offset_to_qb", "y_offset_to_qb", "o_offense", "s_offense", "x_offset_to_offense", "y_offset_to_offense", "qb_angle"]


def knn_week_8(df):
    """
    Function to obtain pressure values for week 8 using KNN.
    """

    # Split dataframe into training and testing dataframe. Here, training will be weeks 1-7, while testing will be week 8.
    df_knn_train_8 = df.loc[df["week"] != 8].reset_index(drop=True)
    df_knn_test_8 = df.loc[df["week"] == 8].reset_index(drop=True)
    X_train_knn_8 = df_knn_train_8[features]  # Features
    y_train_knn_8 = df_knn_train_8['pressure']  # Outcome variable
    X_test_knn_8 = df_knn_test_8[features]  # Features

    # Set the StandardScaler, fit and transform dataframe.
    scaler = StandardScaler()
    scaler.fit(X_train_knn_8)
    scaled_features_train_8 = scaler.transform(X_train_knn_8)
    X_train_knn_8[features] = scaled_features_train_8
    scaled_features_test_8 = scaler.transform(X_test_knn_8)
    X_test_knn_8[features] = scaled_features_test_8

    # Train and fit KNN for week 8 with n=500 and the distance to the neighbors having been given weight.
    knn_8 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_8 = knn_8.fit(X_train_knn_8, y_train_knn_8)

    # Add probabilities for each prediction to the dataframe, displaying the pressure probability/pressure amount for each pass rusher for each frame.
    probabilities_8 = knn_8.predict_proba(X_test_knn_8)
    pressure_probability = probabilities_8[:, 1]
    df_knn_test_8['pressure_probability'] = pressure_probability

    # Populate the full dataframe with all players from week 8 with the predicted pressure probabilities. Players not involved in the pass rush are naturally assigned a pressure amount of 0.
    week8 = all_weeks.loc[all_weeks["week"] == 8].reset_index(drop=True)
    week8 = pd.merge(week8, df_knn_test_8[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week8['pressure_probability'].fillna(0, inplace=True)
    return week8


# Repeat for all other 7 weeks:
def knn_week_7(df):
    """
    Function to obtain pressure values for week 7 using KNN.
    """

    df_knn_train_7 = df.loc[df["week"] != 7].reset_index(drop=True)
    df_knn_test_7 = df.loc[df["week"] == 7].reset_index(drop=True)
    X_train_knn_7 = df_knn_train_7[features]
    y_train_knn_7 = df_knn_train_7['pressure']
    X_test_knn_7 = df_knn_test_7[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_7)
    scaled_features_train_7 = scaler.transform(X_train_knn_7)
    X_train_knn_7[features] = scaled_features_train_7
    scaled_features_test_7 = scaler.transform(X_test_knn_7)
    X_test_knn_7[features] = scaled_features_test_7
    knn_7 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_7 = knn_7.fit(X_train_knn_7, y_train_knn_7)
    probabilities_7 = knn_7.predict_proba(X_test_knn_7)
    pressure_probability = probabilities_7[:, 1]
    df_knn_test_7['pressure_probability'] = pressure_probability
    week7 = all_weeks.loc[all_weeks["week"] == 7].reset_index(drop=True)
    week7 = pd.merge(week7, df_knn_test_7[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week7['pressure_probability'].fillna(0, inplace=True)
    return week7


def knn_week_6(df):
    """
    Function to obtain pressure values for week 6 using KNN.
    """

    df_knn_train_6 = df.loc[df["week"] != 6].reset_index(drop=True)
    df_knn_test_6 = df.loc[df["week"] == 6].reset_index(drop=True)
    X_train_knn_6 = df_knn_train_6[features]
    y_train_knn_6 = df_knn_train_6['pressure']
    X_test_knn_6 = df_knn_test_6[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_6)
    scaled_features_train_6 = scaler.transform(X_train_knn_6)
    X_train_knn_6[features] = scaled_features_train_6
    scaled_features_test_6 = scaler.transform(X_test_knn_6)
    X_test_knn_6[features] = scaled_features_test_6
    knn_6 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_6 = knn_6.fit(X_train_knn_6, y_train_knn_6)
    probabilities_6 = knn_6.predict_proba(X_test_knn_6)
    pressure_probability = probabilities_6[:, 1]
    df_knn_test_6['pressure_probability'] = pressure_probability
    week6 = all_weeks.loc[all_weeks["week"] == 6].reset_index(drop=True)
    week6 = pd.merge(week6, df_knn_test_6[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week6['pressure_probability'].fillna(0, inplace=True)
    return week6


def knn_week_5(df):
    """
    Function to obtain pressure values for week 5 using KNN.
    """

    df_knn_train_5 = df.loc[df["week"] != 5].reset_index(drop=True)
    df_knn_test_5 = df.loc[df["week"] == 5].reset_index(drop=True)
    X_train_knn_5 = df_knn_train_5[features]
    y_train_knn_5 = df_knn_train_5['pressure']
    X_test_knn_5 = df_knn_test_5[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_5)
    scaled_features_train_5 = scaler.transform(X_train_knn_5)
    X_train_knn_5[features] = scaled_features_train_5
    scaled_features_test_5 = scaler.transform(X_test_knn_5)
    X_test_knn_5[features] = scaled_features_test_5
    knn_5 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_5 = knn_5.fit(X_train_knn_5, y_train_knn_5)
    probabilities_5 = knn_5.predict_proba(X_test_knn_5)
    pressure_probability = probabilities_5[:, 1]
    df_knn_test_5['pressure_probability'] = pressure_probability
    week5 = all_weeks.loc[all_weeks["week"] == 5].reset_index(drop=True)
    week5 = pd.merge(week5, df_knn_test_5[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week5['pressure_probability'].fillna(0, inplace=True)
    return week5


def knn_week_4(df):
    """
    Function to obtain pressure values for week 4 using KNN.
    """

    df_knn_train_4 = df.loc[df["week"] != 4].reset_index(drop=True)
    df_knn_test_4 = df.loc[df["week"] == 4].reset_index(drop=True)
    X_train_knn_4 = df_knn_train_4[features]
    y_train_knn_4 = df_knn_train_4['pressure']
    X_test_knn_4 = df_knn_test_4[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_4)
    scaled_features_train_4 = scaler.transform(X_train_knn_4)
    X_train_knn_4[features] = scaled_features_train_4
    scaled_features_test_4 = scaler.transform(X_test_knn_4)
    X_test_knn_4[features] = scaled_features_test_4
    knn_4 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_4 = knn_4.fit(X_train_knn_4, y_train_knn_4)
    probabilities_4 = knn_4.predict_proba(X_test_knn_4)
    pressure_probability = probabilities_4[:, 1]
    df_knn_test_4['pressure_probability'] = pressure_probability
    week4 = all_weeks.loc[all_weeks["week"] == 4].reset_index(drop=True)
    week4 = pd.merge(week4, df_knn_test_4[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week4['pressure_probability'].fillna(0, inplace=True)
    return week4


def knn_week_3(df):
    """
    Function to obtain pressure values for week 3 using KNN.
    """

    df_knn_train_3 = df.loc[df["week"] != 3].reset_index(drop=True)
    df_knn_test_3 = df.loc[df["week"] == 3].reset_index(drop=True)
    X_train_knn_3 = df_knn_train_3[features]
    y_train_knn_3 = df_knn_train_3['pressure']
    X_test_knn_3 = df_knn_test_3[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_3)
    scaled_features_train_3 = scaler.transform(X_train_knn_3)
    X_train_knn_3[features] = scaled_features_train_3
    scaled_features_test_3 = scaler.transform(X_test_knn_3)
    X_test_knn_3[features] = scaled_features_test_3
    knn_3 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_3 = knn_3.fit(X_train_knn_3, y_train_knn_3)
    probabilities_3 = knn_3.predict_proba(X_test_knn_3)
    pressure_probability = probabilities_3[:, 1]
    df_knn_test_3['pressure_probability'] = pressure_probability
    week3 = all_weeks.loc[all_weeks["week"] == 3].reset_index(drop=True)
    week3 = pd.merge(week3, df_knn_test_3[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week3['pressure_probability'].fillna(0, inplace=True)
    return week3


def knn_week_2(df):
    """
    Function to obtain pressure values for week 2 using KNN.
    """

    df_knn_train_2 = df.loc[df["week"] != 2].reset_index(drop=True)
    df_knn_test_2 = df.loc[df["week"] == 2].reset_index(drop=True)
    X_train_knn_2 = df_knn_train_2[features]
    y_train_knn_2 = df_knn_train_2['pressure']
    X_test_knn_2 = df_knn_test_2[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_2)
    scaled_features_train_2 = scaler.transform(X_train_knn_2)
    X_train_knn_2[features] = scaled_features_train_2
    scaled_features_test_2 = scaler.transform(X_test_knn_2)
    X_test_knn_2[features] = scaled_features_test_2
    knn_2 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_2 = knn_2.fit(X_train_knn_2, y_train_knn_2)
    probabilities_2 = knn_2.predict_proba(X_test_knn_2)
    pressure_probability = probabilities_2[:, 1]
    df_knn_test_2['pressure_probability'] = pressure_probability
    week2 = all_weeks.loc[all_weeks["week"] == 2].reset_index(drop=True)
    week2 = pd.merge(week2, df_knn_test_2[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week2['pressure_probability'].fillna(0, inplace=True)
    return week2


def knn_week_1(df):
    """
    Function to obtain pressure values for week 1 using KNN.
    """

    df_knn_train_1 = df.loc[df["week"] != 1].reset_index(drop=True)
    df_knn_test_1 = df.loc[df["week"] == 1].reset_index(drop=True)
    X_train_knn_1 = df_knn_train_1[features]
    y_train_knn_1 = df_knn_train_1['pressure']
    X_test_knn_1 = df_knn_test_1[features]
    scaler = StandardScaler()
    scaler.fit(X_train_knn_1)
    scaled_features_train_1 = scaler.transform(X_train_knn_1)
    X_train_knn_1[features] = scaled_features_train_1
    scaled_features_test_1 = scaler.transform(X_test_knn_1)
    X_test_knn_1[features] = scaled_features_test_1
    knn_1 = KNeighborsClassifier(n_neighbors=500, weights='distance', n_jobs=-1)
    knn_1 = knn_1.fit(X_train_knn_1, y_train_knn_1)
    probabilities_1 = knn_1.predict_proba(X_test_knn_1)
    pressure_probability = probabilities_1[:, 1]
    df_knn_test_1['pressure_probability'] = pressure_probability
    week1 = all_weeks.loc[all_weeks["week"] == 1].reset_index(drop=True)
    week1 = pd.merge(week1, df_knn_test_1[['gameId', 'playId', 'nflId', 'frameId', 'pressure_probability']],
                     on=['gameId', 'playId', 'nflId', 'frameId'], how='outer')
    week1['pressure_probability'].fillna(0, inplace=True)
    return week1


week8_knn = knn_week_8(defense_all_features)
week7_knn = knn_week_7(defense_all_features)
week6_knn = knn_week_6(defense_all_features)
week5_knn = knn_week_5(defense_all_features)
week4_knn = knn_week_4(defense_all_features)
week3_knn = knn_week_3(defense_all_features)
week2_knn = knn_week_2(defense_all_features)
week1_knn = knn_week_1(defense_all_features)

# Combine all KNN dataframes into one
frames = [week1_knn, week2_knn, week3_knn, week4_knn, week5_knn, week6_knn, week7_knn, week8_knn]
weeks_knn = pd.concat(frames)
weeks_knn = weeks_knn.reset_index(drop=True)

# Defense only
pressure_defense = weeks_knn.loc[weeks_knn["pff_role"] == "Pass Rush"].reset_index(drop=True)
pressure_defense['pressure'] = pressure_defense['pressure'].astype(int)

pressure_defense.to_csv('pressure_defense.csv')   # CSV file containing pressure values for all pass rushers
