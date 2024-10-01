# Predicted Motion Pressure: Metricizing Pressure Created by Pass Rushers in the NFL and Predicting Their Motions Using K-Nearest Neighbors Machine Learning Models


### Data Science Institute at Hasselt University
Christopher Patzanovsky, PhD Student | [LinkedIn](https://www.linkedin.com/in/christopher-patzanovsky-01a20a100/?originalSubdomain=de)

Prof. Dr. Dirk Valkenborg | [LinkedIn](https://www.linkedin.com/search/results/all/?fetchDeterministicClustersOnly=true&heroEntityKey=urn%3Ali%3Afsd_profile%3AACoAAAEwSEMBKq6Q4PaDN09mGlz9-LmdN2stwhQ&keywords=dirk%20valkenborg&origin=RICH_QUERY_SUGGESTION&position=0&searchId=2a4bef15-cdb4-4f61-80b3-a35390b02b49&sid=k%3B!&spellCorrectionEnabled=false)


## Code
### 1: Data_Preprocessing_Feature_Selection.py
This code creates all the necessary features associated with pressure in pass rush and prepares the data for the upcoming KNN.

### 2: Pressure_Metric_KNN.py
Eight KNN models predicting the pressure probability for every player at every frame, for every play, for every game, for all eight weeks in the 2021 NFL season.

### 3: Predicted_Motion_Pressure.py
Firstly, eight KNN models predict the motion of all pass rushers one frame into the future, by predicting their x-offset, y-offset, and orientation in the next frame. From this their speed, and all pressure-related variables can be computed and input into the earlier pressure KNNs, returning pressure probabilities for the predicted motions of the pass rushers.

### 4: Analysis.py
Obtaining the difference between the average actual pressure and average predicted motion pressure for all players on all plays, and then also averaging those differences for each player individually.


## Data
### nfl-big-data-bowl-2023
This folder contains all data provided by the NFL Big Data Bowl 2023.
- games.csv contains game related data.
- pffScoutingData.csv contains pass rush related data for every player on each play provided by ProFootballFocus (PFF).
- player.csv contains player related data.
- plays.csv contains play related data.
- week1-8.csv contain tracking data for all 22 players and the football on every play every tenth of a second. Irrelevant columns have been dropped here to reduce file sizes.

### Created_Data
- pmp_per_play.csv is created by running Analysis.py and ranks every player based on the highest true pressure vs Predicted Motion Pressure obtained on one play.
- pmp_per_player.csv is created by running Analysis.py and ranks every player based on the highest true pressure vs Predicted Motion Pressure obtained average across all of their plays.
- pmp_per_player_min_5.csv is created by running Analysis.py and ranks every player based on the highest true pressure vs Predicted Motion Pressure obtained average across all of their plays but only accounts pass rushers who have participated in at least five pass rushes.

### Additional Data
These are the .csv files that were created along the computational process and can be found in [this Google Drive folder](https://drive.google.com/drive/folders/1kAPqqgPEN-1JVS7bJWOLVKaUk9XjABNo?usp=sharing).
- all_weeks.csv is created by running Data_Preprocessing_Feature_Selection.py and combines all tracking data into one file.
- defense_all_features.csv is created by running Data_Preprocessing_Feature_Selection.py and contains only the pass rushers with all pressure-related features in regards to their closest pass blocker and quarterback.
- pressure_defense.csv is created by running Pressure_Metric_KNN.py and contains all predicted pressure values for the pass rushers obtained from the KNNs.
- pressure_defense_pmp.csv is created by running Predicted_Motion_Pressure.py and contains all predicted pressure values for the pass rushers based on their predicted motion obtained by the KNNs.
