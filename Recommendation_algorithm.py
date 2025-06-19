import mplsoccer
from mplsoccer import Pitch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import statsmodels.api as sm
import streamlit as st
from fuzzywuzzy import process, fuzz
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold

# 1. Load the data
df = pd.read_csv("CompleteDataset.csv")
df_transfers = pd.read_csv("SuccessfulAdaptationHistoricData.csv")
df = df.dropna()
df_transfers = df_transfers.dropna()
season_of_interest = '2024/2025'

df['Latitude_country_player'] = df['Latitude_country_player'].astype(str).str.replace(',', '.').astype(float)
df['Longitude_country_player'] = df['Longitude_country_player'].astype(str).str.replace(',', '.').astype(float)
df['club_country_latitude'] = df['club_country_latitude'].astype(str).str.replace(',', '.').astype(float)
df['club_country_longitude'] = df['club_country_longitude'].astype(str).str.replace(',', '.').astype(float)

# Dictionary of features by position
position_features = {
            "Goalkeeper": ["player_id", "weight","height_cm", "foot_id", "is_top5_league_team",
                "total_matches", "FIFA_score", "gk_diving_score", "gk_handling_score", "gk_kicking_score", "gk_positioning_score",
                "gk_reflexes_score", "jumping_score", "reactions_score", "strength_score", "stamina_score", "minutes_played",
                "tot_clean_sheets", "total_missed_matches_inj", "yellow_cards", "red_cards", "conceeded_goals","completed_minutes_ratio"
            ],
            "Centre-Back": ["player_id", "weight","height_cm", "foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "defense_standing_tackle_|score",
                "defense_sliding_tackles_score", "interceptions_score", "heading_accuracy_score", "jumping_score", "strength_score",
                "stamina_score", "balance_score", "reactions_score", "short_passing_score", "passing_score", "shot_power_score",
                "composure_score", "yellow_cards", "red_cards", "conceeded_goals", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Left-Back": ["player_id", "weight","height_cm", "foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "defense_standing_tackle_score",
                "defense_sliding_tackles_score", "interceptions_score", "heading_accuracy_score", "jumping_score", "strength_score",
                "stamina_score", "balance_score", "reactions_score", "short_passing_score", "passing_score", "acceleration_score",
                "sprint_score", "crossing_score", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Right-Back": ["player_id", "weight","height_cm", "foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "defense_standing_tackle_score",
                "defense_sliding_tackles_score", "interceptions_score", "heading_accuracy_score", "jumping_score", "strength_score",
                "stamina_score", "balance_score", "reactions_score", "short_passing_score", "passing_score", "acceleration_score",
                "sprint_score", "crossing_score", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Defensive Midfield": ["player_id", "weight","height_cm", "foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "interceptions_score",
                "short_passing_score", "passing_score", "long_passing_score", "vision_score", "stamina_score", "balance_score",
                "strength_score", "reactions_score", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Central Midfield": ["player_id", "weight","height_cm", "foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "short_passing_score", "passing_score", "long_passing_score", "vision_score",
                "dribbling_score", "ball_control_score", "composure_score", "stamina_score", "reactions_score", "yellow_cards",
                "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Attacking Midfield": ["player_id", "weight","foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "vision_score", "short_passing_score", "passing_score", "long_passing_score",
                "dribbling_score", "ball_control_score", "composure_score", "agility_score", "finishing_score", "shot_power_score",
                "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Left Winger": ["player_id", "height_cm", "weight","foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "passing_score", "finishing_score", "shot_power_score", "ball_control_score", "stamina_score",
                "vision_score", "goals", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Right Winger": ["player_id", "height_cm", "weight", "foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "passing_score", "finishing_score", "shot_power_score", "ball_control_score", "stamina_score",
                "vision_score", "goals", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Left Midfield": ["player_id", "height_cm", "weight", "foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "short_passing_score", "passing_score", "long_passing_score", "stamina_score", "vision_score",
                "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Right Midfield": ["player_id", "weight", "height_cm","foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "short_passing_score", "passing_score", "long_passing_score", "stamina_score", "vision_score",
                "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Centre-Forward": ["player_id", "height_cm", "weight","foot_id",  "is_top5_league_team",
                "total_matches", "FIFA_score", "goals", "finishing_score", "shot_power_score", "heading_accuracy_score",
                "ball_control_score", "agility_score", "acceleration_score", "sprint_score", "balance_score", "dribbling_score",
                "jumping_score", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ],
            "Second Striker": ["player_id","height_cm", "weight","foot_id", "is_top5_league_team",
                "total_matches", "FIFA_score", "goals", "finishing_score", "shot_power_score", "vision_score", "dribbling_score",
                "agility_score", "acceleration_score", "sprint_score", "balance_score", "ball_control_score", "assists",
                "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj","completed_minutes_ratio"
            ]
        }

# Fuzzy function to find similar players by name
def find_similar_players_by_name(player_name, df):
    df_2024_2025 = df[df["season"] == "2024/2025"]
    player_names = df_2024_2025["player_name"].tolist()
    matches = process.extract(player_name, player_names, limit=10, scorer=fuzz.token_sort_ratio)
    similar_players = []
    for match in matches:
        matched_player = df_2024_2025[df_2024_2025["player_name"] == match[0]].iloc[0]
        similar_players.append({
            "player_name": matched_player["player_name"],
            "player_id": matched_player["player_id"],
            "club_name": matched_player["club_name"],
            "field_position": matched_player["field_position"],
            "field_sub_position": matched_player["field_sub_position"],
            "age": matched_player["age"],
            "club_country": matched_player["club_country"]

        })
    return pd.DataFrame(similar_players)

# Streamlit UI
st.title("Football Player Replacement Recommendation")

# Restart full flow
if st.button("🔁 Restart Full Search"):
    keys_to_clear = [
        'player_input', 'player_confirmed', 'selected_id', 
        'selected_player', 'input_player_team', 'input_player_position','input_player_age','input_club_country']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Setup player input
if "player_input" not in st.session_state:
    st.session_state.player_input = ""

player_name_input = st.text_input(
    "Enter the name of the player to replace:",
    value=st.session_state.player_input,
    key="player_input"
)

if player_name_input:
    candidates_df = find_similar_players_by_name(player_name_input, df)
    
    if candidates_df.empty:
        st.warning("No players found.")
    else:
        if "player_confirmed" not in st.session_state:
            st.session_state.player_confirmed = False

        if not st.session_state.player_confirmed:
            with st.form("player_id_confirmation"):
            
                selected_id = st.selectbox(
                    "Confirm the player selection:",
                    options=candidates_df["player_id"].tolist(),
                    format_func=lambda x: f"{candidates_df.loc[candidates_df['player_id'] == x, 'player_name'].values[0]} ({candidates_df.loc[candidates_df['player_id'] == x, 'club_name'].values[0]})"
)
                st.write("If you are sure of the player of your choice, please continue.")
                continue_button0 = st.form_submit_button("Continue")



            if continue_button0:
                st.session_state.selected_id = int(selected_id)
                selected_player = candidates_df[candidates_df["player_id"] == st.session_state.selected_id].iloc[0]
                st.session_state.selected_player = selected_player.to_dict()
                st.session_state.player_confirmed = True
                st.rerun()

        else:
            selected_player = st.session_state.selected_player
            input_name = selected_player["player_name"]
            input_player_team = selected_player["club_name"]
            input_player_position = selected_player["field_sub_position"]
            input_player_general_position = selected_player["field_position"]
            input_player_age = selected_player["age"]
            input_club_country = selected_player["club_country"]
            
            st.markdown(f"**Selected: {selected_player['player_name']} - {input_player_position} - {selected_player['club_name']}**")

            selected_player_details = df[
                (df["player_id"] == st.session_state.selected_id) & 
                (df["club_name"] == input_player_team)
                #&(df["season"] == season_of_interest)
            ]
        

            most_recent_season = season_of_interest
            recent_season_data = selected_player_details[selected_player_details['season'] == most_recent_season]

            age= selected_player_details['age'].values[0]
            total_matches = selected_player_details['total_matches'].sum()
            total_minutes = selected_player_details['minutes_played'].sum()
            total_goals = selected_player_details['goals'].sum()
            total_assists = selected_player_details['assists'].sum()
            total_yellows = selected_player_details['yellow_cards'].sum()
            total_reds = selected_player_details['red_cards'].sum()
            fifa_score = recent_season_data['FIFA_score'].values[0] 
            

            
            st.subheader("🧾 Player to be replaced")
            st.markdown("""
            <style>
            .stat-card {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 24px 28px 20px 28px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                max-width: 420px;
                margin: 0 auto 24px auto;
            }
            .stat-card h4 {
                margin-top: 0;
                margin-bottom: 18px;
                color: #1f77b4;
                text-align: center;
            }
            .stat-list {
                list-style: none;
                padding-left: 0;
                margin-bottom: 0;
            }
            .stat-list li {
                padding: 8px 0;
                border-bottom: 1px solid #ececec;
                font-size: 16px;
                display: flex;
                align-items: center;
            }
            .stat-list li:last-child {
                border-bottom: none;
            }
            .stat-label {
                flex: 1;
                color: #555;
                font-weight: 600;
            }
            .stat-value {
                flex: 1;
                text-align: right;
                font-weight: bold;
                color: #222;
            }
            </style>
            """, unsafe_allow_html=True)

            # HTML block for the stat card
            st.markdown(f"""
            <div class="stat-card">
                <h4>📊 Stats since 2018/2019 season</h4>
                <ul class="stat-list">
                    <li><span class="stat-label">Name</span><span class="stat-value">{input_name}</span></li>    
                    <li><span class="stat-label">Total Matches Played</span><span class="stat-value">{total_matches}</span></li>
                    <li><span class="stat-label">Goals</span><span class="stat-value">{total_goals}</span></li>
                    <li><span class="stat-label">Assists</span><span class="stat-value">{total_assists}</span></li>
                    <li><span class="stat-label">Yellow Cards</span><span class="stat-value">{total_yellows}</span></li>
                    <li><span class="stat-label">Red Cards</span><span class="stat-value">{total_reds}</span></li>
                    <li><span class="stat-label">FIFA Score (Most Recent Season)</span><span class="stat-value">{fifa_score}</span></li>
               </ul>
            </div>
            """, unsafe_allow_html=True)

            with st.form("player_filter_form"):
                st.subheader("Transfer filters")
                min_age, max_age = st.slider("Select age range", 16, 40, (20, 30))
                min_val, max_val = st.slider("Select market value range (millions €)", 0, 150, (0, 40))
                exclude_teams = st.multiselect(
                    "Exclude players from these teams",
                    options=df["club_name"].unique(),
                    default=[]
                )
                continue_button1 = st.form_submit_button("Find Replacements")

            if continue_button1:
                df_2024_2025 = df[df["season"] == "2024/2025"]
                
                df_filtered = df_2024_2025[
                (df_2024_2025["player_id"] == st.session_state.selected_id) |
                (
                    (df_2024_2025["field_position"] == input_player_general_position) &
                    (df_2024_2025["age"] >= min_age) &
                    (df_2024_2025["age"] <= max_age) &
                    (df_2024_2025["market_value"] >= min_val * 1_000_000) &
                    (df_2024_2025["market_value"] <= max_val * 1_000_000) &
                    (df_2024_2025["club_name"] != input_player_team)&
                    # Exclude players from the specified teams
                    (df_2024_2025["club_name"].isin(exclude_teams) == False))
                    ]
                
                position_feature_set = position_features[input_player_position]
               
                # Include additional features for the selected player
                position_feature_set += ["club_country", "country","Latitude_country_player","Longitude_country_player", "club_country_latitude","club_country_longitude"]
                df_filtered = df_filtered[position_feature_set]
                
                
                # Filter complete dataset including the id players taken in the df_filtered


                # The first position of df_filter_complete is the player we are going to replace
                df_filter_complete = pd.DataFrame(columns=position_feature_set)
                st.write(df_filter_complete = pd.DataFrame(columns=position_feature_set)
)
                
                
                df_filter_complete= df[df["player_id"] == st.session_state.selected_id]
                
                # Add the filtered players to the complete dataset
                df_filter_complete = df[df["player_id"].isin(df_filtered["player_id"])]
                df_filter_complete=df_filter_complete[position_feature_set]
                
                # Use the selected player's club country latitude and longitude
                clubs_latitude = df[df["player_id"] == st.session_state.selected_id]["club_country_latitude"].values[0]
                clubs_longitude = df[df["player_id"] == st.session_state.selected_id]["club_country_longitude"].values[0]
                 

                # Calculate the distance between the player's country and the club's country using the Haversine formula
                def haversine_np(lat1, lon1, lat2, lon2):
                    R = 6371.0  # km

                    phi1 = np.radians(lat1)
                    phi2 = np.radians(lat2)
                    d_phi = np.radians(lat2 - lat1)
                    d_lambda = np.radians(lon2 - lon1)

                    a = np.sin(d_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

                    return R * c
                df_filter_complete['distance_nationality_league'] = haversine_np(
                    df["Latitude_country_player"],
                    df["Longitude_country_player"],
                    clubs_latitude,
                    clubs_longitude
                )
              
                # Exclude latitude and longitude columns from position_feature_set
                position_feature_set = [
                    col for col in position_feature_set 
                    if col not in [
                        "Latitude_country_player", "Longitude_country_player", 
                        "club_country_latitude", "club_country_longitude",
                        "club_country", "country"
                    ]
                ]
                # Include distance_nationality_league in position_feature_set
                position_feature_set += ["distance_nationality_league"]
                

                def compute_weighted_features(player_df, player_id):
                    
                    player_data = player_df[player_df["player_id"] == player_id].copy()
                    if "distance_nationality_league" not in player_data.columns:
                        player_data["distance_nationality_league"] = haversine_np(
                            player_data["Latitude_country_player"],
                            player_data["Longitude_country_player"],
                            clubs_latitude,
                            clubs_longitude
                        )
                    # Calculate exponential weights
                    player_data["exp_weight"] = np.exp(-0.1 * (int(season_of_interest[:4]) - player_data["season"].str[:4].astype(int)))
                    player_data["exp_weight"] /= player_data["exp_weight"].sum()

                    # Filter for available features only
                    features_to_use = [col for col in position_feature_set if col in player_data.columns]

                    # Compute weighted average of selected features
                    weighted = (player_data[features_to_use].T * player_data["exp_weight"].values).T.sum()
                    weighted["player_id"] = player_id

                    return weighted
                    


                                        
                # Create a new DataFrame with the averaged features per player
                def averaged_features_player(df_filter_complete):
                    records = []
                    for pid in df_filter_complete["player_id"].unique():
                        records.append(compute_weighted_features(df, pid))
                    df_weighted = pd.DataFrame(records)
      
                    return df_weighted
                
                df_weighted = averaged_features_player(df_filter_complete)
                
                # Perform the KNN search
                input_vector = df_weighted[df_weighted["player_id"] == st.session_state.selected_id]
    
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_weighted.drop(columns=["player_id"]))
                input_scaled = scaler.transform(input_vector.drop(columns=["player_id"]))
                
                # Calculate the distance to the player we are going to replace for the whole dataset

                total_candidates = df_weighted.shape[0]-1
                total_candidates_field_position=df['field_position']
                knn_all = NearestNeighbors(n_neighbors=total_candidates)
                knn_all.fit(X_scaled)
                distance_total, indices_total = knn_all.kneighbors(input_scaled)
                average_knn_distance_total = round(distance_total.mean(),2)

                # Perform the KNN for the 10 closest players                                      
                knn = NearestNeighbors(n_neighbors=11)
                knn.fit(X_scaled)
                distances, indices = knn.kneighbors(input_scaled)
                average_knn_distance = round(distances.mean(),2)
                average_top_10_vs_total = abs(round(((average_knn_distance/average_knn_distance_total)-1),2))

                similar_players = df_weighted.iloc[indices[0]].copy()
                similar_players["distance"] = distances[0]
                similar_players = similar_players[similar_players["player_id"] != st.session_state.selected_id]

                season_of_interest = '2024/2025'
                info = df[df['season'] == season_of_interest][['season','player_id', 'age', 'player_name', 'country', 'club_name','field_sub_position','market_value','season']].drop_duplicates('player_id')
                #info_cols = df[["player_id", "player_name", "club_name",  "country","field_sub_position"]].drop_duplicates("player_id")
                
               
                similar_players = similar_players.merge(info, on="player_id")
                #Take out duplicated columns
                similar_players = similar_players.loc[:,~similar_players.columns.duplicated()]
              
                # Show the top 10 players
                st.write("**The 10 most similar players to", selected_player["player_name"]," are:**")
                similar_players["rank"] = range(1, len(similar_players) + 1)
                
                st.dataframe(similar_players[["rank","player_name", "club_name", "age", "field_sub_position"]].head(10),hide_index=True)
                # Take out the distance column
                similar_players = similar_players.drop(columns=["distance_nationality_league"])
                # and take the column out of the position feature set
                position_feature_set = [    
                    col for col in position_feature_set 
                    if col not in ["distance_nationality_league"]]
                

                # Logistic regression model
                # Filter only players with the same position as the input player
                X = df_transfers[df_transfers["field_sub_position"] == input_player_position]
                
                # Select only the relevant features
                X = X[position_feature_set]
                y = df_transfers.loc[X.index, 'successful_adaptation']
                
                # Drop the player_id column
                X = X.drop(columns=["player_id"])


                # Split the data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Scale the features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Add constant term for statsmodels
                X_train_sm = sm.add_constant(X_train_scaled)
                

                # Fit logistic regression model using statsmodels (to get p-values)
                logit_model = sm.Logit(y_train, X_train_sm)
                result = logit_model.fit(disp=False)  # Suppress console output (good for Streamlit)

                # Optionally display model summary with p-values
                with st.expander("Show model summary with p-values"):
                    st.text(result.summary())

                # Select only statistically significant features (p < 0.05), excluding the intercept
                significant_indices = result.pvalues[result.pvalues < 0.05].index
                significant_indices = [i for i in significant_indices if i != 'const']

                # Helper function to map 'x0', 'x1'... from statsmodels back to column names
                def map_x_to_colname(xname, columns):
                    if xname.startswith('x') and xname[1:].isdigit():
                        idx = int(xname[1:])
                        if 0 <= idx < len(columns):
                            return columns[idx]
                    return None

                # Map indices to column names and filter out None values
                significant_columns = [map_x_to_colname(col, X.columns) for col in significant_indices]
                significant_columns = [col for col in significant_columns if col is not None]

                # Get column positions for indexing into scaled numpy arrays
                significant_col_indices = [X.columns.get_loc(col) for col in significant_columns]

                st.subheader("Significant variables for the Logistic Regression Model at p < 0.05")
                st.write("Significant features:", significant_columns)

                # Prepare data for cross-validation
                X_train_sig = X_train_scaled[:, significant_col_indices]
                X_test_sig = X_test_scaled[:, significant_col_indices]

                # Prepare the similar players data, scaled and filtered by the same features
                X_similar = scaler.transform(similar_players[position_feature_set].drop(columns=["player_id"]))
                X_similar_sig = X_similar[:, significant_col_indices]

                # Lists to store fold metrics
                probs_list = []
                accuracies = []
                roc_aucs = []
                precisions = []
                recalls = []
                f1_scores = []
                fold = 1

                # st.subheader("K-Fold Cross-Validation Performance")
                # Initialize cross-validation
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                # Perform K-Fold cross-validation
                for train_idx, val_idx in skf.split(X_train_sig, y_train):
                    X_tr, X_val = X_train_sig[train_idx], X_train_sig[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    model_cv = LogisticRegression(max_iter=1000)
                    model_cv.fit(X_tr, y_tr)

                    y_pred = model_cv.predict(X_val)
                    y_proba = model_cv.predict_proba(X_val)[:, 1]

                    # Compute metrics
                    acc = accuracy_score(y_val, y_pred)
                    auc = roc_auc_score(y_val, y_proba)
                    prec = precision_score(y_val, y_pred, zero_division=0)
                    rec = recall_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred)

                    # Save metrics for averaging later
                    accuracies.append(acc)
                    roc_aucs.append(auc)
                    precisions.append(prec)
                    recalls.append(rec)
                    f1_scores.append(f1)

                    # Predict probabilities for similar players in this fold
                    probs_list.append(model_cv.predict_proba(X_similar_sig)[:, 1])

                    # Confusion matrix
                    cm = confusion_matrix(y_val, y_pred)

                    st.write(f"Fold {fold}: Accuracy = {acc:.4f}, ROC AUC = {auc:.4f}, "
                            f"Precision = {prec:.4f}, Recall = {rec:.4f}, F1 Score = {f1:.4f}")
                    fold += 1

                
                # Average the predicted probabilities for similar players
                mean_probs = sum(probs_list) / len(probs_list)
                similar_players["adaptation_prob"] = mean_probs
                similar_players = similar_players.sort_values("adaptation_prob", ascending=False)
                similar_players = similar_players.reset_index(drop=True)
                similar_players["rank"] = range(1, len(similar_players) + 1)
                
                st.subheader("Successful Adaptation Probability Ranking")
                st.dataframe(similar_players[["rank","player_name", "adaptation_prob"]].head(10),hide_index=True)
                st.subheader("Recommended Replacement")

                # Give the best candidate the sub_position of the player we are going to replace
                st.write("**The player with the highest probability of adaptation is to replace: ", selected_player["player_name"]," is:**")
                best_candidate_player = similar_players.iloc[0]
                best_candidate_player["field_sub_position"] = input_player_position
                best_candidate_player["field_position"] = input_player_general_position

                col1, col2 = st.columns(2)
                col1.markdown(f"**Player Name:** {best_candidate_player['player_name']}")
                col2.markdown(f"**Current Club:** {best_candidate_player['club_name']}")

                col3, col4 = st.columns(2)
                col3.markdown(f"**Age:** {best_candidate_player['age']}")
                col4.markdown(f"**Adaptation Probability:** {best_candidate_player['adaptation_prob']:.2%}")
                
                
                 # General stats of the replaced player in the current season
                to_be_replaced = df[
                (df["player_name"] ==input_name) &
                (df["season"] == season_of_interest)
                    ]
                
                # General stats of the new player in the current season
                recommended_player_details = df[
                (df["player_name"] ==best_candidate_player['player_name']) & 
                (df["club_name"] ==best_candidate_player['club_name'] )&
                (df["season"] == season_of_interest)
            ]

                most_recent_season = season_of_interest
                recent_season_data = recommended_player_details[recommended_player_details['season'] == most_recent_season]

                age_p1= round(to_be_replaced['age'].values[0],0)
                total_matches_p1 = to_be_replaced['total_matches'].sum()
                total_minutes_p1 = to_be_replaced['minutes_played'].sum()
                total_goals_p1 = to_be_replaced['goals'].sum()
                total_assists_p1 = to_be_replaced['assists'].sum()
                total_yellow_p1 = to_be_replaced['yellow_cards'].sum()
                total_reds_p1 = to_be_replaced['red_cards'].sum()
                fifa_score_p1 = to_be_replaced['FIFA_score'].values[0]

                age_p2= round(recommended_player_details['age'].values[0],0)
                total_matches_p2 = recommended_player_details['total_matches'].sum()
                total_minutes_p2 = recommended_player_details['minutes_played'].sum()
                total_goals_p2 = recommended_player_details['goals'].sum()
                total_assists_p2 = recommended_player_details['assists'].sum()
                total_yellow_p2 = recommended_player_details['yellow_cards'].sum()
                total_reds_p2 = recommended_player_details['red_cards'].sum()
                fifa_score_p2= recommended_player_details['FIFA_score'].values[0]
                
                current_season = season_of_interest
                
                # Compare the stats of the new player with the replaced player
                st.subheader("📋 Player Replacement Comparison")
                # Custom CSS in a separate markdown block
                
                st.markdown("""
                <style>
                .compare-card {
                    background-color: #ffffff;
                    border-radius: 12px;
                    padding: 20px;
                    color: #333;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    max-width: 720px;
                    margin: 0 auto;
                
                }
                .compare-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 12px 0;
                    border-bottom: 1px solid #e0e0e0;
                }
                .compare-row.header {
                    font-weight: bold;
                    border-bottom: 2px solid #ccc;
                }
                .label {
                    flex: 1;
                    text-align: center;
                    font-size: 16px;
                }
                .label-title {
                    flex: 1;
                    text-align: center;
                    font-weight: bold;
                    font-size: 16px;
                    color: #555;
                }
                .player-name {
                    font-size: 20px;
                    font-weight: bold;
                    text-align: center;
                    color: #222;
                }
                .player-title {
                    font-size: 14px;
                    text-align: center;
                    color: #888;
                    font-weight: bold;
                    margin-bottom: 2px;
                }
                .compare-row:nth-child(even) {
                    background-color: #f9f9f9;
                }
                </style>
                """, unsafe_allow_html=True)

                # Custom HTML in a separate markdown block
                st.markdown(f"""
                <div class="compare-card">
                    <div class="compare-card">
                        <div class="stats-subtitle">Stats this season: {current_season}</div>
                        <div class="compare-row header">
                            <div class="label"></div>
                            <div class="label">
                            <div class="player-title">Player to be replaced</div>
                            <div class="player-name" style="color: #1f77b4;">{input_name}</div>
                        </div>
                        <div class="label">
                            <div class="player-title">Recommended replacement</div>
                            <div class="player-name" style="color: #d62728;">{best_candidate_player['player_name']}</div>
                        </div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">Age</div>
                        <div class="label">{int(age_p1)} years</div>
                        <div class="label">{int(age_p2)} years</div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">Matches</div>
                        <div class="label">{total_matches_p1}</div>
                        <div class="label">{total_matches_p2}</div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">Minutes</div>
                        <div class="label">{total_minutes_p1} min</div>
                        <div class="label">{total_minutes_p2} min</div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">Goals</div>
                        <div class="label">{total_goals_p1}</div>
                        <div class="label">{total_goals_p2}</div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">Assists</div>
                        <div class="label">{total_assists_p1}</div>
                        <div class="label">{total_assists_p2}</div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">Yellow cards</div>
                        <div class="label">{total_yellow_p1}</div>
                        <div class="label">{total_yellow_p2}</div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">Red cards</div>
                        <div class="label">{total_reds_p1}</div>
                        <div class="label">{total_reds_p2}</div>
                    </div>
                    <div class="compare-row">
                        <div class="label-title">FIFA Score</div>
                        <div class="label">{fifa_score_p1}</div>
                        <div class="label">{fifa_score_p2}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


                # Results of the algorithm
                st.subheader("📊 Results of the considered algorithms")
                # KNN Performance metrics
                st.subheader("📊 10-Nearest Neighbors Performance Metrics")

                col1, col2 = st.columns(2)
                col1.metric("Total Candidates", total_candidates)
                col2.metric(f"Avg. Distance from {input_name}", f"{average_knn_distance_total:.2f}")

                col3, col4 = st.columns(2)
                col3.metric(f"Top 10 Avg. Distance from {input_name}", f"{average_knn_distance:.2f}")
                col4.metric(f"Top 10 Avg. Distance vs all players avg. Distance", round(average_top_10_vs_total*100,2),"%")

                # Logistic Regression Performance metrics
                # Calculate metrics
                accuracy = sum(accuracies)/len(accuracies)
                precision = sum(precisions)/len(precisions)
                recall = sum(recalls)/len(recalls)
                f1 = sum(f1_scores)/len(f1_scores)
                roc_auc = sum(roc_aucs)/len(roc_aucs)
                conf_matrix = cm

                # Display metrics
                st.subheader("📊 Logistic Regression Performance Metrics")

                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Accuracy:** {accuracy:.2f}")
                col2.markdown(f"**Precision:** {precision:.2f}")
                col3.markdown(f"**Recall:** {recall:.2f}")

                col4, col5 = st.columns(2)
                col4.markdown(f"**F1 Score:** {f1:.2f}")
                col5.markdown(f"**ROC AUC:** {roc_auc:.2f}")


                # Confusion Matrix
                st.subheader("🧮 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")
                st.pyplot(fig_cm)

                # Visualizing the new team's XI
                best_candidate_player = similar_players.iloc[0]
                similar_players.iloc[0]['field_sub_position'] = input_player_position


                def new_11_team_1(df_team_stats, input_player_position, input_player_team, best_candidate_player):
                    # Define min & max limits for positions
                    min_positions = {
                        'Goalkeeper': 1, 'Right-Back': 1, 'Centre-Back': 2, 'Left-Back': 1,
                        'Central Midfield': 1, 'Left Winger': 1,
                        'Right Winger': 1, 'Centre-Forward': 1
                    }

                    max_limits = {
                        'Left Winger': 1, 'Right Winger': 1,
                        'Left-Back': 1, 'Right-Back': 1, 'Central Midfield': 2,
                        'Left Midfield': 1, 'Right Midfield': 1, 'Centre-Forward': 1
                    }

                    # Filter the players from the same team and season
                    df_team = df_team_stats[
                        (df_team_stats['club_name'] == input_player_team) & 
                        (df_team_stats['season'] == '2024/2025')
                    ]

                    df_team_sorted = df_team.sort_values(
                        by=['total_matches', 'minutes_played'], 
                        ascending=[False, False]
                    )

                    positions = df_team_sorted['field_sub_position'].unique()

                    # If no Right Winger but there is a Left Winger, convert one
                    if 'Right Winger' not in positions and 'Left Winger' in positions:
                        # Promote the top-performing Left Winger to Right Winger
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Left Winger'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Right Winger'

                    # Similarly, if no Left Winger but a Right Winger exists
                    elif 'Left Winger' not in positions and 'Right Winger' in positions:
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Right Winger'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Left Winger'

                    # If no left back but there is a right back, convert one
                    if 'Left-Back' not in positions and 'Right-Back' in positions:
                        # Promote the top-performing Right Back to Left Back
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Right-Back'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Left-Back'
                    
                    # If no right back but there is a left back, convert one
                    elif 'Right-Back' not in positions and 'Left-Back' in positions:
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Left-Back'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Right-Back'
                                       

                    order = [
                        "Goalkeeper", "Right-Back", "Centre-Back", "Centre-Back", "Left-Back",
                        "Defensive Midfield", "Central Midfield", "Right Winger",
                        "Centre-Forward", "Left Winger"
                    ]

                    # ✅ Ensure the selected position is included
                    if input_player_position not in min_positions:
                        min_positions[input_player_position] = 1
                    if input_player_position not in order:
                        order.append(input_player_position)

                   

                    starting_xi = []
                    already_selected_players = set()
                    position_counts = {}

                    # Step 1: Fill required positions
                    for pos, min_count in min_positions.items():
                        pos_players = df_team_sorted[df_team_sorted['field_sub_position'] == pos]
                        selected_count = 0
                        for _, player in pos_players.iterrows():
                            if player['player_id'] not in already_selected_players and selected_count < min_count:
                                starting_xi.append(player.to_dict())
                                already_selected_players.add(player['player_id'])
                                selected_count += 1
                                position_counts[pos] = position_counts.get(pos, 0) + 1

                    # Step 2: Complete squad up to 11 players
                    remaining_players = df_team_sorted[~df_team_sorted['player_id'].isin(already_selected_players)]

                    while len(starting_xi) < 11 and not remaining_players.empty:
                        player = remaining_players.iloc[0].to_dict()
                        pos = player['field_sub_position']

                        # Respect max limits
                        if pos in max_limits and position_counts.get(pos, 0) >= max_limits[pos]:
                            remaining_players = remaining_players.iloc[1:]
                            continue

                        starting_xi.append(player)
                        already_selected_players.add(player['player_id'])
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                        remaining_players = remaining_players[~remaining_players['player_id'].isin(already_selected_players)]

                    # Step 3: Replace selected player with the best candidate
                    # Guarantee that the player to replaced was the one selected
                    
                    # Add the best candidate player
                    for i, player in enumerate(starting_xi):
                        if (player.get('field_sub_position') == input_player_position):
                            st.subheader(f"\n🔄 Replacing {input_name} with {best_candidate_player['player_name']}.")
                            starting_xi[i] = best_candidate_player.to_dict()
                            break

                    # Step 4: Sort and show team
                    sorted_xi = sorted(
                        starting_xi,
                        key=lambda p: order.index(p['field_sub_position']) if p['field_sub_position'] in order else float('inf')
                    )

              
                    # Ensure that the best candidate is in the correct position
                    for i, player in enumerate(sorted_xi):
                        if player['player_id'] == best_candidate_player['player_id']:
                            sorted_xi[i]['field_sub_position'] = input_player_position
                            break
                    
                    # for player in sorted_xi:
                        #st.write(f"{player['player_name']} ({player['field_sub_position']})")

                    return sorted_xi


                # Lineup 4-4-2
                def new_11_team_2(df_team_stats, input_player_position, input_player_team, best_candidate_player):
                    # Define min & max limits for positions
                    min_positions = {
                        'Goalkeeper': 1, 'Right-Back': 1, 'Centre-Back': 2, 'Left-Back': 1,
                        'Defensive Midfield':1,'Central Midfield': 1, 'Centre-Forward': 2, 'Left Winger': 1,
                        'Right Winger': 1
                    }

                    max_limits = {
                        'Left Winger': 1, 'Right Winger': 1,
                        'Left-Back': 1, 'Right-Back': 1, 'Central Midfield': 2, 'Defensive Midfield':2,
                        'Left Midfield': 1, 'Right Midfield': 1, 'Centre-Forward': 2,
                    }

                    
                    # Filter the players from the same team and season
                    df_team = df_team_stats[
                        (df_team_stats['club_name'] == input_player_team) & 
                        (df_team_stats['season'] == '2024/2025')
                    ]

                    df_team_sorted = df_team.sort_values(
                        by=['total_matches', 'minutes_played'], 
                        ascending=[False, False]
                    )

                    positions = df_team_sorted['field_sub_position'].unique()
                    # If no Right Winger but there is a Left Winger, convert one
                    if 'Right Winger' not in positions and 'Left Winger' in positions:
                        # Promote the top-performing Left Winger to Right Winger
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Left Winger'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Right Winger'

                    # Similarly, if no Left Winger but a Right Winger exists
                    elif 'Left Winger' not in positions and 'Right Winger' in positions:
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Right Winger'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Left Winger'

                    # If no left back but there is a right back, convert one
                    if 'Left-Back' not in positions and 'Right-Back' in positions:
                        # Promote the top-performing Right Back to Left Back
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Right-Back'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Left-Back'
                    
                    # If no right back but there is a left back, convert one
                    elif 'Right-Back' not in positions and 'Left-Back' in positions:
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Left-Back'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Right-Back'
                    
                    

                    order = [
                        "Goalkeeper", "Right-Back", "Centre-Back", "Centre-Back", "Left-Back",
                        "Defensive Midfield", "Central Midfield", "Right Winger",
                        "Centre-Forward", "Left Winger"
                    ]

                    # ✅ Ensure the selected position is included
                    if input_player_position not in min_positions:
                        min_positions[input_player_position] = 1
                    if input_player_position not in order:
                        order.append(input_player_position)

                    # Filter the players from the same team and season
                    df_team = df_team_stats[
                        (df_team_stats['club_name'] == input_player_team) & 
                        (df_team_stats['season'] == '2024/2025')
                    ]

                    df_team_sorted = df_team.sort_values(
                        by=['total_matches', 'minutes_played'], 
                        ascending=[False, False]
                    )

                    starting_xi = []
                    already_selected_players = set()
                    position_counts = {}

                    # Step 1: Fill required positions
                    for pos, min_count in min_positions.items():
                        pos_players = df_team_sorted[df_team_sorted['field_sub_position'] == pos]
                        selected_count = 0
                        for _, player in pos_players.iterrows():
                            if player['player_id'] not in already_selected_players and selected_count < min_count:
                                starting_xi.append(player.to_dict())
                                already_selected_players.add(player['player_id'])
                                selected_count += 1
                                position_counts[pos] = position_counts.get(pos, 0) + 1

                    # Step 2: Complete squad up to 11 players
                    remaining_players = df_team_sorted[~df_team_sorted['player_id'].isin(already_selected_players)]

                    while len(starting_xi) < 11 and not remaining_players.empty:
                        player = remaining_players.iloc[0].to_dict()
                        pos = player['field_sub_position']

                        # Respect max limits
                        if pos in max_limits and position_counts.get(pos, 0) >= max_limits[pos]:
                            remaining_players = remaining_players.iloc[1:]
                            continue

                        starting_xi.append(player)
                        already_selected_players.add(player['player_id'])
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                        remaining_players = remaining_players[~remaining_players['player_id'].isin(already_selected_players)]

                    # Step 3: Replace selected player with the best candidate
                    # Guarantee that the player to replaced was the one selected
                    
                    # Add the best candidate player
                    for i, player in enumerate(starting_xi):
                        if (player.get('field_sub_position') == input_player_position):
                            
                            starting_xi[i] = best_candidate_player.to_dict()
                            break

                    # Step 4: Sort and show team
                    sorted_xi = sorted(
                        starting_xi,
                        key=lambda p: order.index(p['field_sub_position']) if p['field_sub_position'] in order else float('inf')
                    )

                  
                    # Ensure that the best candidate is in the correct position
                    for i, player in enumerate(sorted_xi):
                        if player['player_id'] == best_candidate_player['player_id']:
                            sorted_xi[i]['field_sub_position'] = input_player_position
                            break
                    
                    # for player in sorted_xi:
                        #st.write(f"{player['player_name']} ({player['field_sub_position']})")

                    return sorted_xi
                
                #Line up 4-3-3
                def new_11_team_3(df_team_stats, input_player_position, input_player_team, best_candidate_player):
                    # Define min & max limits for positions
                    min_positions = {
                        'Goalkeeper': 1, 'Right-Back': 1, 'Centre-Back': 2, 'Left-Back': 1,
                        'Central Midfield': 1, 'Centre-Forward': 1, 'Left Winger': 1, 'Right Winger': 1,'Attacking Midfield': 2
                    }

                    max_limits = {
                        'Left Winger': 1, 'Right Winger': 1,
                        'Left-Back': 1, 'Right-Back': 1,
                        'Centre-Back': 2, 'Left-Back': 1,
                        'Central Midfield': 1, 'Attacking Midfield': 2,'Centre-Forward': 1
                    }

                    # Filter the players from the same team and season
                    df_team = df_team_stats[
                        (df_team_stats['club_name'] == input_player_team) & 
                        (df_team_stats['season'] == '2024/2025')
                    ]

                    df_team_sorted = df_team.sort_values(
                        by=['total_matches', 'minutes_played'], 
                        ascending=[False, False]
                    )

                    positions = df_team_sorted['field_sub_position'].unique()
                    if 'Right Winger' not in positions and 'Left Winger' in positions:
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Left Winger'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Right Winger'

                    # Similarly, if no Left Winger but a Right Winger exists
                    elif 'Left Winger' not in positions and 'Right Winger' in positions:
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Right Winger'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Left Winger'
                        
                    # If no left back but there is a right back, convert one
                    if 'Left-Back' not in positions and 'Right-Back' in positions:
                        # Promote the top-performing Right Back to Left Back
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Right-Back'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Left-Back'
                    
                    # If no right back but there is a left back, convert one
                    elif 'Right-Back' not in positions and 'Left-Back' in positions:
                        idx = df_team_sorted[df_team_sorted['field_sub_position']=='Left-Back'].index[0]
                        df_team_sorted.at[idx, 'field_sub_position'] = 'Right-Back'

                    

                    order = [
                        "Goalkeeper", "Right-Back", "Centre-Back", "Centre-Back", "Left-Back",
                        "Defensive Midfield", "Central Midfield", "Attacking Midfield", "Right Winger",
                        "Centre-Forward", "Left Winger"
                    ]

                    # ✅ Ensure the selected position is included
                    if input_player_position not in min_positions:
                        min_positions[input_player_position] = 1
                    if input_player_position not in order:
                        order.append(input_player_position)

                    

                    starting_xi = []
                    already_selected_players = set()
                    position_counts = {}

                    # Step 1: Fill required positions
                    for pos, min_count in min_positions.items():
                        pos_players = df_team_sorted[df_team_sorted['field_sub_position'] == pos]
                        selected_count = 0
                        for _, player in pos_players.iterrows():
                            if player['player_id'] not in already_selected_players and selected_count < min_count:
                                starting_xi.append(player.to_dict())
                                already_selected_players.add(player['player_id'])
                                selected_count += 1
                                position_counts[pos] = position_counts.get(pos, 0) + 1

                    # Step 2: Complete squad up to 11 players
                    remaining_players = df_team_sorted[~df_team_sorted['player_id'].isin(already_selected_players)]

                    while len(starting_xi) < 11 and not remaining_players.empty:
                        player = remaining_players.iloc[0].to_dict()
                        pos = player['field_sub_position']

                        # Respect max limits
                        if pos in max_limits and position_counts.get(pos, 0) >= max_limits[pos]:
                            remaining_players = remaining_players.iloc[1:]
                            continue

                        starting_xi.append(player)
                        already_selected_players.add(player['player_id'])
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                        remaining_players = remaining_players[~remaining_players['player_id'].isin(already_selected_players)]

                    # Step 3: Replace selected player with the best candidate
                    # Guarantee that the player to replaced was the one selected
                    
                    # Add the best candidate player
                    for i, player in enumerate(starting_xi):
                        if (player.get('field_sub_position') == input_player_position):
                            
                            starting_xi[i] = best_candidate_player.to_dict()
                            break

                    # Step 4: Sort and show team
                    sorted_xi = sorted(
                        starting_xi,
                        key=lambda p: order.index(p['field_sub_position']) if p['field_sub_position'] in order else float('inf')
                    )

                    
                    # Ensure that the best candidate is in the correct position
                    for i, player in enumerate(sorted_xi):
                        if player['player_id'] == best_candidate_player['player_id']:
                            sorted_xi[i]['field_sub_position'] = input_player_position
                            break
                    
                    # for player in sorted_xi:
                        #st.write(f"{player['player_name']} ({player['field_sub_position']})")

                    return sorted_xi
                
                st.subheader("✅ Optimized Starting XI:")
                starting_xi_1 = new_11_team_1(df, input_player_position, input_player_team, best_candidate_player)
                                
                # Base pitch
                pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='black')
                fig, ax = pitch.draw(figsize=(10, 7))

                # Predefined coordinates for common positions (base points)
                if 'Defensive Midfield' in [p['field_sub_position'] for p in starting_xi_1]:
                    base_positions = {
                        'Goalkeeper': [(5, 40)],
                        'Right-Back': [(20, 70)],
                        'Left-Back': [(20, 10)],
                        'Centre-Back': [(20, 30), (20, 50)],
                        'Defensive Midfield': [(50, 30)],
                        'Central Midfield': [(50, 50)],          
                        'Attacking Midfield': [(75, 40)],
                        'Left Midfield': [(75, 20)],
                        'Right Midfield': [(75, 60)],
                        'Left Winger': [(75, 10)],
                        'Right Winger': [(75, 70)],
                        'Centre-Forward': [(100, 40)],
                        'Second Striker': [(100, 40)]
                        }
                else:
                    base_positions = {
                        'Goalkeeper': [(5, 40)],
                        'Right-Back': [(20, 70)],
                        'Left-Back': [(20, 10)],
                        'Centre-Back': [(20, 30), (20, 50)],
                        'Central Midfield': [(50,30),(50, 50)],          
                        'Attacking Midfield': [(75, 40)],
                        'Left Midfield': [(75, 20)],
                        'Right Midfield': [(75, 60)],
                        'Left Winger': [(75, 10)],
                        'Right Winger': [(75, 70)],
                        'Centre-Forward': [(100, 40)],
                        'Second Striker': [(100, 40)]
              }

                # Counting players by position
                position_counts = Counter([p['field_sub_position'] for p in starting_xi_1])
                position_counter = {}
                final_positions = {}

                for player in starting_xi_1:
                    position = player['field_sub_position']
                    player_name = player['player_name']
                    
                    coords = base_positions.get(position, [(60, 40)])
                    count = position_counts[position]

                    # If there are 3 players and only 2 base coords → interpolate a third one in the middle
                    if len(coords) == 2 and count == 3:
                        x0, y0 = coords[0]
                        x1, y1 = coords[1]
                        interpolated_coords = [
                            (x0, y0),
                            ((x0 + x1) / 2, (y0 + y1) / 2),
                            (x1, y1)
                        ]
                        idx = position_counter.get(position, 0)
                        idx = min(idx, 2)
                        final_positions[player_name] = interpolated_coords[idx]
                        position_counter[position] = idx + 1

                    # If there are 4 players and only 2 base coords → interpolate 2 in the middle and place 2 at the ends

                    
                    if len(coords) == 2 and count == 4:
                        x0, y0 = coords[0]
                        x1, y1 = coords[1]
                        interpolated_coords = [
                            (x0, y0),
                            (x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3),
                            (x0 + (x1 - x0) * 2 / 3, y0 + (y1 - y0) * 2 / 3),
                            (x1, y1)
                        ]
                        idx = position_counter.get(position, 0)
                        idx = min(idx, 3)
                        final_positions[player_name] = interpolated_coords[idx]
                        position_counter[position] = idx + 1


                    # If there is 1 player and 2 coords → use the average
                    elif len(coords) == 2 and count == 1:
                        x = (coords[0][0] + coords[1][0]) / 2
                        y = (coords[0][1] + coords[1][1]) / 2
                        final_positions[player_name] = (x, y)

                    # Standard distribution
                    else:
                        idx = position_counter.get(position, 0)
                        idx = min(idx, len(coords) - 1)
                        final_positions[player_name] = coords[idx]
                        position_counter[position] = idx + 1

                # Plot players
                for player in starting_xi_1:
                    player_name = player['player_name']
                    x, y = final_positions[player_name]

                    pitch.scatter(x, y, ax=ax, color='blue', s=100, zorder=2)
                    ax.text(
                        x, y - 3,
                        player_name,
                        fontsize=9,
                        ha='center',
                        va='center',
                        color='white',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'),
                        zorder=3
                    )

                plt.title("New Starting XI- Alternative 1", fontsize=14)
                st.pyplot(fig)

                starting_xi_2 = new_11_team_2(df, input_player_position, input_player_team, best_candidate_player)
                               
                # Base pitch
                pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='black')
                fig2, ax = pitch.draw(figsize=(10, 7))

                # Predefined coordinates for common positions (base points)
                if 'Defensive Midfield' in [p['field_sub_position'] for p in starting_xi_2]:
                    base_positions = {
                        'Goalkeeper': [(5, 40)],
                        'Right-Back': [(20, 70)],
                        'Left-Back': [(20, 10)],
                        'Centre-Back': [(20, 30), (20, 50)],
                        'Defensive Midfield': [(50, 30)],
                        'Central Midfield': [(50, 50)],          
                        'Attacking Midfield': [(75, 10)],
                        'Attacking Midfield': [(75, 70)],
                        'Left Midfield': [(75, 10)],
                        'Right Midfield': [(75, 70)],
                        'Left Winger': [(75, 10)],
                        'Right Winger': [(75, 70)],
                        'Centre-Forward': [(100, 30), (100, 50)],
                        'Second Striker': [(100, 30), (100, 50)]
                    }
                else:
                    base_positions = {
                        'Goalkeeper': [(5, 40)],
                        'Right-Back': [(20, 70)],
                        'Left-Back': [(20, 10)],
                        'Centre-Back': [(20, 30), (20, 50)],
                        'Central Midfield': [(50, 30), (50, 50)],          
                        'Attacking Midfield': [(75, 10)],
                        'Attacking Midfield': [(75, 70)],
                        'Left Midfield': [(75, 10)],
                        'Right Midfield': [(75, 70)],
                        'Left Winger': [(75, 10)],
                        'Right Winger': [(75, 70)],
                        'Centre-Forward': [(100, 30), (100, 50)],
                        'Second Striker': [(100, 30), (100, 50)]
                    }

                positions = base_positions.copy()

                
                # Contador de jugadores por posición
                position_counts = Counter([p['field_sub_position'] for p in starting_xi_2])
                position_counter = {}
                final_positions = {}

                for player in starting_xi_2:
                    position = player['field_sub_position']
                    player_name = player['player_name']
                    
                    coords = base_positions.get(position, [(60, 40)])
                    count = position_counts[position]

                    # If there are 3 players and only 2 base coords → interpolate a third one in the middle
                    if len(coords) == 2 and count == 3:
                        x0, y0 = coords[0]
                        x1, y1 = coords[1]
                        interpolated_coords = [
                            (x0, y0),
                            ((x0 + x1) / 2, (y0 + y1) / 2),
                            (x1, y1)
                        ]
                        idx = position_counter.get(position, 0)
                        idx = min(idx, 2)
                        final_positions[player_name] = interpolated_coords[idx]
                        position_counter[position] = idx + 1

                    # If there are 4 players and only 2 base coords → interpolate 2 in the middle and place 2 at the ends

                    
                    if len(coords) == 2 and count == 4:
                        x0, y0 = coords[0]
                        x1, y1 = coords[1]
                        interpolated_coords = [
                            (x0, y0),
                            (x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3),
                            (x0 + (x1 - x0) * 2 / 3, y0 + (y1 - y0) * 2 / 3),
                            (x1, y1)
                        ]
                        idx = position_counter.get(position, 0)
                        idx = min(idx, 3)
                        final_positions[player_name] = interpolated_coords[idx]
                        position_counter[position] = idx + 1


                    # If there is 1 player and 2 coords → use the average
                    elif len(coords) == 2 and count == 1:
                        x = (coords[0][0] + coords[1][0]) / 2
                        y = (coords[0][1] + coords[1][1]) / 2
                        final_positions[player_name] = (x, y)

                    # Standard distribution
                    else:
                        idx = position_counter.get(position, 0)
                        idx = min(idx, len(coords) - 1)
                        final_positions[player_name] = coords[idx]
                        position_counter[position] = idx + 1

                # Plot players
                for player in starting_xi_2:
                    player_name = player['player_name']
                    x, y = final_positions[player_name]

                    pitch.scatter(x, y, ax=ax, color='blue', s=100, zorder=2)
                    ax.text(
                        x, y - 3,
                        player_name,
                        fontsize=9,
                        ha='center',
                        va='center',
                        color='white',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'),
                        zorder=3
                    )

                plt.title("New Starting XI- Alternative 2", fontsize=14)
                st.pyplot(fig2)

                
                starting_xi_3 = new_11_team_3(df, input_player_position, input_player_team, best_candidate_player)
                # Base pitch
                pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='black')
                fig3, ax = pitch.draw(figsize=(10, 7))

                # Predefined coordinates for common positions (base points)
                if 'Defensive Midfield' in [p['field_sub_position'] for p in starting_xi_3]:
                    base_positions = {
                        'Goalkeeper': [(5, 40)],
                        'Right-Back': [(20, 70)],
                        'Left-Back': [(20, 10)],
                        'Centre-Back': [(20, 30), (20, 50)],
                        'Central Midfield': [(60,40)],
                        'Attacking Midfield': [(70, 20), (70, 60)],
                        
                        'Left Midfield': [(70, 20)],
                        'Right Midfield': [(70, 60)],
                        'Left Winger': [(100, 10)],
                        'Right Winger': [(100, 70)],
                        'Centre-Forward': [(100, 40)],
                        'Second Striker': [(100, 40)]
                        
                    }
                else:
                    base_positions = {
                        'Goalkeeper': [(5, 40)],
                        'Right-Back': [(20, 70)],
                        'Left-Back': [(20, 10)],
                        'Centre-Back': [(20, 30), (20, 50)],
                        'Central Midfield': [(60,40)],
                        'Attacking Midfield': [(70, 20), (70, 60)],
                        'Left Midfield': [(70, 20)],
                        'Right Midfield': [(70, 60)],
                        'Left Winger': [(100, 10)],
                        'Right Winger': [(100, 70)],
                        'Centre-Forward': [(100, 40)],
                        'Second Striker': [(100, 40)]
                    }

                # Contador de jugadores por posición
                position_counts = Counter([p['field_sub_position'] for p in starting_xi_3])
                position_counter = {}
                final_positions = {}

                for player in starting_xi_3:
                    position = player['field_sub_position']
                    player_name = player['player_name']
                    
                    coords = base_positions.get(position, [(60, 40)])
                    count = position_counts[position]

                    # If there are 3 players and only 2 base coords → interpolate a third one in the middle
                    if len(coords) == 2 and count == 3:
                        x0, y0 = coords[0]
                        x1, y1 = coords[1]
                        interpolated_coords = [
                            (x0, y0),
                            ((x0 + x1) / 2, (y0 + y1) / 2),
                            (x1, y1)
                        ]
                        idx = position_counter.get(position, 0)
                        idx = min(idx, 2)
                        final_positions[player_name] = interpolated_coords[idx]
                        position_counter[position] = idx + 1

                    # If there are 4 players and only 2 base coords → interpolate 2 in the middle and place 2 at the ends

                    
                    if len(coords) == 2 and count == 4:
                        x0, y0 = coords[0]
                        x1, y1 = coords[1]
                        interpolated_coords = [
                            (x0, y0),
                            (x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3),
                            (x0 + (x1 - x0) * 2 / 3, y0 + (y1 - y0) * 2 / 3),
                            (x1, y1)
                        ]
                        idx = position_counter.get(position, 0)
                        idx = min(idx, 3)
                        final_positions[player_name] = interpolated_coords[idx]
                        position_counter[position] = idx + 1


                    # If there is 1 player and 2 coords → use the average
                    elif len(coords) == 2 and count == 1:
                        x = (coords[0][0] + coords[1][0]) / 2
                        y = (coords[0][1] + coords[1][1]) / 2
                        final_positions[player_name] = (x, y)

                    # Standard distribution
                    else:
                        idx = position_counter.get(position, 0)
                        idx = min(idx, len(coords) - 1)
                        final_positions[player_name] = coords[idx]
                        position_counter[position] = idx + 1

                # Plot players
                for player in starting_xi_3:
                    player_name = player['player_name']
                    x, y = final_positions[player_name]

                    pitch.scatter(x, y, ax=ax, color='blue', s=100, zorder=2)
                    ax.text(
                        x, y - 3,
                        player_name,
                        fontsize=9,
                        ha='center',
                        va='center',
                        color='white',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'),
                        zorder=3
                    )

                plt.title("New Starting XI- Alternative 3", fontsize=14)
                st.pyplot(fig3)
         

                
                # Streamlit button to restart or change parameters
                col1 = st.columns
                if st.button("🔄 Change Filters / Parameters"):
                    st.session_state['step'] = 'filter_parameters'  
                    st.rerun()

        # cd "C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos"
        #  streamlit run Recommendation_algorithm.py
        #
