import mplsoccer
from mplsoccer import Pitch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import streamlit as st
from fuzzywuzzy import process, fuzz
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Cargar los datos
df = pd.read_csv("CompleteDataset.csv")
df_transfers = pd.read_csv("Results2.csv")
df = df.dropna()
df_transfers = df_transfers.dropna()
season_of_interest = '2024/2025'

# Limpiar coordenadas
df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '.').astype(float)
df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '.').astype(float)
df_transfers['Latitude'] = df_transfers['Latitude'].astype(str).str.replace(',', '.').astype(float)
df_transfers['Longitude'] = df_transfers['Longitude'].astype(str).str.replace(',', '.').astype(float)

# Diccionario de features por posición
position_features = {
            "Goalkeeper": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "gk_diving_score", "gk_handling_score", "gk_kicking_score", "gk_positioning_score",
                "gk_reflexes_score", "jumping_score", "reactions_score", "strength_score", "stamina_score", "minutes_played",
                "tot_clean_sheets", "total_missed_matches_inj", "yellow_cards", "red_cards", "conceeded_goals"
            ],
            "Centre-Back": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "defense_standing_tackle_|score",
                "defense_sliding_tackles_score", "interceptions_score", "heading_accuracy_score", "jumping_score", "strength_score",
                "stamina_score", "balance_score", "reactions_score", "short_passing_score", "passing_score", "shot_power_score",
                "composure_score", "yellow_cards", "red_cards", "conceeded_goals", "minutes_played", "total_missed_matches_inj"
            ],
            "Left-Back": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "defense_standing_tackle_score",
                "defense_sliding_tackles_score", "interceptions_score", "heading_accuracy_score", "jumping_score", "strength_score",
                "stamina_score", "balance_score", "reactions_score", "short_passing_score", "passing_score", "acceleration_score",
                "sprint_score", "crossing_score", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Right-Back": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "defense_standing_tackle_score",
                "defense_sliding_tackles_score", "interceptions_score", "heading_accuracy_score", "jumping_score", "strength_score",
                "stamina_score", "balance_score", "reactions_score", "short_passing_score", "passing_score", "acceleration_score",
                "sprint_score", "crossing_score", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Defensive Midfield": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "defense_score", "defense_awareness_score", "interceptions_score",
                "short_passing_score", "passing_score", "long_passing_score", "vision_score", "stamina_score", "balance_score",
                "strength_score", "reactions_score", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Central Midfield": [
            "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "short_passing_score", "passing_score", "long_passing_score", "vision_score",
                "dribbling_score", "ball_control_score", "composure_score", "stamina_score", "reactions_score", "yellow_cards",
                "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Attacking Midfield": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "vision_score", "short_passing_score", "passing_score", "long_passing_score",
                "dribbling_score", "ball_control_score", "composure_score", "agility_score", "finishing_score", "shot_power_score",
                "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Left Winger": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "passing_score", "finishing_score", "shot_power_score", "ball_control_score", "stamina_score",
                "vision_score", "goals", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Right Winger": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "passing_score", "finishing_score", "shot_power_score", "ball_control_score", "stamina_score",
                "vision_score", "goals", "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Left Midfield": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "short_passing_score", "passing_score", "long_passing_score", "stamina_score", "vision_score",
                "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Right Midfield": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "acceleration_score", "sprint_score", "dribbling_score", "agility_score",
                "crossing_score", "short_passing_score", "passing_score", "long_passing_score", "stamina_score", "vision_score",
                "assists", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Centre-Forward": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "goals", "finishing_score", "shot_power_score", "heading_accuracy_score",
                "ball_control_score", "agility_score", "acceleration_score", "sprint_score", "balance_score", "dribbling_score",
                "jumping_score", "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
            ],
            "Second Striker": [
                "season", "player_id", "Latitude", "Longitude", "age", "height_cm", "weight", "foot_id", "market_value", "is_top5_league_team",
                "total_matches", "FIFA_score", "goals", "finishing_score", "shot_power_score", "vision_score", "dribbling_score",
                "agility_score", "acceleration_score", "sprint_score", "balance_score", "ball_control_score", "assists",
                "yellow_cards", "red_cards", "minutes_played", "total_missed_matches_inj"
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
            "field_sub_position": matched_player["field_sub_position"]
        })
    return pd.DataFrame(similar_players)

# Streamlit UI
st.title("Football Player Replacement Recommendation")

# Restart full flow
if st.button("🔁 Restart Full Search"):
    keys_to_clear = [
        'player_input', 'player_confirmed', 'selected_id', 
        'selected_player', 'input_player_team', 'input_player_position'
    ]
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
                    "Select the correct player ID:",
                    options=candidates_df["player_id"].tolist(),
                    format_func=lambda x: f"{candidates_df.loc[candidates_df['player_id'] == x, 'player_name'].values[0]} ({x})"
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
            input_player_team = selected_player["club_name"]
            input_player_position = selected_player["field_sub_position"]
            st.write(f"Selected: {selected_player['player_name']} - {input_player_position} - {selected_player['player_id']}")


            with st.form("player_filter_form"):
                st.subheader("Transfer filters")
                min_age, max_age = st.slider("Select age range", 16, 40, (20, 30))
                min_val, max_val = st.slider("Select market value range (millions €)", 0, 200, (0, 40))
                continue_button1 = st.form_submit_button("Find Replacements")

            if continue_button1:
                df_2024_2025 = df[df["season"] == "2024/2025"]
                
                df_filtered = df_2024_2025[
                (df_2024_2025["player_id"] == st.session_state.selected_id) |
                (
                    (df_2024_2025["field_sub_position"] == input_player_position) &
                    (df_2024_2025["age"] >= min_age) &
                    (df_2024_2025["age"] <= max_age) &
                    (df_2024_2025["market_value"] >= min_val * 1_000_000) &
                    (df_2024_2025["market_value"] <= max_val * 1_000_000)
                )]
                position_feature_set = position_features[input_player_position]
                df_filtered = df_filtered[position_feature_set]
                #st.write(selected_player)
                #st.write(df_filtered)
                #st.write(df_filtered.shape)

                # Filter complete dataset including the id players taken in the df_filtered
                # The first position of df_filter_complete is the player we are going to replace
                df_filter_complete = pd.DataFrame(columns=position_feature_set)
                df_filter_complete= df[df["player_id"] == st.session_state.selected_id]
                # Add the filtered players to the complete dataset
                df_filter_complete = df[df["player_id"].isin(df_filtered["player_id"])]
            
                
                #st.write(df_filter_complete)
                #st.write(df_filter_complete.shape)

                def compute_weighted_features(player_df, player_id):
                    player_data = player_df[player_df["player_id"] == player_id]
                    player_data["exp_weight"] = np.exp(-0.1 * (int(season_of_interest[:4]) - player_data["season"].str[:4].astype(int)))
                    player_data["exp_weight"] /= player_data["exp_weight"].sum()

                    # Dont use the season column or the market value in the weighted average
                    features_to_use = [col for col in position_feature_set if col not in ("season", "market_value")]
                    weighted = (player_data[features_to_use].T * player_data["exp_weight"]).T.sum()
                    weighted["player_id"] = player_id
                    return weighted
                
                #st.write("Computing weighted features...")
                # Compute the weighted features for each player
                #print type of df_filter_complete
                #st.write(df_filter_complete.dtypes)
            
                # Create a new DataFrame with the averaged features per player
            
                def averaged_features_player(df_filter_complete):
                    
                    records = []
                    for pid in df_filter_complete["player_id"].unique():
                        records.append(compute_weighted_features(df, pid))
                    df_weighted = pd.DataFrame(records)
                            
                    return df_weighted
                
                df_weighted = averaged_features_player(df_filter_complete)
                #st.write(df_weighted)
                #st.write(df_weighted.shape)

                #st.write(df_weighted.dtypes)

                # Perform the KNN search

                input_vector = df_weighted[df_weighted["player_id"] == st.session_state.selected_id]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_weighted.drop(columns=["player_id"]))
                input_scaled = scaler.transform(input_vector.drop(columns=["player_id"]))

                knn = NearestNeighbors(n_neighbors=11)
                knn.fit(X_scaled)
                distances, indices = knn.kneighbors(input_scaled)
                similar_players = df_weighted.iloc[indices[0]].copy()
                similar_players["distance"] = distances[0]
                similar_players = similar_players[similar_players["player_id"] != st.session_state.selected_id]
                #st.write(similar_players)

                season_of_interest = '2024/2025'
                info = df[df['season'] == season_of_interest][['player_id', 'player_name', 'country', 'club_name','field_sub_position','market_value','season']].drop_duplicates('player_id')
                #info_cols = df[["player_id", "player_name", "club_name",  "country","field_sub_position"]].drop_duplicates("player_id")
                
               
                similar_players = similar_players.merge(info, on="player_id")
                st.write("The 10 most similar players are to:", selected_player["player_name"]," are:")
                st.subheader("Similar Players")
                # show the number of the player # in the top 10 ranking, 1 if first..
                similar_players["rank"] = range(1, len(similar_players) + 1)
                st.dataframe(similar_players[["rank","player_name", "club_name", "age", "distance"]])

                #st.write(similar_players)
                # Logistic regression model

                X=df_transfers

                # Filter the transfer dataset to include only players from the same position
                X = X[X["field_sub_position"] == input_player_position]

                # Keep only the position features
                X = df_transfers[position_feature_set]  # Features for traning
                y = df_transfers['successful_adaptation']  # Target variable
                
                # Drop the season column and the player_id column
                X = X.drop(columns=["season", "player_id"])
    
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Escalar las características (importante para la regresión logística)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)  # Ajustar y transformar los datos de entrenamiento
                X_test_scaled = scaler.transform(X_test)  # Solo transformar los datos de prueba
                
                # Train the model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train_scaled, y_train)
                
                # Merge all the features of the players in the Top 10 ranking
                
                # Make predictions
                #similar_players = similar_players.drop(columns=["season", "player_id"])
                
                X_similar = scaler.transform(similar_players[position_feature_set].drop(columns=["season", "player_id"]))
               
                # Predict the probabilities of adaptation
                probs = model.predict_proba(X_similar)[:, 1]
                similar_players["adaptation_prob"] = probs
                similar_players = similar_players.sort_values("adaptation_prob", ascending=False)
                similar_players = similar_players.reset_index(drop=True)

                st.subheader("Top 10 Recommended Replacements")
                st.dataframe(similar_players[["player_name", "club_name", "age", "adaptation_prob"]].head(10))
                st.subheader("Recommended Replacement")
                st.write("The player with the highest probability of adaptation is to replace: ", selected_player["player_name"]," is:")
                st.write(similar_players[["player_name", "club_name", "age", "adaptation_prob"]].head(1))

                # Visualizing the new team's XI
                best_candidate_player = similar_players.iloc[0]
                def new_11_team(df_team_stats, input_player_position, input_player_team, best_candidate_player):
                    # Define min & max limits for positions
                    min_positions = {
                        'Goalkeeper': 1, 'Right-Back': 1, 'Centre-Back': 2, 'Left-Back': 1,
                        'Central Midfield': 1, 'Left Winger': 1,
                        'Right Winger': 1, 'Centre-Forward': 1
                    }

                    max_limits = {
                        'Left Winger': 1, 'Right Winger': 1,
                        'Left-Back': 1, 'Right-Back': 1,
                        'Left Midfield': 1, 'Right Midfield': 1
                    }

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
                    for i, player in enumerate(starting_xi):
                        if player.get('field_sub_position') == input_player_position:
                            st.write(f"\n🔄 Replacing {player['player_name']} with {best_candidate_player['player_name']}.")
                            starting_xi[i] = best_candidate_player.to_dict()
                            break

                    # Step 4: Sort and show team
                    sorted_xi = sorted(
                        starting_xi,
                        key=lambda p: order.index(p['field_sub_position']) if p['field_sub_position'] in order else float('inf')
                    )

                    st.subheader("✅ Optimized Starting XI")
                    for player in sorted_xi:
                        st.write(f"{player['player_name']} ({player['field_sub_position']})")

                    return sorted_xi

                # Call the function to create the new starting XI
                starting_xi = new_11_team(df, input_player_position, input_player_team, best_candidate_player)
                
                # Base pitch
                
                pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='black')
                fig, ax = pitch.draw(figsize=(10, 7))

                # Predefined coordinates for common positions (base points)
                base_positions = {
                    'Goalkeeper': [(5, 40)],
                    'Right-Back': [(20, 70)],
                    'Left-Back': [(20, 10)],
                    'Centre-Back': [(20, 30), (20, 50)],
                    'Defensive Midfield': [(40, 30), (40, 50)],
                    'Central Midfield': [(60, 30), (60, 50)],
                    'Attacking Midfield': [(75, 40)],
                    'Left Midfield': [(65, 20)],
                    'Right Midfield': [(65, 60)],
                    'Left Winger': [(80, 15)],
                    'Right Winger': [(80, 65)],
                    'Centre-Forward': [(100, 35), (100, 45)],
                    'Second Striker': [(90, 40)]
                }

                # Contador de jugadores por posición
                position_counts = Counter([p['field_sub_position'] for p in starting_xi])
                position_counter = {}
                final_positions = {}

                for player in starting_xi:
                    position = player['field_sub_position']
                    player_name = player['player_name']
                    
                    coords = base_positions.get(position, [(60, 40)])
                    count = position_counts[position]

                    # Si hay 3 jugadores y solo 2 coords base → interpolamos una tercera en el medio
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

                    # Si hay 4 jugadores y solo 2 coords base → interpolamos 2 coords en el medio y los 2 otros de extremos

                    
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


                    # Si hay 1 jugador y 2 coords → usar promedio
                    elif len(coords) == 2 and count == 1:
                        x = (coords[0][0] + coords[1][0]) / 2
                        y = (coords[0][1] + coords[1][1]) / 2
                        final_positions[player_name] = (x, y)

                    # Distribución estándar
                    else:
                        idx = position_counter.get(position, 0)
                        idx = min(idx, len(coords) - 1)
                        final_positions[player_name] = coords[idx]
                        position_counter[position] = idx + 1

                # Plot players
                for player in starting_xi:
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

                plt.title("New Starting XI", fontsize=14)
                st.pyplot(fig)


                
                # Streamlit button to restart or change parameters
                col1 = st.columns
                if st.button("🔄 Change Filters / Parameters"):
                    st.session_state['step'] = 'filter_parameters'  
                    st.rerun()

        # cd "C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos"
        #  streamlit run app3.py
        #