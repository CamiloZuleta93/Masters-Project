--- 1. Create Database
CREATE DATABASE databasefootball;
USE databasefootball;

---2. Create table
CREATE TABLE fifa_players (
    name VARCHAR(255),
    fifa_version VARCHAR(255),
    overall INT,
    pace FLOAT,
    nation VARCHAR(255),
    league_name VARCHAR(255),
    club_name VARCHAR(255),
    age FLOAT,
    shooting FLOAT,
    passing FLOAT,
    dribbling FLOAT,
    defending FLOAT,
    physic FLOAT,
    movement_acceleration FLOAT,
    movement_sprint_speed FLOAT,
    mentality_positioning FLOAT,
    attacking_finishing FLOAT,
    power_shot_power FLOAT,
    power_long_shots FLOAT,
    attacking_volleys FLOAT,
    mentality_penalties FLOAT,
    mentality_vision FLOAT,
    attacking_crossing FLOAT,
    skill_fk_accuracy FLOAT,
    attacking_short_passing FLOAT,
    skill_long_passing FLOAT,
    skill_curve FLOAT,
    skill_dribbling FLOAT,
    movement_agility FLOAT,
    movement_balance FLOAT,
    movement_reactions FLOAT,
    skill_ball_control FLOAT,
    mentality_composure FLOAT,
    mentality_interceptions FLOAT,
    attacking_heading_accuracy FLOAT,
    defending_marking_awareness FLOAT,
    defending_standing_tackle FLOAT,
    defending_sliding_tackle FLOAT,
    power_jumping FLOAT,
    power_stamina FLOAT,
    power_strength FLOAT,
    mentality_aggression FLOAT,
    club_position VARCHAR(255),
    weak_foot VARCHAR(255),
    skill_moves FLOAT,
    preferred_foot VARCHAR(255),
    height VARCHAR(10),
    weight VARCHAR(10),
    goalkeeping_diving FLOAT,
    goalkeeping_handling FLOAT,
    goalkeeping_kicking FLOAT,
    goalkeeping_positioning FLOAT,
    goalkeeping_reflexes FLOAT,
    season VARCHAR(10),
    Latitude FLOAT,
    Longitude FLOAT
);

CREATE TABLE CDI (
    Country_1 VARCHAR(255),
    Country_2 VARCHAR(255),
    CDI FLOAT
);

CREATE TABLE LANGUAGES (
    Country_1 VARCHAR(255),
    Country_2 VARCHAR(255),
    LANGUAGE_score FLOAT
);

CREATE TABLE injuries (
    season VARCHAR(10),
    injury VARCHAR(255),
    injury_from DATE,
    injury_until DATE,
    days INT,
    games_missed INT,
    player_name VARCHAR(255),
    player_id INT NOT NULL,
    url_injuries VARCHAR(500)
);

CREATE TABLE history (
    appearance_id VARCHAR(255),
    game_id INT,
    player_id INT,
    player_club_id INT,
    player_current_club_id INT,
    date DATE,
    player_name VARCHAR(255),
    competition_id VARCHAR(10),
    yellow_cards INT,
    red_cards INT,
    goals INT,
    assists INT,
    minutes_played INT,
    season  VARCHAR(10)
);

CREATE TABLE players (
    player_id INT PRIMARY KEY,
    first_name NVARCHAR(255),
    last_name NVARCHAR(255),
    name NVARCHAR(255),
    last_season INT,
    current_club_id NVARCHAR(255),
    player_code NVARCHAR(255),
    country_of_birth NVARCHAR(255),
    city_of_birth NVARCHAR(255),
    country_of_citizenship NVARCHAR(255),
    date_of_birth DATE,
    sub_position NVARCHAR(255),
    position NVARCHAR(255),
    foot NVARCHAR(100),
    height_in_cm INT,
    contract_expiration_date DATE,
    agent_name NVARCHAR(255),
    image_url NVARCHAR(255),
    url NVARCHAR(255),
    current_club_domestic_competition_id NVARCHAR(255),
    current_club_name NVARCHAR(255),
    market_value_in_eur INT,
    highest_market_value_in_eur INT,
    Latitude FLOAT,
    Longitude FLOAT
);

CREATE TABLE clubs (
    club_id INT PRIMARY KEY,
    club_code VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    domestic_competition_id VARCHAR(50),
    total_market_value VARCHAR(50),
    squad_size INT,
    average_age DECIMAL(4,1),
    foreigners_number INT,
    foreigners_percentage DECIMAL(5,2),
    national_team_players INT,
    stadium_name VARCHAR(255),
    stadium_seats INT,
    net_transfer_record VARCHAR(50),
    coach_name VARCHAR(255),
    last_season VARCHAR(50),
    filename VARCHAR(255),
    url VARCHAR(500)
);


CREATE TABLE games (
    game_id INT PRIMARY KEY,
    competition_id VARCHAR(50),
    season VARCHAR(50),
    round VARCHAR(50),
    date DATETIME,
    home_club_id INT,
    away_club_id INT,
    home_club_goals INT,
    away_club_goals INT,
    home_club_position INT,
    away_club_position INT,
    home_club_manager_name VARCHAR(255),
    away_club_manager_name VARCHAR(255),
    stadium VARCHAR(255),
    attendance INT,
    referee VARCHAR(255),
    url VARCHAR(255),
    home_club_formation VARCHAR(255),
    away_club_formation VARCHAR(255),
    home_club_name VARCHAR(255),
    away_club_name VARCHAR(255),
    aggregate DATETIME,
    competition_type VARCHAR(50)
);

CREATE TABLE competitions (
    competition_id NVARCHAR(255),
    competition_code NVARCHAR(50),
    name NVARCHAR(255),
    sub_type NVARCHAR(50),
    type NVARCHAR(50),
    country_id INT,
    country_name NVARCHAR(255),
    domestic_league_code NVARCHAR(50),
    confederation NVARCHAR(50),
    url NVARCHAR(255),
    is_major_national_league NVARCHAR(255)
);

CREATE TABLE player_valuations (
    player_id INT NOT NULL,
    date DATE NOT NULL,
    market_value_in_eur INT NOT NULL,
    current_club_id INT,
    player_club_domestic_competition_id NVARCHAR(50),
    season VARCHAR(10)
);

CREATE TABLE transfers (
    player_id INT NOT NULL,
    transfer_date DATE,
    transfer_season NVARCHAR(50),
    from_club_id INT,
    to_club_id INT,
    from_club_name NVARCHAR(255),
    to_club_name NVARCHAR(255),
    transfer_fee NVARCHAR(50),
    market_value_in_eur INT,
    player_name NVARCHAR(255),
);

CREATE TABLE lineups (
    game_lineups_id VARCHAR(255) PRIMARY KEY,
    date DATE NOT NULL,
    game_id INT NOT NULL,
    player_id INT NOT NULL,
    club_id INT NOT NULL,
    player_name VARCHAR(255),
    type VARCHAR(50),
    position VARCHAR(50),
    number VARCHAR(50),
    team_captain BIT,
    season VARCHAR(50)
);

CREATE TABLE game_events (
    game_event_id VARCHAR(255) PRIMARY KEY,
    date DATE NOT NULL,
    game_id INT NOT NULL,
    minute INT NOT NULL CHECK (minute >= 0 AND minute <= 120),
    type VARCHAR(50) NOT NULL,
    club_id INT NOT NULL,
    player_id VARCHAR(255) NOT NULL,
    description TEXT,
    player_in_id VARCHAR(255),
    player_assist_id VARCHAR(255)
);

CREATE TABLE club_games (
    game_id INT,
    club_id VARCHAR(255),
    own_goals INT,
    own_position VARCHAR(255),
    own_manager_name VARCHAR(100),
    opponent_id INT,
    opponent_goals INT,
    opponent_position VARCHAR(255),
    opponent_manager_name VARCHAR(100),
    hosting VARCHAR(255),
    is_win BIT,
    clean_sheets INT,
);
DELETE FROM players;


---3. Filling out of the tables
BULK INSERT fifa_players
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\Merged_Fifa_Data.csv'
WITH (
     FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2
);

BULK INSERT injuries
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\InjuriesAdjusted.csv'
WITH (
     FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2
);

BULK INSERT CDI
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\CDIAdjusted.csv'
WITH (
     FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2
);

BULK INSERT LANGUAGES
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\country_languages_similarity.csv'
WITH (
     FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2
);

BULK INSERT history
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\appearancesAdjusted.csv'
WITH (
     FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2
);

BULK INSERT clubs
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\clubsAdjusted.csv'
WITH (
     FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2
);

BULK INSERT players
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\playersAdjusted.csv'
WITH (
     FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',  
    FIRSTROW = 2
);

BULK INSERT games
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\gamesAdjusted.csv'
WITH (
    FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);

BULK INSERT competitions
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\competitionsAdjusted.csv'
WITH (
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);

BULK INSERT player_valuations
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\player_valuationsAdjusted.csv'
WITH (
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);

BULK INSERT transfers
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\transfersAdjusted.csv'
WITH (
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);

BULK INSERT lineups
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\game_lineupsAdjusted.csv'
WITH (
    FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);

BULK INSERT game_events
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\game_eventsAdjusted.csv'
WITH (
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);

BULK INSERT club_games
FROM 'C:\Users\camil\OneDrive\Documentos\Camilo Zuleta\Maestria EDHEC Business School\Master Project\Base de datos\club_gamesAdjusted.csv'
WITH (
    FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);

---4.Clear tables when updating
DELETE from club_games
DELETE FROM game_events;
DELETE FROM lineups;
DELETE from transfers
DELETE FROM player_valuations;
DELETE FROM competitions;
DELETE from games;
DELETE FROM players;
DELETE from clubs;
DELETE FROM history;


DELETE FROM fifa_players;




---5. Consolidated Query
WITH history_season AS (
    SELECT 
        season,
        player_name,
        player_id, 
        player_club_id,
        date,
        game_id,
        competition_id,
        yellow_cards,
        red_cards,
        goals,
        assists,
        minutes_played
    FROM history
    
),
total_minutes_team AS (
    SELECT 
        season,
        player_club_id,
        COUNT(DISTINCT game_id) * 90 as total_possible_minutes
    FROM history
    group by season, player_club_id
    
),
fifa_season_agg AS (
    SELECT
        name,
        season, 
        max(weight) as weight,
        MAX(overall) AS fifa_score,
        MAX(defending) AS defense_score,
        MAX(defending_marking_awareness) AS defense_awareness_score,
        MAX(defending_standing_tackle) AS defense_standing_tackle_score,
        MAX(defending_sliding_tackle) AS defense_sliding_tackles_score,
        MAX(physic) AS physical_condition_score,
        MAX(dribbling) AS dribbling_score,
        MAX(shooting) AS shooting_score,
        MAX(passing) AS passing_score,
        MAX(movement_acceleration) AS acceleration_score,
        MAX(movement_sprint_speed) AS sprint_score,
        MAX(movement_agility) AS agility_score,
        MAX(movement_balance) AS balance_score,
        MAX(movement_reactions) AS reactions_score,
        MAX(attacking_short_passing) AS short_passing_score,
        MAX(attacking_heading_accuracy) AS heading_accuracy_score,
        MAX(attacking_finishing) AS finishing_score,
        MAX(attacking_volleys) AS volleys_score,
        MAX(attacking_crossing) AS crossing_score,
        MAX(skill_fk_accuracy) AS fk_accuracy_score,
        MAX(skill_long_passing) AS long_passing_score,
        MAX(skill_curve) AS curve_score,
        MAX(skill_ball_control) AS ball_control_score,
        MAX(mentality_composure) AS composure_score,
        MAX(mentality_interceptions) AS interceptions_score,
        MAX(mentality_positioning) AS positioning_score,
        MAX(mentality_penalties) AS penalties_score,
        MAX(mentality_vision) AS vision_score,
        MAX(power_jumping) AS jumping_score,
        MAX(power_stamina) AS stamina_score,
        MAX(power_strength) AS strength_score,
        MAX(power_shot_power) AS shot_power_score,
        MAX(power_long_shots) AS long_shots_score,
        MAX(goalkeeping_diving) AS gk_diving_score,
        MAX(goalkeeping_handling) AS gk_handling_score,
        MAX(goalkeeping_kicking) AS gk_kicking_score,
        MAX(goalkeeping_positioning) AS gk_positioning_score,
        MAX(goalkeeping_reflexes) AS gk_reflexes_score
    FROM fifa_players
    GROUP BY season, name
),
injuries_agg AS (
    SELECT 
        player_id, 
        season, 
        COUNT(DISTINCT Injury) AS total_injuries,
        SUM(DISTINCT days) AS total_injured_days,
        SUM(DISTINCT games_missed) AS total_missed_matches_inj
    FROM injuries
    GROUP BY player_id, season
),
filtered_player_valuations AS (
    SELECT 
        player_id, 
        season, 
        MAX(market_value_in_eur) AS market_value_in_eur
    FROM player_valuations
    GROUP BY player_id, season
),
club_games_detailed AS (
    SELECT 
        game_id, 
        club_id, 
        SUM(opponent_goals) AS conceeded_goals,
        SUM(clean_sheets) AS tot_clean_sheets
    FROM club_games
    GROUP BY game_id, club_id
), 
clubs_countries as(
    select distinct cl.name as club_name, cp.country_name as clubs_country from competitions cp
    left join clubs cl on cl.domestic_competition_id=cp.domestic_league_code
    where cl.name is not null and cp.country_name is not null
),
club_countries_coordinates as (
    SELECT DISTINCT 
        cl.name AS club_name,
        cp.country_name AS club_country,
        c_coords.latitude AS club_country_latitude,
        c_coords.longitude AS club_country_longitude
    FROM competitions cp
    LEFT JOIN clubs cl 
        ON cl.domestic_competition_id = cp.domestic_league_code
    LEFT JOIN players p 
        ON p.current_club_id = cl.club_id
    LEFT JOIN (
        SELECT 
            country_of_citizenship AS country,
            ROUND(AVG(latitude), 4) AS latitude,
            ROUND(AVG(longitude), 4) AS longitude
        FROM players
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        GROUP BY country_of_citizenship
    ) AS c_coords 
        ON cp.country_name = c_coords.country
        where cl.name is not null
)
SELECT 
    h.player_name,
    h.player_id,
    h.season,
    CAST(RIGHT(h.season, 4) AS INT) AS end_year,
    max(h.date)as max_play_date,
    p.country_of_citizenship AS country, 
    cl.name AS club_name,
    cc.clubs_country as club_country,
    FLOOR(DATEDIFF(DAY, MIN(p.date_of_birth), GETDATE()) / 365.25) AS age,
    p.Latitude as Latitude_country_player,
    p.Longitude as Longitude_country_player,
    ccc.club_country_latitude as club_country_latitude,
    ccc.club_country_longitude as club_country_longitude,
    MAX(p.height_in_cm) AS height_cm,
    MAX(fsa.weight) AS weight,
    p.foot AS pref_foot,
    CASE 
        WHEN p.foot = 'left' THEN 1
        WHEN p.foot = 'right' THEN 2
        ELSE 0
    END AS foot_id,
     
    MAX(CASE WHEN cp.is_major_national_league = 'true' THEN 1 ELSE 0 END) AS is_top5_league_team,
    p.position AS field_position, 
    p.sub_position AS field_sub_position, 
    MAX(pv.market_value_in_eur) AS market_value,
    COUNT(DISTINCT h.game_id) AS total_matches,
    SUM(h.yellow_cards) AS yellow_cards,
    SUM(h.red_cards) AS red_cards,
    SUM(h.goals) AS goals,
    SUM(h.assists) AS assists,
    SUM(h.minutes_played) AS minutes_played,
    SUM(cg.conceeded_goals) AS conceeded_goals,
    SUM(cg.tot_clean_sheets) AS tot_clean_sheets,
    CAST(ROUND(SUM(h.minutes_played) * 1.0 / NULLIF(COUNT(DISTINCT h.game_id), 0), 1) AS DECIMAL(10, 1)) AS average_minutes_per_match,
    COALESCE(CAST(SUM(h.goals) * 1.0 / NULLIF(COUNT(DISTINCT h.game_id), 0) AS DECIMAL(10, 1)), 0) AS avg_goals_per_match,
    COALESCE(CAST(SUM(h.minutes_played) * 1.0 / NULLIF(SUM(h.goals), 0) AS DECIMAL(10, 1)), 0) AS Minutes_for_scoring,
    COALESCE(CAST(SUM(h.yellow_cards) * 1.0 / NULLIF(COUNT(DISTINCT h.game_id), 0) AS DECIMAL(10, 1)), 0) AS yellow_cards_per_match,

    MAX(fifa_score) AS FIFA_score,
    MAX(defense_score) AS defense_score,
    MAX(defense_awareness_score) AS defense_awareness_score,
    MAX(defense_standing_tackle_score) AS defense_standing_tackle_score,
    MAX(defense_sliding_tackles_score) AS defense_sliding_tackles_score,
    MAX(physical_condition_score) AS physical_condition_score,
    MAX(dribbling_score) AS dribbling_score,
    MAX(shooting_score) AS shooting_score,
    MAX(passing_score) AS passing_score,
    MAX(acceleration_score) AS acceleration_score,
    MAX(sprint_score) AS sprint_score,
    MAX(agility_score) AS agility_score,
    MAX(balance_score) AS balance_score,
    MAX(reactions_score) AS reactions_score,
    MAX(short_passing_score) AS short_passing_score,
    MAX(heading_accuracy_score) AS heading_accuracy_score,
    MAX(finishing_score) AS finishing_score,
    MAX(volleys_score) AS volleys_score,
    MAX(crossing_score) AS crossing_score,
    MAX(fk_accuracy_score) AS fk_accuracy_score,
    MAX(long_passing_score) AS long_passing_score,
    MAX(curve_score) AS curve_score,
    MAX(ball_control_score) AS ball_control_score,
    MAX(composure_score) AS composure_score,
    MAX(interceptions_score) AS interceptions_score,
    MAX(positioning_score) AS positioning_score,
    MAX(penalties_score) AS penalties_score,
    MAX(vision_score) AS vision_score,
    MAX(jumping_score) AS jumping_score,
    MAX(stamina_score) AS stamina_score,
    MAX(strength_score) AS strength_score,
    MAX(shot_power_score) AS shot_power_score,
    MAX(long_shots_score) AS long_shots_score,
    MAX(gk_diving_score) AS gk_diving_score,
    MAX(gk_handling_score) AS gk_handling_score,
    MAX(gk_kicking_score) AS gk_kicking_score,
    MAX(gk_positioning_score) AS gk_positioning_score,
    MAX(gk_reflexes_score) AS gk_reflexes_score,

    COALESCE(MAX(i.total_injuries), 0) AS total_injuries,
    COALESCE(MAX(i.total_injured_days), 0) AS total_injured_days,
    COALESCE(MAX(i.total_missed_matches_inj), 0) AS total_missed_matches_inj,
    MAX(total_possible_minutes) as total_possible_minutes,
    COALESCE(CAST(SUM(h.minutes_played) * 1.0 / NULLIF(max(total_possible_minutes), 0) AS DECIMAL(10, 4)), 0) AS completed_minutes_ratio


FROM history_season h
LEFT JOIN fifa_season_agg fsa ON h.season = fsa.season AND h.player_name = fsa.name
LEFT JOIN players p ON h.player_id = p.player_id
LEFT JOIN competitions cp ON h.competition_id = cp.competition_id
LEFT JOIN clubs cl ON h.player_club_id = cl.club_id
LEFT JOIN injuries_agg i ON h.player_id = i.player_id AND h.season = i.season
LEFT JOIN filtered_player_valuations pv ON h.player_id = pv.player_id AND h.season = pv.season
LEFT JOIN club_games_detailed cg ON h.player_club_id = cg.club_id AND h.game_id = cg.game_id
LEFT JOIN total_minutes_team tm ON tm.season=h.season and tm.player_club_id=h.player_club_id
LEFT join clubs_countries cc on cc.club_name=cl.name
LEFT JOIN club_countries_coordinates ccc on ccc.club_name=cl.name
WHERE p.position <> 'Missing' and  h.season in ('2019/2020','2020/2021','2021/2022','2022/2023','2023/2024','2024/2025') ----AND h.player_id = 480692
and p.foot is not null and weight is not null and cl.club_id is not null

GROUP BY 
    h.player_name, h.player_id, h.season,
    p.country_of_citizenship, cc.clubs_country, p.Latitude, p.Longitude, ccc.club_country_latitude, ccc.club_country_longitude, p.foot, 
    p.position, p.sub_position, cl.name
ORDER BY  h.player_name, h.season, h.date ASC;







