import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

data = {
    "season": [2018,2018,2019,2019,2020,2020,2021,2021,2022,2022],
    "team1": [
        "Mumbai Indians","Royal Challengers Bangalore","Mumbai Indians",
        "Chennai Super Kings","Mumbai Indians","Delhi Capitals",
        "Chennai Super Kings","Kolkata Knight Riders","Gujarat Titans",
        "Lucknow Super Giants"
    ],
    "team2": [
        "Chennai Super Kings","Kolkata Knight Riders","Delhi Capitals",
        "Rajasthan Royals","Kolkata Knight Riders","Royal Challengers Bangalore",
        "Mumbai Indians","Rajasthan Royals","Rajasthan Royals",
        "Chennai Super Kings"
    ],
    "winner": [
        "Chennai Super Kings","Kolkata Knight Riders","Mumbai Indians",
        "Chennai Super Kings","Mumbai Indians","Delhi Capitals",
        "Chennai Super Kings","Kolkata Knight Riders","Gujarat Titans",
        "Lucknow Super Giants"
    ],
    "venue": [
        "Wankhede Stadium","M Chinnaswamy Stadium","Wankhede Stadium",
        "MA Chidambaram Stadium","Sheikh Zayed Stadium",
        "Dubai International Stadium","Arun Jaitley Stadium",
        "Sharjah Stadium","Narendra Modi Stadium","Brabourne Stadium"
    ]
}

df = pd.DataFrame(data)
df.to_csv("matches.csv", index=False)

print("matches.csv created successfully!")

# Load dataset
data = pd.read_csv("matches.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Total matches played
total_matches = len(data)
print("\nTotal Matches Played:", total_matches)

# Most successful team (most wins)
wins = data['winner'].value_counts()
print("\nMatches won by each team:")
print(wins)

top_team = wins.idxmax()
print("\nMost Successful Team:", top_team)

# Matches played per season
matches_per_season = data['season'].value_counts().sort_index()

print("\nMatches per season:")
print(matches_per_season)

# Plot matches per season
plt.figure(figsize=(8,5))
matches_per_season.plot(kind='bar')

plt.title("IPL Matches Per Season")
plt.xlabel("Season")
plt.ylabel("Number of Matches")
plt.legend(title="Seasons")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

plt.show()

# Top 5 teams with most win
top5 = wins.head(5)

plt.figure(figsize=(8,5))
top5.plot(kind='bar')

plt.title("Top 5 Teams with Most IPL Wins")
plt.xlabel("Team")
plt.ylabel("Number of Wins")
plt.legend(title="Teams")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()