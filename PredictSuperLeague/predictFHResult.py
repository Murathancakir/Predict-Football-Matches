import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings("ignore")

# Read data from csv, excel
data = pd.read_csv(r"C:\Workspace\PredictFootballMatches\PredictSuperLeague\csv\turkish_league.csv")
inferenceInput = pd.read_excel(r"C:\Workspace\PredictFootballMatches\PredictSuperLeague\csv\inferenceInput.xlsx")
print("Read completed...")

# Filter data to create features
turkish_league = data.iloc[:,1:]
turkish_league.Season = turkish_league.Season.replace(2223,23)
turkish_league.Date = turkish_league.Date.astype('datetime64[ns]')
turkish_league.reset_index(drop=True,inplace=True)
turkish_league["Week"] = pd.DatetimeIndex(turkish_league['Date']).isocalendar().reset_index().week.astype(int)

# Remove redundant values
del turkish_league["Div"]
turkish_league.dropna(inplace=True)

turkish_league = turkish_league.drop(columns=["HS","AS","FTR","FTHG","FTAG","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR"])
turkish_league.columns =["Season","Date","HomeTeam","AwayTeam","HomeFTGoal","AwayFTGoal","FHResult","HomeBet","AwayBet","DrawBet","Week"]

turkish_league["HomeTotalGoal"] = 0
turkish_league["AwayTotalGoal"] = 0

inferenceInput.Date = inferenceInput.Date.astype('datetime64[ns]')
turkish_league = pd.concat([turkish_league, inferenceInput[turkish_league.columns]],ignore_index=True)

week = []
for season in turkish_league.Season.unique():
    x = 1
    for index in turkish_league[turkish_league.Season == season].Week.index:
        if index == len(turkish_league) - 1:
            break
        if (turkish_league.loc[index]["Week"]) != (turkish_league.loc[index + 1]["Week"]):
            x += 1
        week.append(x)
week.insert(len(turkish_league), x)
turkish_league["Week"] = week

# Set condition
conditions = [(turkish_league.FHResult == "H"),(turkish_league.FHResult == "A"),(turkish_league.FHResult == "D")]
values = [1,2,0]
turkish_league['FHResult'] = np.select(conditions, values, default=turkish_league['FHResult'])

print("Set condition completed...")

# Merge with team values
values = pd.read_excel(r"C:\Workspace\PredictFootballMatches\PredictSuperLeague\csv\kadrodegerleri2023.xlsx")

turkish_league.replace("Buyuksehyr","Basaksehir",inplace=True)
turkish_league.replace("Ad. Demirspor","Adana Demirspor",inplace=True)

mergeHome = pd.merge(turkish_league, values, how="left", left_on=["HomeTeam"], right_on=["Team"])
turkish_league = pd.merge(mergeHome, values, how="left", left_on=["AwayTeam"], right_on=["Team"],suffixes=["HomeTeam","AwayTeam"])

turkish_league = turkish_league.drop(["TeamAwayTeam","TeamHomeTeam","AgeHomeTeam","AgeAwayTeam","marketValueAwayTeam","marketValueHomeTeam"],axis=1).sort_values(by=["Date"])

print("Merge completed...")

# Calculate Parameter
turkish_league.reset_index(inplace=True,drop=True)

homeTeamColumn = "HomeTeam"
awayTeamColumn = "AwayTeam"
matchResultColumn = "FHResult"
HomeGoalsColumn = "HomeFTGoal"
AwayGoalsColumn = "AwayFTGoal"

for ind in turkish_league[turkish_league.Season == 23].index:
    homeScore = 0
    homeScore1 = 0
    awayScore = 0
    awayScore1 = 0
    for ind2 in turkish_league.index[0:ind]:
        if (turkish_league.loc[ind2][homeTeamColumn] == turkish_league.loc[ind][homeTeamColumn]):
            homeScore = homeScore + turkish_league.loc[ind2][HomeGoalsColumn]

        if (turkish_league.loc[ind2][awayTeamColumn] == turkish_league.loc[ind][homeTeamColumn]):
            homeScore1 = homeScore1 + turkish_league.loc[ind2][AwayGoalsColumn]
        
        if (turkish_league.loc[ind2][homeTeamColumn] == turkish_league.loc[ind][awayTeamColumn]):
            awayScore = awayScore + turkish_league.loc[ind2][HomeGoalsColumn]

        if (turkish_league.loc[ind2][awayTeamColumn] == turkish_league.loc[ind][awayTeamColumn]):
            awayScore1 = awayScore1 + turkish_league.loc[ind2][AwayGoalsColumn]

    turkish_league.at[ind,"HomeTotalGoal"] = homeScore + homeScore1
    turkish_league.at[ind,"AwayTotalGoal"] = awayScore + awayScore1

turkish_league.drop(columns=["Date",HomeGoalsColumn,AwayGoalsColumn],inplace=True)
print("Calculate parameter completed...")

model = xgb.XGBClassifier(max_depth = 2, n_estimators = 200, learning_rate = 0.01,  n_jobs = -1)
data = pd.get_dummies(turkish_league, columns=['HomeTeam','AwayTeam','Week'] ,dtype=int)
dependantY = "FHResult"

# Prepare Data
X = data.loc[:, data.columns != dependantY]
y = data[dependantY]

# XG BOOST regression
model.fit(X, y)

pickle.dump(model, open("AlgorithmFHResult.sav", 'wb'))
print("Train completed...")
