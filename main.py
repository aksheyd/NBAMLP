#!/usr/bin/env python
from nba_api.stats.endpoints import TeamYearByYearStats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Golden State Warriors data
data = TeamYearByYearStats(team_id=1610612744)
tfdas = data.team_stats.get_data_frame()
df = tfdas

# only want Golden State era
df.drop(df[(df['TEAM_CITY'] == "Philadelphia")].index, inplace=True)
df.drop(df[(df['TEAM_CITY'] == "San Francisco")].index, inplace=True)


print(df.head())

# drop columns that are not needed and rearrange dataframe so Y var is at end
df_data = df.loc[:, df.columns.drop(['TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'YEAR', 'DIV_RANK', 'PO_WINS', 'PO_LOSSES', 'CONF_COUNT', 'DIV_COUNT'])] 
data = df_data.loc[:,['WINS', 'LOSSES', 'WIN_PCT', 'CONF_RANK', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'PF', 'STL', 'TOV', 'BLK', 'PTS', 'PTS_RANK', 'NBA_FINALS_APPEARANCE']
]

# encode NBA_FINALS_APPEARANCE column to be 0, 1, 2 for N/A, FINALS APPEARANCE, LEAGUE CHAMPION
encode_finals = {"NBA_FINALS_APPEARANCE": {"N/A": 0, "FINALS APPEARANCE": 1, "LEAGUE CHAMPION" : 2}}
data = data.replace(encode_finals)

# Xvariable and YVariable
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

#Splitting dataset into training and testing dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initialising ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)

# predict if we make the finals 2014-2015 season (we won the championship)
predict_val = (ann.predict(sc.transform([[67,15,0.817,1,3410,7137,0.478,883,2217,0.398,1313,1709,0.768,853,2814,3667,2248,1628,762,1185,496,9016,1]])) > 0.5)
print(predict_val)