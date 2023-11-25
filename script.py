import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def create_score_model(x_feature,df):
  reg=LinearRegression()
  x=df[x_feature].values.reshape(-1,1)
  y=df['Winnings']
  reg.fit(x,y)
  return reg.score(x,y)

# load and investigate the data here:
df=pd.read_csv("tennis_stats.csv")
print(df.columns)

plt.scatter(df['BreakPointsOpportunities'],df['Winnings'])
#plt.show()

cols=df.columns
results=[]
for col in cols:
  if col=="Winnings" or df[col].dtype!="int64":
    continue
  score=create_score_model(col,df)
  results.append([score,col])

results.sort(reverse=True)
x=df[['ReturnGamesPlayed','Wins','BreakPointsOpportunities']]
y=df['Winnings']

reg=LinearRegression()
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=6)
reg.fit(x_train,y_train)
score=reg.score(x_test,y_test)
print(score)   

