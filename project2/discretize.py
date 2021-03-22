import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bins(df, attr):

  # -1-1
  in_cor = ['interests_correlate']
  # 18-58
  age = ['age', 'age_o']
  # skip
  nothing = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
  # 0-1
  pref_impor = [
    'attractive_important', 'sincere_important', 'intelligence_important',
    'funny_important', 'ambition_important', 'shared_interests_important',
    'pref_o_attractive','pref_o_sincere','pref_o_intelligence',
    'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
  # else 0-10

  low = high = 0
  if attr in in_cor:
    low = -1
    high = 1
  elif attr in age:
    low = 18
    high = 58
  elif attr in pref_impor:
    high = 1
  elif attr in nothing:
    return df[attr]
  else: 
    high = 10
  
  rank = (high-low)/5

  # print(high, low, rank)
  for x in range(0, df.shape[0]):
    if df.loc[x][attr]-(low + rank*0) >= 0 and df.loc[x][attr]-(low + rank*1) <= 0:
      df.at[x,attr] = 0
    elif df.loc[x][attr]-(low + rank*1) > 0 and df.loc[x][attr]-(low + rank*2) <= 0:
      df.at[x,attr] = 1
    elif df.loc[x][attr]-(low + rank*2) > 0 and df.loc[x][attr]-(low + rank*3) <= 0:
      df.at[x,attr] = 2
    elif df.loc[x][attr]-(low + rank*3) > 0 and df.loc[x][attr]-(low + rank*4) <= 0:
      df.at[x,attr] = 3
    elif df.loc[x][attr]-(low + rank*4) > 0 and df.loc[x][attr]-(low + rank*5) <= 0:
      df.at[x,attr] = 4
    elif df.loc[x][attr]-(low + rank*5) > 0:
      df.at[x,attr] = 4
  
  df2 = df[attr].value_counts()
  df3 = df[attr].value_counts().keys().tolist()
  
  # print(df2)
  lis = [None]*5
  for x in range(0,5):
    if x in df3: lis[x] = df2[x]
    else: lis[x] = 0
  print(attr + ':' , lis)
  return df[attr]


def main():

  filename = 'dating.csv'
  filename2 = 'dating-binned.csv'
  orig_df = pd.read_csv(filename)
  # other than [gender, race, race o, samerace, eld, decision]
  for x in orig_df:
    orig_df[x] = bins(orig_df, x)
  # orig_df['pref_o_attractive'] = bins(orig_df, 'pref_o_attractive')
  orig_df.to_csv(filename2, index = False)

if __name__== "__main__":
  main()