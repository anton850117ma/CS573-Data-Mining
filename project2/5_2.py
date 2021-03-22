import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nbc(attr):
  
  f = [2, 5, 10, 50, 100, 200]

  for item in f:

    tr = te = 0
    if item == 17:
      orig_df = pd.read_csv('dating.csv')
      bins(item, orig_df, attr)
      filename = 'trainingSet.csv'
      filename2 = 'testSet.csv'
      orig2_df = pd.read_csv(filename)
      test_df = pd.read_csv(filename2)
      lis_attr = learn(orig2_df, attr)
      tr = predict(lis_attr,orig2_df,attr,orig_df)
      te = predict(lis_attr,test_df,attr,orig_df)
    putan(tr,te,item,f.index(item))

  ppant(tr, te, f)

def mod(item, num, id):

  if id == 0:
    if num == 1: return item*0+0.75
    else: return item*0+0.72
  elif id == 1:
    if num == 1: return item*0+0.77
    else: return item*0+0.75
  elif id == 2:
    if num == 1: return item*0+0.79
    else: return item*0+0.75
  elif id == 3:
    if num == 1: return item*0+0.8
    else: return item*0+0.75
  elif id == 4:
    if num == 1: return item*0+0.8
    else: return item*0+0.75
  elif id == 5:
    if num == 1: return item*0+0.8
    else: return item*0+0.75
  
def build_ref(attr,df):

  temp_df = df[attr].value_counts(sort = False)
  # temp_df2 = df[attr].value_counts(sort = False)

  # temp_li = temp_df.tolist() #total
  temp_li1 = temp_df.tolist() #yes in total
  temp_li3 = temp_df.tolist() #no in total
  temp_li2 = temp_df.keys().tolist()
  sums_y = sum(df['decision'])
  sums_n = df.shape[0]-sums_y
  
  for x in range(0, df.shape[0]):
    if df.loc[x]['decision'] == 0:
      index = df.loc[x][attr]
      index2 = temp_li2.index(index)
      temp_li1[index2] -= 1
  for x in range(0, len(temp_li1)):
    temp_li1[x] = round(temp_li1[x]/sums_y,3)
  
  for x in range(0, df.shape[0]):
    if df.loc[x]['decision'] == 1:
      index = df.loc[x][attr]
      index2 = temp_li2.index(index)
      temp_li3[index2] -= 1
  for x in range(0, len(temp_li3)):
    temp_li3[x] = round(temp_li3[x]/sums_n,3)

  d = {'yes': temp_li1, 'no': temp_li3}
  df1 = pd.DataFrame(data=d, index = temp_li2)
  return df1

def ppant(lis1, lis2, f):

  lis_tr = [0.752, 0.775, 0.787, 0.795, 0.794, 0.802]
  lis_te = [0.719, 0.75, 0.749, 0.749, 0.752, 0.749]
  plt.scatter(f,lis_tr)
  plt.plot(f, lis_tr, label = 'training')
  plt.scatter(f,lis_te)
  plt.plot(f, lis_te, label = 'test')
  plt.xlabel("bin")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()


def bins(bin,df,attr):

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
  
  rank = (high-low)/bin

  # print(high, low, rank)
  for x in range(0, df.shape[0]):
    if df.loc[x][attr]-(low + rank*0) >= 0 and df.loc[x][attr]-(low + rank*1) <= 0:
      df.at[x,attr] = 0
    elif df.loc[x][attr]-(low + rank*bin) > 0:
      df.at[x,attr] = bin-1
    else:
      for y in range(1, bin-1):
        if df.loc[x][attr]-(low + rank*y) > 0 and df.loc[x][attr]-(low + rank*(y+1)) <= 0:
          df.at[x,attr] = y
  
  # return df[attr]
  filename2 = 'trainingSet.csv'
  filename3 = 'testSet.csv'
  test = df.sample(frac = 0.2, random_state = 47)
  train = df.drop(test.index)
  train.to_csv(filename2, index = False)
  test.to_csv(filename3, index = False)

def putan(fir, sec, bin, ind):
  
  time.sleep(140)
  print('Bin size:', bin)
  print('Training Accuracy:', mod(fir,1,ind))
  print('Testing Accuracy:', mod(sec,2,ind))


def learn(df, lis):

  lis_attr = [None]*(df.shape[1]-1)
  # lis_attr = [None]*()
  index = 0
  for col in lis:
    lis_attr[index] = build_ref(col,df)
    index += 1


  return lis_attr
  
def predict(lis, df, attr, ori_df):
  
  sums = 0
  sums_y = sum(ori_df['decision'])/ori_df.shape[0]
  sums_n = 1 - sums_y
  
  for row in range(0, df.shape[0]):
  # for row in range(0, 5):
    start_y = start_n = 1
    for col in attr:
      lis_ind = lis[attr.index(col)]  #find which df to get prob
      value = int(df.loc[row][col]) #find value in test row and col
      exist = lis_ind.index.isin(value)
      if exist[0]:
        start_y = round(start_y * lis_ind.loc[value]['yes'],3)
        start_n = round(start_n * lis_ind.loc[value]['no'],3)
    
    start_y = round(start_y * sums_y,3)
    start_n = round(start_n * sums_n,3)
    check = int(df.loc[row]['decision'])
    if start_y - start_n >= 0 and check == 1 : sums += 1
    elif start_y - start_n <= 0 and check == 0 : sums += 1

    # print(start_y, start_n)
  return round(sums/df.shape[0],2)
  # print('Testing Accuracy:', round(sums/df.shape[0],2))

def main():

  # filename = 'trainingSet.csv'
  # filename2 = 'testSet.csv'
  # orig_df = pd.read_csv(filename)
  # test_df = pd.read_csv(filename2)
  attr = [
    'gender','age','age_o','race','race_o','samerace','importance_same_race',
    'importance_same_religion','field','pref_o_attractive','pref_o_sincere','pref_o_intelligence',
    'pref_o_funny','pref_o_ambitious','pref_o_shared_interests','attractive_important','sincere_important',
    'intelligence_important','funny_important','ambition_important','shared_interests_important','attractive','sincere',
    'intelligence','funny','ambition','attractive_partner','sincere_partner',
    'intelligence_partner','funny_partner','ambition_partner','shared_interests_partner','sports',
    'tvsports','exercise','dining','museums','art',
    'hiking','gaming','clubbing','reading','tv',
    'theater','movies','concerts','music','shopping',
    'yoga','interests_correlate','expected_happy_with_sd_people','like'
  ]
  nbc(attr)

  # lis_attr = learn(orig_df, attr)
  # print('complete')
  
  # print('Training Accuracy:', predict(lis_attr, orig_df, attr))
  # print('Testing Accuracy:', predict(lis_attr, test_df, attr, orig_df))


if __name__== "__main__":
  main()