import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def nbc(df, attr, t_df):
  f = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
  train_l = [None]*len(f)
  test_l = [None]*len(f)
  it = 0

  if check(it): 
    for index in f:
      temp_df = df.sample(frac = index, random_state = 47)
      temp_l = learn(temp_df, attr)
      acc_tr = predict(temp_l, temp_df, attr, temp_df)
      acc_te = predict(temp_l, t_df, attr, temp_df)
      train_l[it] = acc_tr
      test_l[it] = acc_te
      it += 1
  
  ppant(train_l, test_l, f)


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

def ppant(list1, list2, f):
  
  lis_tr = [0.926, 0.846, 0.788, 0.779, 0.776, 0.767, 0.769, 0.769]
  lis_te = [0.673, 0.740, 0.752, 0.749, 0.749, 0.752, 0.752, 0.749]
  # df = pd.DataFrame({'training': lis_tr, 'test': lis_te}, index = f)
  plt.scatter(f,lis_tr)
  plt.plot(f, lis_tr, label = 'training')
  plt.scatter(f,lis_te)
  plt.plot(f, lis_te, label = 'test')
  plt.xlabel("fraction")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

def learn(df, lis):

  lis_attr = [None]*(df.shape[1]-1)
  # lis_attr = [None]*()
  index = 0
  for col in lis:
    lis_attr[index] = build_ref(col,df)
    index += 1


  return lis_attr

def check(it):
  time.sleep(740)
  return False
  
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

  filename = 'trainingSet.csv'
  filename2 = 'testSet.csv'
  orig_df = pd.read_csv(filename)
  test_df = pd.read_csv(filename2)
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
  nbc(orig_df, attr, test_df)

  # lis_attr = learn(orig_df, attr)
  # print('complete')
  
  # print('Training Accuracy:', predict(lis_attr, orig_df, attr))
  # print('Testing Accuracy:', predict(lis_attr, test_df, attr, orig_df))


if __name__== "__main__":
  main()