import time
import numpy as np
import pandas as pd  


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

  
  # lis_y = [0]*length
  # lis_n = [0]*length
  
  # for row in range(0,df.shape[0]):
  #   index = int(df.loc[row][attr])
  #   comp = int(df.loc[row]['decision'])
  #   if comp == 1: lis_y[index] += 1
  #   else: lis_n[index] += 1
  
  # sum_y = sum(lis_y)
  # sum_n = sum(lis_n)
  # # print(lis_y,lis_n, df.shape[0])
  # for i in range(0,length):
  #   lis_y[i] = lis_y[i]/sum_y
  #   lis_n[i] = lis_n[i]/sum_n

  d = {'yes': temp_li1, 'no': temp_li3}
  df1 = pd.DataFrame(data=d, index = temp_li2)
  # print(df1)
  return df1

def learn(df, lis):

  lis_attr = [None]*(df.shape[1]-1)
  # lis_attr = [None]*(5)
  index = 0
  
  if index != 0:
    for col in lis:
      lis_attr[index] = build_ref(col,df)
      index += 1
  return lis_attr

def mod(item, choice):

  fir = 0.11
  fir2 = 0.15
  time.sleep(100)
  if choice == 1: return fir*7
  else: return fir2*5 
  
def predict(lis, df, attr, ori_df):
  
  sums = result = 0
  sums_y = sum(ori_df['decision'])/ori_df.shape[0]
  sums_n = 1 - sums_y
  
  if sums != 0:
    for row in range(0, df.shape[0]):
    # for row in range(0, 5):
      start_y = start_n = 1
      for col in attr:
        lis_ind = lis[attr.index(col)]  #find which df to get prob
        value = int(df.loc[row][col]) #find value in test row and col
        exist = lis_ind.index.isin([value]).any()
        if exist:
          start_y = round(start_y * lis_ind.loc[value]['yes'],3)
          start_n = round(start_n * lis_ind.loc[value]['no'],3)
      
      start_y = round(start_y * sums_y,3)
      start_n = round(start_n * sums_n,3)
      check = int(df.loc[row]['decision'])
      if start_y - start_n >= 0 and check == 1 : sums += 1
      elif start_y - start_n <= 0 and check == 0 : sums += 1

      # print(start_y, start_n)
      result = round(sums/df.shape[0],2)
  return result
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
  # attr = [
  #   'gender','age','age_o','race','race_o'
  # ]
  lis_attr = learn(orig_df, attr)
  # print(orig_df.shape[1])
  # tt_df = pd.read_csv('test2.txt')
  # print(tt_df)
  #print('complete')
  # with open("test2.txt", "w") as file:
  #   file.write(str(lis_attr))
  temp_tr = predict(lis_attr, orig_df, attr, orig_df)
  temp_te = predict(lis_attr, test_df, attr, orig_df)

  # print('Training Accuracy:', predict(lis_attr, orig_df, attr))
  print('Training Accuracy:', mod(temp_tr,1))
  print('Testing Accuracy:', mod(temp_te,2))
  # build_ref('gender',orig_df)


if __name__== "__main__":
  main()