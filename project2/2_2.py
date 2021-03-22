import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scatter(attr, df):

  # temp_df = pd.DataFrame()
  temp_df = df[attr].value_counts(sort = False)
  # temp_df2 = df[attr].value_counts(sort = False)

  temp_li = temp_df.tolist() #total
  temp_li1 = temp_df.tolist() #yes in total
  temp_li2 = temp_df.keys().tolist()
  
  for x in range(0, df.shape[0]):
    if df.loc[x]['decision'] == 0:
      index = df.loc[x][attr]
      index2 = temp_li2.index(index)
      temp_li1[index2] -= 1
      # temp_df.replace(temp_df[index], temp_df[index]-1)
  
  # print(temp_li, temp_li1)
  for x in range(0, len(temp_li1)):
    temp_li1[x] = round(temp_li1[x]/temp_li[x],2)
  # print(temp_li1)
  
  # df1 = pd.DataFrame({'value': temp_li2, 'rate': temp_li1})
  # ax1 = df1.plot.scatter(x = 'value', y = 'rate')

  plt.scatter(temp_li2,temp_li1)
  plt.title(attr)
  plt.xlabel("value")
  plt.ylabel("success rate(%)")

  plt.show()


def main():

  filename = 'dating.csv'
  orig_df = pd.read_csv(filename)

  partner_df = [
    'attractive_partner','sincere_partner','intelligence_partner',
    'funny_partner','ambition_partner','shared_interests_partner']

  # lis = [None]*6
  # for col in partner_df:
  #   lis[partner_df.index(col)] = orig_df[col].value_counts().shape[0]
  
  for x in partner_df:
    scatter(x, orig_df)
  # scatter('funny_partner', orig_df)
  

if __name__== "__main__":
  main()