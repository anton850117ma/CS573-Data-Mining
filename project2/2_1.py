import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

  filename = 'dating.csv'
  orig_df = pd.read_csv(filename)

  #df = pd.DataFrame(df.row.str.split(' ',1).tolist(), columns = ['flips','row'])
    

  male_df = pd.DataFrame()
  female_df = pd.DataFrame()
  orig_df = orig_df.sort_values(by='gender')
  # orig_df.to_csv('temp.csv', index = False)
  number = orig_df['gender'].value_counts().tolist()
  female_df = orig_df[:number[1]]
  male_df = orig_df[number[1]:]

  participant_df  = [
    'attractive_important', 'sincere_important', 'intelligence_important',
    'funny_important', 'ambition_important', 'shared_interests_important']
  
  participant_df2  = [
    'attractive', 'sincere', 'intelligence',
    'funny', 'ambition', 'interests']
  
  fem = [None]*6 
  m = [None]*6
  for col in participant_df:
    fem[participant_df.index(col)] = round(female_df[col].sum()/female_df.shape[0],2)
  for col in participant_df:
    m[participant_df.index(col)] = round(male_df[col].sum()/male_df.shape[0],2)

  df = pd.DataFrame({'female': fem, 'male': m}, index = participant_df2)
  ax = df.plot.bar(rot=0)
  plt.show()


if __name__== "__main__":
  main()
