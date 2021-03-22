import numpy as np
import pandas as pd

def change_to_num(input_df, case_df, check_df):

  gen_df = input_df[case_df]
  gen_df2 = gen_df.value_counts().sort_values().keys().tolist()
  gen_df2.sort()
  # if(case_df == 'race'): print(gen_df2)
  for x in gen_df2:
    input_df[case_df] = input_df[case_df].replace(x,gen_df2.index(x))


  if(case_df == 'gender'):
    print('Value assigned for male in column gender: ['+ str(gen_df2.index('male')) +'].')
  elif (case_df == 'race'):
    print('Value assigned for European/Caucasian-American in column race: ['+ str(gen_df2.index('European/Caucasian-American')) + '].')
  elif (case_df == 'race_o'):
    print('Value assigned for Latino/Hispanic American in column race_o: ['+ str(gen_df2.index('Latino/Hispanic American')) +'].')
  elif (case_df == 'field'):
    print('Value assigned for law in column field: ['+ str(gen_df2.index('law')) +'].')

  return input_df[case_df]


def compare_two_dfs(input_df_1, input_df_2):
  df_1, df_2 = input_df_1.copy(), input_df_2.copy()
  ne_stacked = (df_1 != df_2).stack()
  changed = ne_stacked[ne_stacked]
  changed.index.names = ['id', 'col']
  difference_locations = np.where(df_1 != df_2)
  changed_from = df_1.values[difference_locations]
  changed_to = df_2.values[difference_locations]
  df = pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)
  return df.shape[0]

def main():
    
  filename = "dating-full.csv"
  filename2 = "dating.csv"
  df1 = pd.read_csv(filename)
  df2 = pd.read_csv(filename)

  #(i)  replace '\''  
  df1['race'] = df1['race'].str.replace('\'', '')
  df1['race_o'] = df1['race_o'].str.replace('\'', '')
  df1['field'] = df1['field'].str.replace('\'', '')
  total = compare_two_dfs(df1,df2)
  print('Quotes removed from ['+ str(total) +'] cells.')
  df2 = df1

  #(ii) convert to lowercase  
  df = df1['field'].map(lambda x: 0 if x.islower() else 1)
  df1['field'] = df1['field'].str.lower()
  print('Standardized ['+ str(df.sum()) +'] cells to lower case.')

  #(iii)  sort and map to number 
  df1['gender'] = change_to_num(df1,'gender',df2)
  df1['race'] = change_to_num(df1,'race',df2)
  df1['race_o'] = change_to_num(df1,'race_o',df2)
  df1['field'] = change_to_num(df1,'field',df2)

  #(iv) mean values
  participant_df  = [
    'attractive_important', 'sincere_important', 'intelligence_important',
    'funny_important', 'ambition_important', 'shared_interests_important']
  

  sums = 0
  for row in range(0, df1.shape[0]) :
    for col in participant_df:
      sums += df1.loc[row][col]
    for col in participant_df:
      df1.loc[row,col] = df1.loc[row][col]/sums
    sums = 0

  partner_df = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence',
    'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

  for row in range(0, df1.shape[0]) :
    for col in partner_df:
      sums += df1.loc[row][col]
    for col in partner_df:
      df1.loc[row,col] = df1.loc[row][col]/sums

    sums = 0

  for col in participant_df:
    print('Mean of '+ col + ':['+ str(round(df1[col].sum()/df1.shape[0],2))+'].')
  for col in partner_df:
    print('Mean of '+ col + ':['+ str(round(df1[col].sum()/df1.shape[0],2))+'].')
  
  df1.to_csv(filename2, index = False)
    
if __name__== "__main__":
  main()
