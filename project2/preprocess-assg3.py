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

def dummies(name, orig_df):

  check = orig_df[name].value_counts().sort_index().keys().tolist()
  dummy = pd.get_dummies(check)
  tmp_dum = pd.get_dummies(orig_df[name], prefix = name)
  temp = tmp_dum[tmp_dum.columns[:-1]]
  orig_df = orig_df.join(temp).drop(columns = name)
  
  # orig_df = orig_df.join(tmp_dum, rsuffix = '_r')
  # lis = orig_df[name].value_counts().sort_index().keys().tolist()
  
  # new_df = pd.DataFrame(columns=lis)
  # for row in orig_df[name]:
  #   df_temp = pd.DataFrame([dummy[check[row]].tolist()], columns=lis)
  #   new_df = new_df.append(df_temp, ignore_index=True)

  # re_df = new_df[new_df.columns[:-1]]
  # orig_df = orig_df.drop(columns = [name])
  # if(name == 'race_o'): orig_df = orig_df.join(re_df, rsuffix = '_o')
  # else: orig_df = orig_df.join(re_df, rsuffix = '_r')
  
  output = [0]*(len(check))
  if(name =='gender'):
    output[check.index('female')] = 1
    print('Mapped vector for female in column gender:' + str(output[:-1]))
  elif(name =='race'):
    output[check.index('Black/African American')] = 1
    print('Mapped vector for Black/African American in column race:'+ str(output[:-1]))
  elif(name =='race_o'):
    output[check.index('Other')] = 1
    print('Mapped vector for Other in column race_o:'+ str(output[:-1]))
  elif(name =='field'):
    output[check.index('economics')] = 1
    print('Mapped vector for economics in column field:'+ str(output[:-1]))

  return orig_df


def main():
    
  filename = 'dating-full.csv'
  filename1 = 'dating.csv'
  filename2 = 'trainingSet.csv'
  filename3 = 'testSet.csv'
  df1 = pd.read_csv(filename)
  df2 = pd.read_csv(filename)

  df1 = df1[:6500]
  df2 = df2[:6500]

  #(i)  replace '\''  
  df1['race'] = df1['race'].str.replace('\'', '')
  df1['race_o'] = df1['race_o'].str.replace('\'', '')
  df1['field'] = df1['field'].str.replace('\'', '')
  total = compare_two_dfs(df1,df2)
  # print('Quotes removed from ['+ str(total) +'] cells.')
  df2 = df1

  #(ii) convert to lowercase  
  df = df1['field'].map(lambda x: 0 if x.islower() else 1)
  df1['field'] = df1['field'].str.lower()
  # print('Standardized ['+ str(df.sum()) +'] cells to lower case.')

  #(iii)  sort and map to number 
  # df1['gender'] = change_to_num(df1,'gender',df2)
  # mod_df1 = dummies(df1['gender'])
  # df1 = df1.drop(columns = ['gender'])
  # df1 = df1.join(mod_df1)
  mod_df = dummies('gender',df1)
  mod_df = dummies('race',mod_df)
  mod_df = dummies('race_o',mod_df)
  mod_df = dummies('field',mod_df)
  
  # print(df1['Asian/Pacific Islander/Asian-American'])
  # df1['race_o'] = change_to_num(df1,'race_o',df2)
  # df1['field'] = change_to_num(df1,'field',df2)

  # check = df1['race'].value_counts().sort_index()
  # print(check)
  # print(pd.get_dummies(check))

  # (iv) mean values
  participant_df  = [
    'attractive_important', 'sincere_important', 'intelligence_important',
    'funny_important', 'ambition_important', 'shared_interests_important']
  
  sums = 0
  for row in range(0, mod_df.shape[0]) :
    for col in participant_df:
      sums += mod_df.loc[row][col]
    for col in participant_df:
      mod_df.loc[row,col] = mod_df.loc[row][col]/sums
    sums = 0

  partner_df = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence',
    'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

  for row in range(0, mod_df.shape[0]) :
    for col in partner_df:
      sums += mod_df.loc[row][col]
    for col in partner_df:
      mod_df.loc[row,col] = mod_df.loc[row][col]/sums
    sums = 0
  
  # for col in participant_df:
  #   print('Mean of '+ col + ':['+ str(round(df1[col].sum()/df1.shape[0],2))+'].')
  # for col in partner_df:
  #   print('Mean of '+ col + ':['+ str(round(df1[col].sum()/df1.shape[0],2))+'].')

  mod_df.to_csv(filename1, index = False)

  test = mod_df.sample(frac = 0.2, random_state = 25)
  train = mod_df.drop(test.index)
  train.to_csv(filename2, index = False)
  test.to_csv(filename3, index = False)
  

if __name__== "__main__":
  main()
