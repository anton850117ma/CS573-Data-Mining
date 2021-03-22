import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import preprocess2 as ps

def int_j(df, lam, weight):

  dec = df['decision'].apply(lambda x: (-1,1)[x>0])
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys2 = temp_df.dot(weight).mul(dec)
  check = cal_ys2.apply(lambda f: (0,1)[f<1])
  new_dec = dec.mul(check).mul(-1)
  temp = new_dec.dot(temp_df).reset_index(drop = True).div(df.shape[0])
  wow = pd.Series(weight).mul(lam).div(df.shape[0])
  final = temp.add(wow)
  final.iloc[-1] -= wow.iloc[-1]
  return final

def test_s(df, weight):

  dec = df['decision'].apply(lambda x: (-1,1)[x>0])
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys = temp_df.dot(weight)
  result = cal_ys.apply(lambda y: (-1,1)[y>0])
  compare = result == dec
  final = compare.apply(lambda f: (0,1)[f])
  return final.sum()/df.shape[0]

def svm(df_tr, df_te):

  init = [0]*(df_tr.shape[1]-1)
  init_s = pd.Series(init)
  maxx = 500
  lam = 0.01
  step_size = 0.5
  threshold = 10**-6

  for i in range(0, maxx):
    change = int_j(df_tr, lam, init_s.to_list())
    temp = change.mul(step_size)
    if np.linalg.norm(temp) < threshold: break
    else: init_s = init_s.sub(temp)

  end2 = test_s(df_te, init_s.tolist())
  return end2

def det_j(df, lam, weight):

  dec = df['decision']
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys = np.exp(temp_df.dot(weight).mul(-1)).add(1).rdiv(1).sub(dec)
  result = cal_ys.dot(temp_df).reset_index(drop = True).div(df.shape[0])
  wow = pd.Series(weight).mul(lam)
  final = result.add(wow)
  final.iloc[-1] -= wow.iloc[-1]
  return final

def test(df, weight, lam):

  dec = df['decision']
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys = np.exp(temp_df.dot(weight).mul(-1)).add(1).rdiv(1)
  result = cal_ys.apply(lambda y: (0,1)[y>0.5])
  compare = result == dec
  final = compare.apply(lambda f: (0,1)[f])
  return final.sum()/df.shape[0]

def lr(df_tr, df_te):

  init = [0]*(df_tr.shape[1]-1)
  init_s = pd.Series(init)
  maxx = 500
  lam = 0.01
  step_size = 0.01
  threshold = 10**-6
  for i in range(0, maxx):
    change = det_j(df_tr, lam, init_s.tolist())
    temp = change.mul(step_size)
    if np.linalg.norm(temp) < threshold: break
    else: init_s = init_s.sub(temp)
  
  end2 = test(df_te, init_s.tolist(), lam)
  return end2

def nbc(df_tr, df_te):
  
  model_y = list()
  model_n = list()
  result = list()

  mask = df_tr['decision'] == 1
  df1 = df_tr[mask]
  df0 = df_tr[~mask]

  for col in df1:
    model_y.append(df1[col].value_counts())
    model_n.append(df0[col].value_counts())
  
  check = pd.Series(df_te['decision'].tolist())
  temp_df = df_te.drop(['decision'], axis = 1)
  const_y = df1.shape[0]/df_tr.shape[0]
  const_n = 1-const_y

  #trash from here

  for index, row in temp_df.iterrows():
    tmp1 = np.array(row)
    tmp2 = np.array(row)
    counter = 0
    for x in tmp1:
      if x in model_y[counter]:
        tmp1[counter] = model_y[counter][x]/df1.shape[0]
      else: tmp1[counter] = 0
      counter += 1
    counter = 0
    for y in tmp2:
      if y in model_n[counter]:
        tmp2[counter] = model_n[counter][y]/df0.shape[0]
      else: tmp2[counter] = 0
      counter += 1
    yes = np.prod(tmp1)/const_y
    no = np.prod(tmp2)/const_n
    if yes > no: result.append(1)
    else: result.append(0)
  
  final = pd.Series(result)
  compare = final == check
  end = compare.apply(lambda f: (0,1)[f])
  return end.sum()/df_te.shape[0]

def prep():

  filename = 'dating2.csv'
  filename2 = 'dating-binned.csv'
  orig_df = pd.read_csv(filename)

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
  for col in orig_df:
    if col in in_cor:
      bins = np.linspace(-1, 1, num = 6)
      orig_df[col] = np.searchsorted(bins, orig_df[col].values, side = 'left')
      mask = orig_df[col] == 0
      orig_df.loc[mask, col] = 1
    elif col in age:
      bins = np.linspace(18, 58, num = 6)
      orig_df[col] = np.searchsorted(bins, orig_df[col].values, side = 'left')
      mask = orig_df[col] == 0
      orig_df.loc[mask, col] = 1
    elif col in nothing:
      orig_df[col] = orig_df[col]
    elif col in pref_impor:
      bins = np.linspace(0, 1, num = 6)
      orig_df[col] = np.searchsorted(bins, orig_df[col].values, side = 'left')
      mask = orig_df[col] == 0
      orig_df.loc[mask, col] = 1
    else:
      bins = np.linspace(0, 10, num = 6)
      mask = orig_df[col] > 10
      orig_df.loc[mask, col] = 10
      orig_df[col] = np.searchsorted(bins, orig_df[col].values, side = 'left')
      mask = orig_df[col] == 0
      orig_df.loc[mask, col] = 1

  orig_df.to_csv(filename2, index = False)

def paint(avg_lr, avg_svm, avg_nbc, se_lr, se_svm, se_nbc, scale):

  plt.scatter(scale,avg_lr)
  plt.plot(scale, avg_lr, label = 'LR')
  plt.scatter(scale,avg_svm)
  plt.plot(scale, avg_svm, label = 'SVM')
  plt.scatter(scale,avg_nbc)
  plt.plot(scale, avg_nbc, label = 'NBC')
  plt.errorbar(scale, avg_lr, yerr = se_lr, fmt = 'none')
  plt.errorbar(scale, avg_svm, yerr = se_svm, fmt = 'none')
  plt.errorbar(scale, avg_nbc, yerr = se_nbc, fmt = 'none')
  plt.title('Accuracy over sizes')
  plt.xlabel("size")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

def to_train(df_list, num, df1, df_list2, df2):

  t_frac = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
  lr_list = list()
  svm_list = list()
  nbc_list = list()

  for frac in t_frac:
    tmp = df1
    tmp2 = df2
    tmp_lr = list()
    tmp_svm = list()
    tmp_nbc = list()
    for idx in range(num):
      test_set = df_list[idx]
      sc = tmp.drop(test_set.index)
      train_set = sc.sample(frac = frac, random_state = 32)
      tmp_lr.append(lr(train_set,test_set))
      tmp_svm.append(svm(train_set,test_set))
      
      test_set2 = df_list2[idx]
      sc2 = tmp2.drop(test_set2.index)
      train_set2 = sc2.sample(frac = frac, random_state = 32)
      tmp_nbc.append(nbc(train_set2, test_set2))

    lr_list.append(tmp_lr)
    svm_list.append(tmp_svm)
    nbc_list.append(tmp_nbc)

  np_lr = np.asarray(lr_list)
  np_svm = np.asarray(svm_list)
  np_nbc = np.asarray(nbc_list)
  avg_lr = np.mean(np_lr, axis = 1)
  avg_svm = np.mean(np_svm, axis = 1)
  avg_nbc = np.mean(np_nbc, axis = 1)
  se_lr = np.std(np_lr, axis = 1)/math.sqrt(num)
  se_svm = np.std(np_svm, axis = 1)/math.sqrt(num)
  se_nbc = np.std(np_nbc, axis = 1)/math.sqrt(num)
  size = df1.shape[0]/num*(num-1)
  scale = [x*size for x in t_frac]
  paint(avg_lr, avg_svm, avg_nbc, se_lr, se_svm, se_nbc, scale)

def main():

  # prep()
  filename = 'trainingSet.csv'
  filename2 = 'dating-binned.csv'
  df = pd.read_csv(filename)
  df2 = pd.read_csv(filename2)
  df2 = df2[:6500]
  tmp = df2.sample(frac = 0.2, random_state = 25)
  other_df2 = df2.drop(tmp.index).reset_index(drop = True)

  df['intercept'] = 1
  train = df.sample(frac = 1, random_state = 18).reset_index(drop = True)
  train2 = other_df2.sample(frac = 1, random_state = 18).reset_index(drop = True)
  tmp_df = train
  tmp_df2 = train2
  split = 10
  df_list = list()
  df_list2 = list()
  rows = int(train.shape[0]/split)
  for i in range(split):
    new_df = tmp_df.head(rows)
    new_df2 = tmp_df2.head(rows)
    df_list.append(new_df)
    df_list2.append(new_df2)
    tmp_df = tmp_df.drop(new_df.index)
    tmp_df2 = tmp_df2.drop(new_df2.index)
  
  to_train(df_list, split, train, df_list2, train2)

if __name__== "__main__":
  main()