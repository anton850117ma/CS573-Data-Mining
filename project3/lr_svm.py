import numpy as np
import pandas as pd


def det_j(df, lam, weight):

  dec = df['decision']
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys = np.exp(temp_df.dot(weight).mul(-1)).add(1).rdiv(1).sub(dec)
  result = cal_ys.dot(temp_df).reset_index(drop = True).div(df.shape[0])
  wow = pd.Series(weight).mul(lam)
  final = result.add(wow)
  final.iloc[-1] -= wow.iloc[-1]
  # print(result)
  return final


def test(df, weight, lam):

  dec = df['decision']
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys = np.exp(temp_df.dot(weight).mul(-1)).add(1).rdiv(1)
  # print(cal_ys)
  result = cal_ys.apply(lambda y: (0,1)[y>0.5]) #not sure equal
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
  summ = 0
  # print(threshold)
  for i in range(0, maxx):
    change = det_j(df_tr, lam, init_s.tolist())
    temp = change.mul(step_size)
    if np.linalg.norm(temp) < threshold: break
    else:
      summ += 1
      # print(summ)
      init_s = init_s.sub(temp)

  # print(init_s)
  end1 = test(df_tr, init_s.tolist(), lam)
  print('Training Accuracy LR:', round(end1,2))
  end2 = test(df_te, init_s.tolist(), lam)
  print('Testing Accuracy LR:', round(end2,2))

def int_j(df, lam, weight):

  
  dec = df['decision'].apply(lambda x: (-1,1)[x>0])
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys2 = temp_df.dot(weight).mul(dec)
  check = cal_ys2.apply(lambda f: (0,1)[f<1])

  new_dec = dec.mul(check).mul(-1)
  temp = new_dec.dot(temp_df).reset_index(drop = True).div(df.shape[0])
  wow = pd.Series(weight).mul(lam).div(df.shape[0])
  # print(wow)
  final = temp.add(wow)
  final.iloc[-1] -= wow.iloc[-1]
  # print(final)
  return final

def test_s(df, weight):

  dec = df['decision'].apply(lambda x: (-1,1)[x>0])
  temp_df = df.drop(['decision'], axis = 1)
  cal_ys = temp_df.dot(weight)
  result = cal_ys.apply(lambda y: (-1,1)[y>0]) #not sure equal
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
  summ = 0

  for i in range(0, maxx):
    change = int_j(df_tr, lam, init_s.to_list())
    temp = change.mul(step_size)
    # print(temp)
    if np.linalg.norm(temp) < threshold: break
    else:
      summ += 1
      # print(summ)
      init_s = init_s.sub(temp)
  
  end1 = test_s(df_tr, init_s.tolist())
  print('Training Accuracy SVM:', round(end1,2))
  end2 = test_s(df_te, init_s.tolist())
  print('Testing Accuracy SVM:', round(end2,2))


def main():
    
  filename1 = 'trainingSet.csv'
  filename2 = 'testSet.csv'

  df1 = pd.read_csv(filename1)
  df2 = pd.read_csv(filename2)
  df1['intercept'] = 1
  df2['intercept'] = 1
  lr(df1, df2)
  svm(df1, df2)


if __name__== "__main__":
  main()