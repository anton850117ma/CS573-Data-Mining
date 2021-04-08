import pandas as pd
import numpy as np
import time
import random
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, Reader, Dataset, accuracy, dump
from surprise.model_selection import cross_validate, KFold, GridSearchCV
from plot import bar, plot

# find the best sim_option and k for each KNN model
def find_opt_and_k(data, model, measure, scale, cv, ks, opts):

  param_grid = {'k': ks, 'sim_options': {'name': opts, 'user_based': [False]}}
  gs = None

  if model == 'Basic':
    gs = GridSearchCV(KNNBasic, param_grid, measures=measure, n_jobs=-1, pre_dispatch='1*n_jobs', cv=cv)
  elif model == 'WithMeans':
    gs = GridSearchCV(KNNWithMeans, param_grid, measures=measure, n_jobs=-1, pre_dispatch='1*n_jobs', cv=cv)
  elif model == 'WithZScore':
    gs = GridSearchCV(KNNWithZScore, param_grid, measures=measure, n_jobs=-1, pre_dispatch='1*n_jobs', cv=cv)
  elif model == 'Baseline':
    gs = GridSearchCV(KNNBaseline, param_grid, measures=measure, n_jobs=-1, pre_dispatch='1*n_jobs', cv=cv)

  gs.fit(data)
  df = pd.DataFrame.from_dict(gs.cv_results)
  param = gs.best_params['mae']
  best_k = param['k']
  tmp_param = param['sim_options']

  return df, best_k, tmp_param['name'], gs.best_score['mae']

def to_plot(data, measure, opts, ks, model):

  cos = data[0,:]
  msd = data[1,:]
  pers = data[2,:]
  perb = data[3,:]
  name = model+'_'+measure
  title = measure+' of KNN-'+model+' over K'
  plot(name+'.eps', ks, cos, msd, pers, perb, figsize=None, x_label='K', y_label=measure, legend=opts, legend_loc=None, title=title)
  # bar(name+'.eps', cos, msd, pers, perb, x_ticks=ks, total_width=0.8, x_label='K', y_label=measure, legend=opts, legend_loc='lower right', title=title)

def to_test(k, option, model):
  
  df = pd.read_csv('training_set.dat')
  test_df = pd.read_csv('test_set.dat')
  reader = Reader(rating_scale=(1, 5))
  trainingSet = Dataset.load_from_df(df, reader).build_full_trainset()
  testSet = Dataset.load_from_df(test_df, reader).build_full_trainset().build_testset()

  opt = {'name': option, 'user_based': False}

  
  if model == 'Basic':
    algo = KNNBasic(k = k,sim_options = opt)
    algo.fit(trainingSet)
    # dump.dump("KNNBS.model", algo=algo, verbose=1)
  elif model == 'WithMeans':
    algo = KNNWithMeans(k = k,sim_options = opt)
    algo.fit(trainingSet)
    # dump.dump("KNNWM.model", algo=algo, verbose=1)
  elif model == 'WithZScore':
    algo = KNNWithZScore(k = k,sim_options = opt)
    algo.fit(trainingSet)
    # dump.dump("KNNWZS.model", algo=algo, verbose=1)
  elif model == 'Baseline':
    algo = KNNBaseline(k = k,sim_options = opt)
    algo.fit(trainingSet)
    # dump.dump("KNNBSL.model", algo=algo, verbose=1)
  

def main():

  start = time.time()
  random.seed(0)
  input_file = 'training_set.dat'
  training_df = pd.read_csv(input_file)
  # data = np.loadtxt(input_file, dtype=int, delimiter=',')
  data = training_df.values
  scale = np.size(data, 0)
  df = pd.DataFrame(data, columns=['id', 'movie', 'rating'])
  reader = Reader(rating_scale=(1, 5))
  trainingSet = Dataset.load_from_df(df[['id', 'movie', 'rating']], reader)
  knn_list = ['Basic', 'WithMeans', 'WithZScore', 'Baseline']
  opt_list = ['cosine', 'msd', 'pearson', 'pearson_baseline']
  kss = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
  ks = np.arange(20)+1
  
  para_list = []

  for i in range(len(knn_list)):
    results_df, best_k, best_opt, best_score = find_opt_and_k(trainingSet, knn_list[i], ['rmse','mae'], scale, 10, ks, opt_list)
    # build model
    para_list.append([best_score, best_k, best_opt])
    # to_save(best_k, best_opt, knn_list[i])

    # plot figure
    rmse_df = results_df[['mean_test_rmse', 'mean_test_mae', 'param_k', 'param_sim_options']]
    rmse = np.zeros((len(opt_list), ks.size))
    mae = np.zeros((len(opt_list), ks.size))
    
    for index, row in rmse_df.iterrows():
      option = row['param_sim_options']
      if option['name'] == 'cosine':
        rmse[0, np.where(ks == row['param_k'])] = row['mean_test_rmse']
        mae[0, np.where(ks == row['param_k'])] = row['mean_test_mae']
      elif option['name'] == 'msd':
        rmse[1, np.where(ks == row['param_k'])] = row['mean_test_rmse']
        mae[1, np.where(ks == row['param_k'])] = row['mean_test_mae']
      elif option['name'] == 'pearson':
        rmse[2, np.where(ks == row['param_k'])] = row['mean_test_rmse']
        mae[2, np.where(ks == row['param_k'])] = row['mean_test_mae']
      elif option['name'] == 'pearson_baseline':
        rmse[3, np.where(ks == row['param_k'])] = row['mean_test_rmse']
        mae[3, np.where(ks == row['param_k'])] = row['mean_test_mae']

    to_plot(rmse, 'RMSE', opt_list, ks, knn_list[i])
    to_plot(mae, 'MAE', opt_list, ks, knn_list[i])
    
  print(para_list)
  print(time.time()-start)

if __name__ == "__main__":
  main()