import pandas as pd
import numpy as np
import os
import sys
import time
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, fclusterdata, maxdists
import matplotlib.pyplot as plt

def shan_entropy(c):
  c_normalized = c / float(np.sum(c))
  c_normalized = c_normalized[np.nonzero(c_normalized)]
  H = -sum(c_normalized* np.log2(c_normalized))  
  return H

def NMI(label_true, label_pred, Ks): 
  
  H_x = 0
  for K in range(Ks):
    if np.sum(label_pred == K) > 0:
      p = np.mean(label_pred == K)
      H_x += -p * np.log2(p)

  H_y = 0
  for K in range(Ks):
    if np.sum(label_true == K) > 0:
      p = np.mean(label_true == K)
      H_y += -p * np.log2(p)

  I_xy = 0
  for K in range(Ks):
    if np.sum(label_pred == K) > 0:
      for label in range(10):
        if np.sum(label_true == label) > 0:
          p_xy = np.mean((label_pred == K) & (label_true == label))
          if p_xy > 0:
            p_x = np.mean(label_pred == K)
            p_y = np.mean(label_true == label)
            I_xy += p_xy * np.log2(p_xy/(p_x*p_y))

  NMI = I_xy/(H_x + H_y)
  return NMI

def SC(data, centers, K):

  sum_s = np.array([0])
  for center in range(K):
    in_c = data[np.where(data[:,0] == center)]
    out_c = data[np.where(data[:,0] != center)]
    in_cc = np.delete(in_c, 0, 1)
    out_cc = np.delete(out_c, 0, 1)
    size_in = np.size(in_cc, 0)
    size_out = np.size(out_cc, 0)
    if size_in == 1: continue
    #A
    new_in = np.repeat(in_cc, size_in, 0)
    new_inin = np.tile(in_cc, (size_in, 1))
    self_re = np.linalg.norm(new_in - new_inin, axis = 1)
    new_self = np.sum(np.reshape(self_re, (-1, size_in))/size_in, axis = 1)
    #B
    new_out = np.repeat(in_cc, size_out, 0)
    new_outt = np.tile(out_cc, (size_in, 1))
    other_re = np.linalg.norm(new_out - new_outt, axis = 1)
    new_other = np.sum(np.reshape(other_re, (-1, size_out))/size_out, axis = 1)
    #S
    sis = (new_other-new_self)/np.maximum(new_self, new_other)
    sum_s = np.concatenate((sum_s, sis), axis=None)
  
  final = np.sum(sum_s)/(sum_s.size-1)
  return final
  
def WC_SSD(data, centers, K):
  
  total = 0
  for center in range(K):
    nodes = data[np.where(data[:,0] == center)]
    pro_nodes = np.delete(nodes, 0, 1)
    dists = np.linalg.norm(pro_nodes-centers[center], axis = 1)
    sub_sum = np.dot(dists, dists.T)
    total = total + sub_sum
  return total

def classify(data, centers, it, K):

  sizes = np.size(data, 0)
  new_data = np.repeat(data, K, axis = 0)
  new_cent = np.tile(centers, (sizes, 1))
  result = np.linalg.norm(new_data-new_cent, axis = 1)
  new_result = np.reshape(result, (-1, K))
  mins_pos = np.argmin(new_result, axis = 1) #each node's new classification
  matrix = np.reshape(mins_pos, (1,sizes)) #to matrix
  labeled_data = np.concatenate((matrix.T, data), axis = 1)
  # print(labeled_data)
  if it > 0:
    for center in range(K):
      nodes = labeled_data[np.where(labeled_data[:,0] == center)]
      if np.size(nodes,0) == 0: continue
      cut = np.delete(nodes, 0, 1)
      new_center = np.mean(cut, axis = 0)
      centers[center] = new_center
    return classify(data, centers, it-1, K)
  else:
    return labeled_data, centers

def k_means(data, K, max_it, dist):
  
  N = np.size(data, 0)
  rows = np.random.randint(0, N, size=K)  #get K random centers
  x = np.take(data[:,2], rows)
  y = np.take(data[:,3], rows)
  centers = np.column_stack((x,y))  #construct K centers
  new_data = data[:,2:]
  classfied, new_centers = classify(new_data, centers, max_it-1, K)
  total_WC = WC_SSD(classfied, new_centers, K)
  total_SC = SC(classfied, new_centers, K)
  total_NMI = NMI(data[:,1], classfied[:,0], K)
  
  return total_WC, total_SC, total_NMI

def painter(WC, SC, K, num):

  fig, ax = plt.subplots()
  ax.scatter(SC, WC)
  ax.plot(SC,WC)
  
  for i, txt in enumerate(K):
    ax.annotate(txt, (SC[i], WC[i]))
  
  plt.xlabel('SC')
  plt.ylabel('WC-SSD')
  plt.title('Datasets'+str(num))
  plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', str(num)+'.png']))
  # plt.show()

def paint(data, y, K, num, cases):

  plt.scatter(K, data)
  plt.plot(K,data)
  plt.xlabel('Number of K')
  plt.ylabel(y)
  plt.title(y +' of Dataset '+str(cases))
  plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', str(num) +'.png']))
  plt.clf()

def miner1(data, Ks, num, check):

  wcs = np.zeros(Ks.size)
  scs = np.zeros(Ks.size)
  it = 0
  for K in Ks:
    WC, SC, NMI = k_means(data, K, 50, 'Euclidean')
    wcs[it] = WC
    scs[it] = SC
    it += 1
  if check: 
    paint(wcs, 'WC', Ks, num, num)
    paint(scs, 'SC', Ks, num+3, num)
  
  return wcs, scs

def deal_with_2(WC, SC, Ks):

  #Find Knee of Curve
  values = list(WC)
  nPoints = len(values)
  allCoord = np.vstack((range(nPoints), values)).T
  firstPoint = allCoord[0]
  lineVec = allCoord[-1] - allCoord[0]
  lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
  vecFromFirst = allCoord - firstPoint
  scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
  vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
  vecToLine = vecFromFirst - vecFromFirstParallel
  distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
  idxOfBestPoint = np.argmax(distToLine)
  WC_best = Ks[idxOfBestPoint]

  #Find Max of Curve
  otherBest = np.argmax(SC)
  SC_best = Ks[otherBest]

  if WC_best == SC_best:
    return SC_best
  else:
    return WC_best

def deal_with_1(data, data2, data3, Ks, item):
  
  np.random.seed(item)
  wcs, scs = miner1(data, Ks, 1, True)
  K1 = deal_with_2(wcs, scs, Ks)
  wcs2, scs2 = miner1(data2, Ks, 2, True)
  K2 = deal_with_2(wcs2, scs2, Ks)
  wcs3, scs3 = miner1(data3, Ks, 3, True)
  K3 = deal_with_2(wcs3, scs3, Ks)
  return K1, K2, K3

def paint_STD(data, std, y, K, num, cases):

  plt.scatter(K, data, color = 'blue')
  plt.plot(K,data, color = 'blue')
  plt.errorbar(K, data, yerr = std, color = 'blue')
  plt.xlabel('Number of K')
  plt.ylabel(y)
  plt.title(y +' of Dataset '+str(cases))
  plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', str(num) +'.png']))
  plt.clf()

def minor3(data, Ks, seeds, num):

  re_wc = np.zeros((Ks.size,seeds))
  re_sc = np.zeros((Ks.size,seeds))

  ix = iy = 0
  for K in Ks:
    for item in range(seeds):
      np.random.seed(item+1)
      WC, SC, NMI = k_means(data, K, 50, 'Euclidean')
      re_wc[ix,iy] = WC
      re_sc[ix,iy] = SC
      iy += 1
    iy = 0
    ix += 1
  
  mean_wc = np.mean(re_wc, axis = 1)
  mean_sc = np.mean(re_sc, axis = 1)
  std_wc = np.std(re_wc, axis = 1)
  std_sc = np.std(re_sc, axis = 1)
  paint_STD(mean_wc, std_wc, 'Mean and Std of WC', Ks, 1+num*6, num)
  paint_STD(mean_sc, std_sc, 'Mean and Std of SC', Ks, 2+num*6, num)
  # paint(std_wc, 'Std of WC', Ks, 3+num*6, num)
  # paint(std_sc, 'Std of SC', Ks, 4+num*6, num)
  # return mean_wc, mean_sc, std_wc, std_sc

def deal_with_3(data, data2, data3, Ks, seed):
  
  minor3(data, Ks, seed, 1)
  minor3(data2, Ks, seed, 2)
  minor3(data3, Ks, seed, 3)

def add_k_means(data, K, max_it, dist):
  
  N = np.size(data, 0)
  rows = np.random.randint(0, N, size=K)  #get K random centers
  x = np.take(data[:,2], rows)
  y = np.take(data[:,3], rows)
  centers = np.column_stack((x,y))  #construct K centers
  new_data = data[:,2:]
  classfied, new_centers = classify(new_data, centers, max_it-1, K)
  total_NMI = NMI(data[:,1], classfied[:,0], K)
  
  return classfied[:,0], total_NMI

def minor4(data, K, num):

  clusters, NMI = add_k_means(data, K, 50, 'Euclidean')
  rows = np.size(data, 0)
  row_index = np.random.randint(0, rows, 1000)
  label = np.take(clusters, row_index)
  chosed_x = np.take(data[:,2], row_index)
  chosed_y = np.take(data[:,3], row_index)
  pic = plt.scatter(chosed_x, chosed_y, c = label)
  plt.title('Dataset:' + str(num) + ', NMI:'+ str(round(NMI, 3)))
  plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', str(num+100) +'.png']))
  plt.clf()

def deal_with_4(data, data2, data3, item, k1, k2, k3):

  np.random.seed(item)
  minor4(data, k1, 1)
  minor4(data2, k2, 2)
  minor4(data3, k3, 3)

def analysis(data, K):

  #2.1
  np.random.seed(0)
  WC, SC, NMI = k_means(data, K, 50, 'Euclidean')
  print('WC-SSD:',WC)
  print('SC:',SC)
  print('NMI:',NMI)

  # #2.2~2.4
  Ks = np.array([2, 4, 8, 16, 32])
  data2 = data[np.isin(data[:,1], np.array([2,4,6,7]))]
  data3 = data2[np.isin(data2[:,1], np.array([6,7]))]

  k1, k2, k3 = deal_with_1(data, data2, data3, Ks, 0)

  print('Chosen K for Dataset1 using K-means:', k1)
  print('Chosen K for Dataset2 using K-means:', k2)
  print('Chosen K for Dataset3 using K-means:', k3)

  deal_with_3(data, data2, data3, Ks, 10)
  deal_with_4(data, data2, data3, 0, k1, k2, k3)

def deal_with_hc(data, link_type, num):
  
  matrix = linkage(data, link_type)
  fig = plt.figure(figsize=(25, 10))
  dn = dendrogram(matrix)
  plt.title('Dataset 1 with '+ link_type +' linkage')
  plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', str(num)+'.png']))
  plt.clf()
  return matrix

def HC_paint(data, K, y, types, num):

  plt.scatter(K, data)
  plt.plot(K,data)
  plt.xlabel('Number of K')
  plt.ylabel(y)
  plt.title(y + ' of '+ types + ' linkage')
  plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', str(num) +'.png']))
  plt.clf()  

def calc_WC_SC(matrix, group, Ks, types, num):

  scale = np.size(group, 0)
  re_wc = np.zeros(Ks.size)
  re_sc = np.zeros(Ks.size)
  it = 0

  for K in Ks:
    total_WC = 0
    total_SC = 0
    tmp_label = fcluster(matrix, t=K, criterion='maxclust')
    for diff in range(K):
      data = group[tmp_label == diff+1]
      if data.size == 0: continue
      center = np.mean(data, axis = 0)
      total_WC += np.sum(np.square(data-center))
    
    for each in range(scale):
      center = tmp_label[each]
      in_A = group[tmp_label == center]
      in_B = group[tmp_label != center]
      A = np.linalg.norm(group[each]-in_A)
      B = np.linalg.norm(group[each]-in_B)
      SC = (B-A)/np.max([A,B]) #sometimes B-A < 0
      if np.size(in_A, 0) == 1: SC = 0
      total_SC += SC

    re_wc[it] = total_WC
    re_sc[it] = total_SC/scale
    it += 1
  
  HC_paint(re_wc, Ks, 'WC', types, num*100)
  HC_paint(re_sc, Ks, 'SC', types, num*101)
  return re_wc, re_sc

def preprocessed(data):

  np.random.seed(0)
  group = np.array([[0,0]])
  for index in range(10):
    new_data = data[np.where(data[:,1] == index)]
    rows = np.size(new_data, 0)
    new_data[:,0] = np.arange(rows)
    selected = np.random.randint(0, rows, 10)
    chosed = new_data[selected,2:]
    group = np.concatenate((group, chosed), axis=0)

  sub_group = np.delete(group, np.array([[0,0]]), axis = 0)
  centers = np.repeat(np.arange(10), 10)
  return sub_group, centers

def calc_NMI(matrix, K, labels, num):

  pred_label = fcluster(matrix, t=K, criterion='maxclust')
  result = NMI(labels, pred_label-1, K)
  return result

def Hi_Clustering(data):

  #3.1~3.2
  group, centers = preprocessed(data)
  matrix_s = deal_with_hc(group, 'single', 1000)
  matrix_c = deal_with_hc(group, 'complete', 1001)
  matrix_a = deal_with_hc(group, 'average', 1002)

  #3.3
  Ks = np.arange(2,101)
  wc1, sc1 = calc_WC_SC(matrix_s, group, Ks, 'single', 11)
  wc2, sc2 = calc_WC_SC(matrix_c, group, Ks, 'complete', 12)
  wc3, sc3 = calc_WC_SC(matrix_a, group, Ks, 'average', 13)
  #3.4
  k1 = deal_with_2(wc1, sc1, Ks)
  k2 = deal_with_2(wc2, sc2, Ks)
  k3 = deal_with_2(wc3, sc3, Ks)

  print('Chosen K for Dataset1 using single linkage:', k1)
  print('Chosen K for Dataset1 using complete linkage:', k2)
  print('Chosen K for Dataset1 using average linkage:', k3)

  #3.5
  NMI_1 = calc_NMI(matrix_s, k1, centers, 1)
  NMI_2 = calc_NMI(matrix_c, k2, centers, 2)
  NMI_3 = calc_NMI(matrix_a, k3, centers, 3)

  print('NMI of '+str(k1)+' for Dataset1 using single linkage:', NMI_1)
  print('NMI of '+str(k2)+' for Dataset1 using complete linkage:', NMI_2)
  print('NMI of '+str(k3)+' for Dataset1 using average linkage:', NMI_3)

def main():

  input_df = sys.argv[1]
  input_k = int(sys.argv[2])
  df = pd.read_csv(input_df)
  data = df.values
  start = time.time()
  analysis(data, input_k)
  Hi_Clustering(data)
  end = time.time()
  print('time:', end - start)
  
if __name__== "__main__":

  main()