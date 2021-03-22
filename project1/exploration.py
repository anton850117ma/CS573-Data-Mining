import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

def to_print(data): #print 10 grayscale matrix

  for index in range(10):
    title = "class: "+ str(index)
    new_data = data[np.where(data[:,1] == index)]
    rows = np.size(new_data, 0)
    new_data[:,0] = np.arange(rows)
    selected = np.random.randint(0, rows, 1)
    chosed = new_data[selected[0],]
    matrix = np.reshape(chosed[2:], (-1, 28))
    plt.imshow(matrix, cmap="gray")
    plt.title(title)
    # plt.show()
    plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', str(index+10000)+'.png']))

def to_color(data):

  plt.clf()
  rows = np.size(data, 0)
  row_index = np.random.randint(0, rows, 1000)
  label = np.take(data[:,1], row_index)
  chosed_x = np.take(data[:,2], row_index)
  chosed_y = np.take(data[:,3], row_index)
  pic = plt.scatter(chosed_x, chosed_y, c = label)
  # plt.colorbar(pic)
  plt.title("1000 examples")
  # plt.show()
  plt.savefig(os.sep.join([os.path.expanduser('~'), 'Desktop', 'gg'+'.png']))
  
def main():

  np.random.seed(0)
  input_df = "digits-raw.csv"
  other_df = "digits-embedding.csv"
  df = pd.read_csv(input_df)
  df2 = pd.read_csv(other_df)
  data = df.values
  data2 = df2.values
  to_print(data)
  to_color(data2)

if __name__== "__main__":
  main()
