import numpy as np
import pandas as pd


def main():

  filename = 'dating-binned.csv'
  filename2 = 'trainingSet.csv'
  filename3 = 'testSet.csv'
  orig_df = pd.read_csv(filename)

  test = orig_df.sample(frac = 0.2, random_state = 47)
  train = orig_df.drop(test.index)
  
  train.to_csv(filename2, index = False)
  test.to_csv(filename3, index = False)
  
if __name__== "__main__":
  main()