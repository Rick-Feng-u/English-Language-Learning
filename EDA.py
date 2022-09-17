import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path  = '/data/'

train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')

score_cols = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']

for i,col in enumerate(score_cols):
    fig = sns.countplot(data = train_df, x=col)
    plt.title(col + ' Scores Distribution')
    plt.show()