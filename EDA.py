import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize, sent_tokenize

path  = '/data/'

train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')


#target distribution
score_cols = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']

for i,col in enumerate(score_cols):
    fig = sns.countplot(data = train_df, x=col)
    plt.title(col + ' Scores Distribution')
    plt.show()
    
# Text EDA

##add columns to 'train_df' which calculates the length of full_text (as text_len) 
#and number of words in full_text (as text_word_count)
train_df['text_len'] = train_df['full_text'].astype(str).apply(len)

train_df["text_word_count"] = train_df["full_text"].apply(lambda x: len(x.replace('\n', ' ').split()))

## add number of sentences in full_text (as sent_count)
def get_sent_count(text):
    tokens = sent_tokenize(text, language='english')
    return len(tokens)

train_df['sent_count'] = train_df['full_text'].apply(get_sent_count)

cols = ['text_len', 'text_word_count', 'sent_count']

for i,col in enumerate(cols):
    fig = sns.histplot(data = train_df, x=col)
    plt.title(col + '  Distribution')
    plt.show()

#get the max, min and average sentence count:
print('The maximum number of sentences: {}'.format(train_df['sent_count'].max()))
print('The minimum number of sentences: {}'.format(train_df['sent_count'].min()))
print('The average number of sentences: {}'.format(train_df['sent_count'].mean()))

#get the max, min and average text lengths:
print('The maximum text length: {}'.format(train_df['text_len'].max()))
print('The minimum text length: {}'.format(train_df['text_len'].min()))
print('The average text length: {}'.format(train_df['text_len'].mean()))

#get the max and min word count:
print('The maximum number of words: {}'.format(train_df['text_word_count'].max()))
print('The minimum number of words: {}'.format(train_df['text_word_count'].min()))
print('The average number of words: {}'.format(train_df['text_word_count'].mean()))

# a look at the essay's scores with maximum words(3814F9116CD1):
train_df[train_df.text_word_count == train_df.text_word_count.max()]

#Essay EDA

# single sentence essays:
train_df[train_df.sent_count == train_df.sent_count.min()]

#Score vs Text length 

# lets find out how the total_score is affected by length of the full_text (text_len):
score_text_len = train_df.groupby('total_score')['text_len'].mean().sort_values()

#plot the graph for length of prediction string per type:
score_text_len.plot(kind = 'barh', figsize = (12,8))
plt.xlabel('Mean Text length')
plt.title(' Relationship between length of texts and scoring')

#Number of words vs Score

# lets find out how the total_score is affected by number of words in the full_text (text_word_count):
score_word_count = train_df.groupby('total_score')['text_word_count'].mean().sort_values()

#plot the graph for length of prediction string per type:
score_word_count.plot(kind = 'barh', figsize = (12,8))
plt.xlabel('Average Word_count')
plt.title(' Relationship between number of words and scoring')

# Number of sentences vs Score

# lets find out how the total_score is affected by number of words in the full_text (text_word_count):
score_sent_count = train_df.groupby('total_score')['sent_count'].mean().sort_values()

#plot the graph for length of prediction string per type:
score_sent_count.plot(kind = 'barh', figsize = (12,8))
plt.xlabel('Average Sent_count')
plt.title(' Relationship between number of sentences and scoring')

# Corrlattion between all 6 scoring 

plt.figure(figsize=(15,15))
colormap = sns.color_palette("Blues")
sns.heatmap(train_df.corr(), annot=True, cmap=colormap)