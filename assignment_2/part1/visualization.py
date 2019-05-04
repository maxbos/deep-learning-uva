import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

#list the files
# filelist = os.listdir('./out/out') 
# #read them into pandas
# df_list = [pd.read_table(f'./out/out/{f}') for f in filelist]
# #concatenate them together
# big_df = pd.concat(df_list)
# big_df.to_csv('./out/complete.csv')

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))

big_df = pd.read_csv('./out/complete.csv')

# Solely RNN accuracy curve over palindrome lengths
ax = sns.lineplot(x='palindrome length', y='accuracy', data=big_df[big_df['model type'] == ' RNN'], color="#3498db")
plt.savefig('plots/rnn.png', dpi=100, facecolor='w')

# RNN and LSTM accuracy curve over palindrome lengths
ax = sns.lineplot(x='palindrome length', y='accuracy', hue='model type', data=big_df, palette=["#9b59b6", "#3498db"])
plt.savefig('plots/lstm_rnn.png', dpi=100, facecolor='w')

# Losses
ax = sns.lineplot(x='step', y='value', data=gendefault[gendefault['label'].isin(['loss'])])
ax.set(xlabel='step', ylabel='loss')
plt.savefig("plots/gendefault_loss.png", dpi=600, facecolor="w")
# Accuracy
ax = sns.lineplot(x='step', y='value', data=gendefault[gendefault['label'].isin(['accuracy'])])
ax.set(xlabel='step', ylabel='accuracy')
plt.savefig("plots/gendefault_acc.png", dpi=600, facecolor="w")
