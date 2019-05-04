import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

## Generative, default settings
gendefault = pd.read_csv('./out/eval_results_None.csv')

loss_df = gendefault[gendefault['label'] == 'loss']
loss_df['actual step'] = (loss_df['Unnamed: 0']-1) / 2 * 5

accuracy_df = gendefault[gendefault['label'] == 'accuracy']
accuracy_df['actual step'] = accuracy_df['Unnamed: 0'] / 2 * 5

# Losses
ax = sns.lineplot(x='actual step', y='value', data=loss_df)
ax.set(xlabel='step', ylabel='loss')
plt.savefig("plots/gendefault_loss.png", dpi=100, facecolor="w")
# Accuracy
ax = sns.lineplot(x='actual step', y='value', data=accuracy_df)
ax.set(xlabel='step', ylabel='accuracy')
plt.savefig("plots/gendefault_acc.png", dpi=100, facecolor="w")
