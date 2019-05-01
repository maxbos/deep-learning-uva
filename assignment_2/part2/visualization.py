import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

## Generative, default settings
gendefault = pd.read_csv('./out/eval_results_None.csv')
# Losses
ax = sns.lineplot(x='step', y='value', data=gendefault[gendefault['label'].isin(['loss'])])
ax.set(xlabel='step', ylabel='loss')
plt.savefig("plots/gendefault_loss.png", dpi=600, facecolor="w")
# Accuracy
ax = sns.lineplot(x='step', y='value', data=gendefault[gendefault['label'].isin(['accuracy'])])
ax.set(xlabel='step', ylabel='accuracy')
plt.savefig("plots/gendefault_acc.png", dpi=600, facecolor="w")
