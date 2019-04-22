import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

LOSS_LABELS = ['test loss', 'train loss']
ACC_LABELS = ['test accuracy', 'train accuracy']

sns.set_style("darkgrid")

## MLP numpy
mlpnumpy = pd.read_csv('./eval/mlp_numpy/1555869217.4908721.csv')
# Losses
ax = sns.lineplot(x='step', y='value', hue='label', data=mlpnumpy[mlpnumpy['label'].isin(LOSS_LABELS)])
plt.savefig("plots/mlp_numpy_loss.png", dpi=600, facecolor="w")
# Accuracy
ax = sns.lineplot(x='step', y='value', hue='label', data=mlpnumpy[mlpnumpy['label'].isin(ACC_LABELS)])
plt.savefig("plots/mlp_numpy_acc.png", dpi=600, facecolor="w")

## MLP numpy
mlpnumpy = pd.read_csv('./eval/mlp_pytorch/ly-200-200-100_lr-0.0002.csv')
# Losses
ax = sns.lineplot(x='step', y='value', hue='label', data=mlpnumpy[mlpnumpy['label'].isin(LOSS_LABELS)])
plt.savefig("plots/mlp_torch_loss_best.png", dpi=600, facecolor="w")
# Accuracy
ax = sns.lineplot(x='step', y='value', hue='label', data=mlpnumpy[mlpnumpy['label'].isin(ACC_LABELS)])
plt.savefig("plots/mlp_torch_acc_best.png", dpi=600, facecolor="w")
