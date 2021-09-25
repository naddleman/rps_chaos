"""Produces a grid of scatter plots examining the parameter sweep data for
chaotic solutions"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('chaos_now_please.csv')

fig, axs = plt.subplots(5,3)
#fig.tight_layout()
xvals = np.unique(df['x'].values)

for i, ax in enumerate(axs.flat):
    xval = xvals[i]
    current_df = df.loc[df['x'] == xval]
    probable_chaos = current_df.loc[current_df['lyap_1'] >= 0.02]
    no_chaos = current_df.loc[current_df['lyap_1'] < 0.02]
    ax.plot(no_chaos['y'], no_chaos['z'], 'b.')
    ax.plot(probable_chaos['y'], probable_chaos['z'], 'r.')
    title = 'a = ' + str(xval)
    ax.set_title(title)
    ax.set(xlabel='b', ylabel='c')

for ax in axs.flat:
    ax.label_outer()

plt.show()
