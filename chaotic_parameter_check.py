import numpy as np
import pandas as pd
import sys

args = sys.argv
if len(args) != 2:
    print("usage: python read_data.py xxx.csv")
    sys.exit()

filename = args[1]
df = pd.read_csv(filename)

probable_chaos = df.loc[df['lyap_1'] >= 0.05]
