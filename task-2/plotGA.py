import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('GARunsFinal.csv', index_col=0, header=0)

plt.plot(df)
plt.legend(df)
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.savefig('GA_diff_params.png')
