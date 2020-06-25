import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# output the data set
path = r'C:\Users\JackyWang28\Desktop\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())

# draw the data graph
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()


