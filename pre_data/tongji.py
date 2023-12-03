import matplotlib.pyplot as plt
import pandas as pd

def my_plot(dataset_name):
    data=pd.read_csv('./'+dataset_name+'_in_degree.csv',header=None)
    data.columns = ['A']
    group = data.groupby(['A'])
    d=group.size()
    plt.figure()
    plt.scatter(d.index.tolist(),d.to_numpy())
    plt.title(dataset_name)
my_plot('reddit')
my_plot('citeseer')
my_plot('cora')
plt.show()