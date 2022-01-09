# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
plt.figure(figsize=(10,10))
df = pd.read_csv("yourdata.csv", header = None)
x = df.to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(x)
pca = PCA(2)
data = pca.fit_transform(X)
for k in [3,5]:
    model = KMeans(n_clusters = k, init = "random")
    label = model.fit_predict(data)
    centers = np.array(model.cluster_centers_)
    print(centers)
    axis1 = plt.subplot(2, 1, 1)
    uniq = np.unique(label)
    for i in uniq:
        axis1.scatter(data[label == i , 0] , data[label == i , 1] ,s=10, label = i ,   marker= 10)
    axis1.scatter(centers[:,0], centers[:,1], marker="x",s=20, color='k')
    axis1.legend()
    axis1.set_title("random initialization")
    model = KMeans(n_clusters = k, init = "k-means++")
    label = model.fit_predict(data)
    centers = np.array(model.cluster_centers_)
    print(centers)
    axis2 = plt.subplot(2, 1,  2)
    uniq = np.unique(label)
    for i in uniq:
        axis2.scatter(data[label == i , 0] , data[label == i , 1] ,s=10, label = i ,  marker= 10)
    axis2.scatter(centers[:,0], centers[:,1], marker="x",s=20, color='k')
    axis2.legend()
    axis2.set_title("k means ++ initialization")
    plt.show()
