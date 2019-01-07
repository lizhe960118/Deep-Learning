"""
synthetic minority oversampling technique
合成少数类过采样技术
1.对于少数类中的每一个样本x，以欧式距离为标准计算他到少数类样本集中所有其他样本的距离，得到其k近邻
2.根据样本不平衡比例设置一个采样比例，用来确定采样倍率N。
    对于每一个少数类样本x，从其k近邻中随机选择若干个样本xn
3.基于选择的样本xn，分别与原样本x按照公式
    x_new = x + rand(0,1) * (x_n - x) 来生成新样本。
此方法生成的新样本，在空间上的表示为原样本和其某一k近邻之间。
"""

# code:
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np 

class Smote:
    def __init__(self, samples, N=1, k=5):
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.new_index = 0

    def over_sampling(self):
        # N = int(self.N / 100)
        self.synthetic = np.zeros((self.n_samples, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors', neighbors)
        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance = False)[0]
            print(nnarray)
            self._populate(N, i, nnarray)
        return self.synthetic
        # for each minority class samples, choose N of the K nearest neighbors and
        # generate N synthetic samples

    def _populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k-1)
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            self.synthetic[self.new_index] = self.samples[i] + gap * dif
            self.new_index += 1

a = np.array([[1,2,3], [4,5,6], [2,3,1],[2,1,2],[2,3,4],[2,3,4]])
s = Smote(a, N=2)
print(s.over_sampling())

