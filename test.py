'''
'''
import svm
import numpy as np
import matplotlib.pyplot as plt

# 线性可分样本
sigma = np.array(([0.1, 0.01], [0.01, 0.1]))
r = np.linalg.cholesky(sigma)
miu0 = np.array(([4.0, 5.0]))
s0 = np.dot(np.random.randn(100, 2), r) + miu0
miu1 = np.array(([-1, -4.4]))
s1 = np.dot(np.random.randn(100, 2), r) + miu1
points = np.vstack((s0, s1))
y = np.zeros((200,), np.int)
for i in range(200):
    if i < 100:
        y[i] = 1
    else:
        y[i] = -1
plt.figure(1)
plt.scatter(s0[:, 0], s0[:, 1], c='r')
plt.scatter(s1[:, 0], s1[:, 1], c='g')
#plt.figure(2)
#plt.scatter(points[:, 0], points[:, 1])
#plt.show()
model = svm.SVM(max_iter=200, C=5)
W, b = model.fit(points, y)
kkt = model.kkt_condition(points, y)[0]
print(kkt)
print(np.sum(kkt))
x = [-5, 5]
y1 = [0, 0]
y2 = [0, 0]
y3 = [0, 0]
y1[0] = ((1 - b - W[0] * x[0]) / W[1])
y1[1] = ((1 - b - W[0] * x[1]) / W[1])
y2[0] = ((- b - W[0] * x[0]) / W[1])
y2[1] = ((- b - W[0] * x[1]) / W[1])
y3[0] = ((-1 - b - W[0] * x[0]) / W[1])
y3[1] = ((-1 - b - W[0] * x[1]) / W[1])
plt.plot(x, y1, 'b--')
plt.plot(x, y2, 'b')
plt.plot(x, y3, 'b--')
plt.show()

## 线性不可分样本
#N = 100 # number of points per class
#D = 2 # dimensionality
#K = 2 # number of classes
#X = np.zeros((N * K, D))
#y = np.zeros((N * K))
#for j in range(K):
#    ix = range(N * j, N * (j + 1))
#    r = j * 10 + 5
#    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
#    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
#    y[ix] = j
#y[y < 1] = -1
##plt.figure(0)
##plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
##plt.show()
#model = svm.SVM(max_iter=500, sigma=0.001, C=1.0, mode='gauss')
#model.fit(X, y)
#kkt = model.kkt_condition(X, y)[0]
#print(kkt)
#print(np.sum(kkt))
