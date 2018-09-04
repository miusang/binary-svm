'''
binary SVM
@author : ning
'''
import numpy as np

class SVM(object):

    def __init__(self, max_iter=100, C=2.0, sigma=0.1, mode='linear', 
            step=1e-3):
        '''the parameters initialize
        Parameters
        ----------
        max_iter : int, the maxmimum iter count.

        C : float, the penalty factor.
            the size of this factor reflects the concern about 
            the wrong classification of the model.

        mode : {'linear', 'gauss'}
               select the kernel function to be used in the binnary SVM model.

        sigma : float, the standard deviation of Gauss kernel function.
                sigma越小,升维越高
        '''
        self.max_iter = max_iter
        self.C = C
        self.tol = 1e-3
        self.epsilon = 1e-8
        self.sigma = sigma
        self.step = step
        self.mode = mode

    def fit(self, X, y):
        '''fit the support vector machine model
        Parameters
        ----------
        X : array, shape : nsample, nfeature
            the features of the train samples

        y : ayyay, shape : nsample, 
            the labels of the train samples
        '''
        self.nsample = len(y)
        y = np.array((y)).reshape(self.nsample,)
        # init alpha and bias
        self.init_parameters(y)
        # compute the kernel matrix
        self.calc_kernel_mat(X, self.mode)
        # start training
        for i in range(self.max_iter):
            KKT, error = self.kkt_condition(X, y)
            if(np.sum(KKT) == 0):
                break
            index1 = self.select_first_index(KKT)
            # if not find index1, end the training
            if index1 == -1:
                break
            while True:
                index2 = self.select_second_index(y, index1, error) 
                # not find index2, refind index1 and clear disable index2
                if index2 == -1:
                    self.disable_index2 = []
                    break
                y1 = y[index1]
                y2 = y[index2]
                alpha1_old = self.alpha[index1]
                alpha2_old = self.alpha[index2]
                k11 = self.kernel[index1, index1]
                k12 = self.kernel[index1, index2]
                k22 = self.kernel[index2, index2]
                #if y1 == y2:
                #    Low = max(0, alpha1_old + alpha2_old - self.C)
                #    High = min(self.C, alpha1_old + alpha2_old)
                #else:
                #    Low = max(0, alpha1_old - alpha2_old)
                #    High = min(self.C, alpha1_old - alpha2_old + self.C)
                #eta = k11 + k12 - 2 * k12 + self.epsilon
                #alpha1_new = alpha1_old + y1 * (error[index2] - error[index1]) / eta
                ## clipped the new alpha1
                #if alpha1_new > High:
                #    alpha1_new = High
                #elif alpha1_new < Low:
                #    alpha1_new = Low
                #if abs(alpha1_new - alpha1_old) > self.step:
                #    make_progress = True
                #else:
                #    make_progress = False
                #if not make_progress:
                #    continue
                #alpha2_new = alpha2_old + y1 * y2 * (alpha1_old - alpha1_new)
                if y1 * y2 > 0:
                    Low = max(0, alpha2_old + alpha1_old - self.C)
                    High = min(self.C, alpha2_old + alpha1_old)
                else:
                    Low = max(0, alpha2_old - alpha1_old)
                    High = min(self.C, self.C + alpha2_old - alpha1_old)
                eta = k11 + k22 - 2*k12 + self.epsilon
                alpha2_new = alpha2_old + y2 * (error[index1] - error[index2]) / eta
                if alpha2_new > High:
                    alpha2_new = High
                elif alpha2_new < Low:
                    alpha2_new = Low
                if abs(alpha2_new -alpha2_old) > self.step:
                    make_progress = True 
                else:
                    make_progress = False
                if not make_progress:
                    continue
                alpha1_new = alpha1_old + y1 * y2 *(alpha2_old - alpha2_new)
                b_old = self.b
                self.alpha[index1] = alpha1_new
                self.alpha[index2] = alpha2_new
                self.b = self.calc_bias(X, y)
                # 判断是否更新alpha_new,
                # 更新的条件是:训练误差减小,并且违背kkt条件的样本数目不大于
                # 上次违背kkt条件的数目
                KKT_new, error_new = self.kkt_condition(X, y)
                kkt_num = np.sum(KKT >= 1)
                kkt_new_num = np.sum(KKT_new >= 1)
                total_error = np.sum(np.abs(error))
                total_error_new = np.sum(np.abs(error_new))
                if total_error_new < total_error and kkt_new_num <= kkt_num:
                    self.disable_index1 = []
                    self.disable_index2 = []
                    print(kkt_new_num)
                    break
                else:
                    self.alpha[index1] = alpha1_old
                    self.alpha[index2] = alpha2_old
                    self.b = b_old
        value = self.alpha * y
        value = value.reshape(1, self.nsample)
        W = np.sum(value.T * X, axis=0)
        value = np.sum(value * self.kernel, axis=1) + self.b
        predict_y = value
        print(predict_y)
        predict_y[predict_y < 0] = -1
        predict_y[predict_y >= 0] = 1
        acc = 1.0 - np.sum(np.abs(predict_y - y)) / 2 / self.nsample
        print('train accuracy:', acc)
        print(self.b)
        # 此处返回线性核函数的W和b
        return W, self.b

    def predict(X):
        '''predict the result with given feature X
        '''
        self.calc_kernel_mat(X, self.mode)
        
        pass
    
    def init_parameters(self, y):
        '''初始化alpha参数,初始化方法如下:
        首先统计类标签y中1和-1的个数,假设1和个数为m1, -1的个数为m2,
        则将min(m1, m2)个标签对应的alpha初始化为c/2,其余初始化0,
        这样做的目的是保证 sum: alpha_i * y_i = 0
        '''
        # init alpha
        count1 = 0
        for i in range(self.nsample):
            if y[i] == 1:
                count1 += 1
        count = min(count1, self.nsample - count1)
        self.alpha = np.zeros((self.nsample,), np.float)
        count1 = 0
        count2 = 0
        for i in range(self.nsample):
            if y[i] == 1:
                if count1 < count:
                    self.alpha[i] = self.C / 4.0
                    count1 += 1
            else:
                if count2 < count:
                    self.alpha[i] = self.C / 4.0
                    count2 += 1
        # init bias
        self.b = 0.0
        self.disable_index1 = []
        self.disable_index2 = []

    def select_first_index(self, KKT):
        '''select the first alpha that needed to be optimized.
        '''
        # first select the samples on boundary
        for i in range(self.nsample):
            if KKT[i] == 2:
                if i not in self.disable_index1:
                    self.disable_index1.append(i)
                    return i
        for i in range(self.nsample):
            if KKT[i] == 1:
                if i not in self.disable_index1:
                    self.disable_index1.append(i)
                    return i
        return -1

    def select_second_index(self, y, index1, error):
        '''select the second alpha that needed to be optimized.
        '''
        #y1 = y[index1]
        #self.disable_index2.append(index1)
        #for i in range(self.nsample):
        #    if i not in self.disable_index2:
        #        y2 = y[i]
        #        if y2 == y1:
        #            if self.alpha[index1] + self.alpha[i] < 2.0 * self.C:
        #                self.disable_index2.append(i)
        #                self.disable_index2.remove(index1)
        #                return i
        #        else:
        #            num = self.alpha[index1] - self.alpha[i]
        #            if num > -self.C and num < self.C:
        #                self.disable_index2.append(i)
        #                self.disable_index2.remove(index1)
        #                return i
        #return -1
        y1 = y[index1]
        error1 = error[index1]
        max_error = 0.0
        index2 = -1
        for i in range(self.nsample):
            if i not in self.disable_index2:
                error2 = error[i]
                if abs(error1 - error2) > max_error:
                    index2 = i
                    max_error = abs(error1 - error2)
        if index2 == -1:
            return -1
        else:
            self.disable_index2.append(index2)
            return index2
       
    def kkt_condition(self, X, y):
        '''judge whether the KKT condition is satisfied.
        alphai == 0       <==>    yi*(W*Xi + b) >= 1
        0 < alphai < C    <==>    yi*(W*Xi + b) == 1
        alphai = C        <==>    yi*(W*Xi + b) <= 1
        if satisfy the condition set 0 else set 1 or 2
        '''
        KKT = np.zeros((self.nsample,), np.int)
        value = self.alpha * y
        value = value.reshape(1, self.nsample)
        value = np.sum(value * self.kernel, axis=1) + self.b
        # Ei = W*Xi + b - yi
        error = value - y
        value = value * y
        for i in range(self.nsample):
            if self.alpha[i] <= self.epsilon:
                if value[i] < 1.0:
                    KKT[i] = 1
            elif self.alpha[i] > self.epsilon and self.alpha[i] < self.C - self.epsilon:
                if abs(value[i] - 1.0) > self.tol:
                    KKT[i] = 2
            else:
                if value[i] > 1.0:
                    KKT[i] = 1
        return KKT, error

    def calc_bias(self, X, y):
        '''compute the new bias with new alpha
        截距的计算关系到几何间隔的大小,计算方法会影响到几何间隔是否为1.
        '''
        value = self.alpha * y
        value = value.reshape(1, self.nsample)
        value = np.sum(value * self.kernel, axis=1)
        value = y - value
        bias = 0
        count = 0
        for i in range(self.nsample):
            if self.alpha[i] > self.epsilon and self.alpha[i] < self.C - self.epsilon:
                bias += value[i]
                count += 1
        bias = bias / count
        return bias

    def calc_kernel_mat(self, X, mode='linear'):
        '''compute the kernel 
        为了计算提高计算效率,此处计算核函数矩阵,下次获取其中
        某两个元素的函数只需要访问核函数矩阵对应的索引即可.
        '''
        if mode == 'linear':
            self.kernel = np.dot(X, X.T)
        elif mode == 'gauss':
            # print('kernel function: gaussian')
            distance = np.zeros((self.nsample, self.nsample), np.float)
            for i in range(self.nsample):
                for j in range(self.nsample):
                    distance[i, j] = np.linalg.norm(X[i] - X[j]) ** 2
            self.kernel = np.exp(-distance / (self.sigma ** 2))

