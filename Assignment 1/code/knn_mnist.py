import numpy as np
import operator
import matplotlib.pyplot as plt

def loadData(filename):
    f = open(filename)
    lines = f.readlines()            
    f.close()
    data_array = np.zeros(shape=(784, 1877))
    for line in lines:
        line = line.strip('\n').split(' ')
        line_list = [float(i) for i in line if i != '']
        line_array = np.reshape(np.array(line_list), (784, 1877), 'F')  # reshape data
        data_array = line_array
    f = open('MNIST-5-6-Subset-Labels.txt')  # get label
    lines = f.readlines()
    f.close()
    label_array = np.zeros(shape=(1, 1877))
    for line in lines:
        line = line.strip('\n').split(' ')
        line_list = [int(i) for i in line if i != '']
        line_array = np.array(line_list)
        label_array = line_array
    return data_array, label_array

# split data with different value of n
def split_data(data, label, n):    
    validation_sets = []
    validation_labels = []
    print("----" + str(n))
    for i in range(1, 6):   # validation set i
        validation_index = [100 + i * n + cnt for cnt in range(0, n)]
        validation_labels.append(label[validation_index])    
        validation_sets.append(data[:,validation_index]) 
    train_data = data[:,0:100]
    train_labels = label[0:100]
    return train_data, train_labels, validation_sets, validation_labels

# KNN algorithm
def KNN(test_vec, train_data, train_label, k):
    train_data_num = train_data.shape[1] 
    dif_mat = np.tile(np.reshape(test_vec, (784, 1)), (1, train_data_num)) - train_data 
    sqr_dif_mat = dif_mat ** 2
    sqr_dis = sqr_dif_mat.sum(axis=0) 
    sorted_idx = sqr_dis.argsort() # get the index
    class_cnt = {}  
    for i in range(k):
        tmp_class = train_label[sorted_idx[i]]
        if tmp_class in class_cnt:
            class_cnt[tmp_class] += 1
        else:
            class_cnt[tmp_class] = 1
    sortedVotes = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedVotes[0][0]  # return the prediction of this test data

def draw_plot():
    data, label = loadData('MNIST-5-6-Subset.txt')   # data 784*1877, label 1*1877
    value_of_n = [10, 20, 40, 80]
    value_of_k = [i for i in range(1, 51)]
    for j in range(len(value_of_n)):
        train_data, train_labels, validation_sets, validation_labels = split_data(data, label, value_of_n[j])
        accuracy_sets = []
        plt.subplot(3, 2, j + 1)
        for i in range(len(validation_sets)):  
            # predictions of validation sets with the same size of n
            predictions = [[KNN(test, train_data, train_labels, k) for test in validation_sets[i].T] for k in value_of_k]
            accuracy = np.array([list(map(abs, line - validation_labels[i])) for line in predictions])
            accuracy_mean = np.mean(accuracy, axis=1)
            accuracy_sets.append(accuracy_mean)
            plt.plot(value_of_k, accuracy_mean, label = str(i + 1))
            plt.title("n = " + str(value_of_n[j])) 
            plt.ylabel("validation error") 
            plt.xlabel("value of K", loc = 'right')
            plt.legend(title = 'value of i')
        plt.subplot(3, 1, 3)
        accuracy_sets_var = np.var(np.array(accuracy_sets), axis=0)  # compute the variance 
        plt.plot(value_of_k, accuracy_sets_var, label =str(value_of_n[j]))
        plt.title("variance") 
        plt.ylabel("validation error") 
        plt.xlabel("value of K", loc = 'right')
        plt.legend(title = 'value of n')
    plt.show()

draw_plot()