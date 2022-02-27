import numpy as np
import operator
import matplotlib.pyplot as plt
import knn_mnist

def get_plot(filename, plotname):
    data, label = knn_mnist.loadData(filename)   # data 784*1877, label 1*1877
    value_of_n = 80   # n = 80
    value_of_k = [i for i in range(1, 51)]
    train_data, train_labels, validation_sets, validation_labels = knn_mnist.split_data(data, label, value_of_n)
    accuracy_sets = []
    for i in range(len(validation_sets)): 
        predictions = [[knn_mnist.KNN(test, train_data, train_labels, k) for test in validation_sets[i].T] for k in value_of_k]
        accuracy = np.array([list(map(abs, line - validation_labels[i])) for line in predictions])
        accuracy_mean = np.mean(accuracy, axis=1)
        accuracy_sets.append(accuracy_mean)
        plt.plot(value_of_k, accuracy_mean, label = str(i + 1))
        plt.ylabel("validation error") 
        plt.xlabel("value of K", loc = 'right')
        plt.title(plotname)
        plt.legend(title = 'value of i')

def draw_plot():
    plt.subplot(2, 2, 1)
    get_plot('MNIST-5-6-Subset.txt', 'normal')
    plt.subplot(2, 2, 2)
    get_plot('MNIST-5-6-Subset-Light-Corruption.txt', 'light-corruption')
    plt.subplot(2, 2, 3)
    get_plot('MNIST-5-6-Subset-Moderate-Corruption.txt', 'moderate-corruption')
    plt.subplot(2, 2, 4)
    get_plot('MNIST-5-6-Subset-Heavy-Corruption.txt', 'heavy-corruption')
    plt.show()

draw_plot()