import knn_mnist
import corrupt
import inequality
import regression

# This main file can call other files with functions

knn_mnist.draw_plot()
print("the figure of knn with normal MNIST subset has been saved")
corrupt.draw_plot()
print("the figure of knn with corrupt MNIST subset has been saved")
inequality.draw_plot()
print("the figure of inequalities has been saved")
regression.draw_plot()
print("the figure of regression has been saved")