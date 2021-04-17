#linear and non linear SVM
#The numeric input variables (x) in your data (the columns) form an n-dimensional space. 
#For example, if you had two input variables, this would form a two-dimensional space.

#A hyperplane is a line that splits the input variable space. 
#In SVM, a hyperplane is selected to best separate the points in the input variable space by their class, either class 0 or class 1. 
#In two-dimensions you can visualize this as a line and let’s assume that all of our input points can be completely separated by this line. 
#For example: B0 + (B1 * X1) + (B2 * X2) = 0

#Where the coefficients (B1 and B2) that determine the slope of the line and the intercept (B0) are found by the learning algorithm, and X1 and X2 are the two input variables.

#You can make classifications using this line. 
#By plugging in input values into the line equation, 
#you can calculate whether a new point is above or below the line.

#Above the line, the equation returns a value greater than 0 and the point belongs to the first class (class 0).
#Below the line, the equation returns a value less than 0 and the point belongs to the second class (class 1).
#A value close to the line returns a value close to zero and the point may be difficult to classify.
#If the magnitude of the value is large, the model may have more confidence in the prediction.
#The distance between the line and the closest data points is referred to as the margin. 
#The best or optimal line that can separate the two classes is the line that as the largest margin. This is called the Maximal-Margin hyperplane.


#The margin is calculated as the perpendicular distance from the line to only the closest points. Only these points are relevant in defining the line and in the construction of the classifier. These points are called the support vectors.
#They support or define the hyperplane.

#The hyperplane is learned from training data using an optimization procedure that maximizes the margin.

#linear model
# y =aX + b 

#equation of hyperplane is x*w + b = 0
#if y(i)= 1, w*x+b >= 0 
#if y(i)=-1 , w*x+b <0
#loss is calculated by Hinges loss
# f= y(w.x + b) is positive if point is correctly classified

import numpy as np
from sklearn import datasets
import  matplotlib.pyplot as plt

class SVM:
    def __init__(self,learning_rate=0.001,n_iters=1000,lambda_param=0.01):
       self.lr=learning_rate
       self.lambda_param=lambda_param
       self.n_iters=n_iters
       self.w=None
       self.b=None
       
    def fit(self,X,y):
      y_ = np.where (y <=0 , -1 , 1)
      n_samples,n_features=X.shape
      
      self.w=np.zeros(n_features)
      self.b=0
      
      #gradient descent
      
      for _ in range(self.n_iters):
         for idx,x_i in enumerate(X):
             condition = y_[idx] * (np.dot(x_i,self.w)-self.b) >= 1      
             if condition :
                 self.w -= self.lr * (2 * self.lambda_param * self.w)
             else :
                 self.w -=  self.lr * (2 * self.lambda_param * self.w -np.dot (x_i,y_[idx]))
                 self.b -= self.lr  * y[idx]
                 
      
    def predict(self,x):
        approx=np.dot(X,self.w)-self.b
        return np.sign(approx)
    
    
if __name__  == '__main__' :
  
  X, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
  y = np.where(y == 0, -1, 1)
  clf = SVM()
  clf.fit(X, y)
#predictions = clf.predict(X)
 
  def visualize_svm():
     def get_hyperplane_value(x, w, b, offset):
          return (-w[0] * x + b + offset) / w[1]

     fig = plt.figure()
     ax = fig.add_subplot(1,1,1)
     plt.scatter(X[:,0], X[:,1], marker='o',c=y)

     x0_1 = np.amin(X[:,0])
     x0_2 = np.amax(X[:,0])

     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

     ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
     ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
     ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

     x1_min = np.amin(X[:,1])
     x1_max = np.amax(X[:,1])
     ax.set_ylim([x1_min-3,x1_max+3])

     plt.show()

visualize_svm()
       
