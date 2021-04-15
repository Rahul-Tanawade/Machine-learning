"""regression predicts continious values
# y =wx+b  
# b is shift/bias , y is op and w is slope
# aim is to find w and b
#cost function (MSE)should be minimum , difference between actual & predicted
#gradient of cost function needs to be found to minimise value of it
 with best values of  w and b  
 in each iteration, value of w & b is updated to get minimum cost function

"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LinearRegression:
     
   def __init__(self,n_iters,lr=0.01):
       self.lr=lr
       self.n_iters=n_iters
       self.weights=None   
       self.bias=None

 
      
   def fit(self,X,y):
       #implemet gradient descent
       n_samples,n_features=X.shape
       self.weights=np.zeros(n_features)
       self.bias=0
       self.loss=list()
       for _ in range(self.n_iters):
           #in each iteration update wieghts
           y_predicted=np.dot(X,self.weights)+self.bias
           
           #noting down loss for each iteration
           self.loss.append(np.mean((y-y_predicted)**2))
           
           dw= (1/n_samples) * np.dot(X.T,(y_predicted - y))
           db= (1/n_samples) * np.sum(y_predicted -y)
           
           self.weights -= self.lr*dw
           self.bias -= self.lr*db


   def predict(self,x):
       y_predicted=np.dot(x,self.weights)+self.bias     
       return y_predicted
   
   def mse(self,y_true,y_predicted):
        return np.mean((y_true-y_predicted)**2)
    
   def plotting_loss(self):
       plt.plot(self.loss)
       plt.xlabel("iteration number")
       plt.ylabel("loss")
       plt.title("loss plot")
       plt.show()
       
   def plotting_final_result(self):
       
       fig=plt.figure(figsize=(8,6))
       
       m1=plt.scatter(X_train,y_train,color='blue',s=10)#plotting training set
       m2=plt.scatter(X_test,y_test,color='red',s=10)#plotting test set
       
       y_pred_line=self.predict(X)#with final bias,weights finding y_pred for all
       plt.plot(X,y_pred_line,color='black',linewidth=2,label='prediction')
       
       plt.show()    
           
if __name__ == '__main__':
    
     X,y= datasets.make_regression(n_samples=100, n_features=1,noise=20,random_state=4)
     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
    
     Regressor=LinearRegression(500)       
     Regressor.fit(X_train,y_train)  
     predicted=Regressor.predict(X_test)
     
     mse_calculated=Regressor.mse(y_test,predicted)
     Regressor.plotting_loss()
     Regressor.plotting_final_result()
     
