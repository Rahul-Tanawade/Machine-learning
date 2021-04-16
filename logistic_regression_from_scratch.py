#implementation of logistic Regresion with numpy
#dataset is breast cancer dataset from sklearn datasets

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self,n_iters,lr):#lr is learning rate and n_iters is no of iteration
       self.n_iters=n_iters
       self.lr=lr
       self.weight=None
       self.bias=None
       
    def fit(self,X,y):# X is feature dataset here and y are labels
        #initialse parameters
        n_samples,n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        self.loss=list()      
        #gradient descent
        
        for _ in range(self.n_iters):
            linear_model=np.dot(X,self.weight)+ self.bias
            y_pred=self._sigmoid(linear_model)
            self.loss.append(np.sqrt(( (y_pred - y) ** 2).mean()))#calculate loss
        
            dw=(1/n_samples)*np.dot(X.T,(y_pred - y))#with gradient descent update weight and bias 
            db=(1/n_samples)*np.sum(y_pred-y)
            
            self.weight -= self.lr*dw
            self.bias -= self.lr*db
            
       
    def _sigmoid(self,x):
    
        return 1 / (1 + np.exp(-x))#sigmoid function implementation
    

    def predict(self,X):

        linear_model=np.dot(X,self.weight)+ self.bias
        y_pred=self._sigmoid(linear_model)

        
        y_pred_class=[1 if i > 0.5 else 0 for i in y_pred]# if probability is above 0.5, its class 1
        return y_pred_class
    
    def plot_loss(self):
        plt.plot(self.loss)
        plt.show()
    
    
if __name__ == '__main__':
    dataset=datasets.load_breast_cancer()
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
    
    
    regressor=LogisticRegression(500,0.0001)
    regressor.fit(X_train,y_train)
    y_pred=regressor.predict(X_test)
    
    def accuracy(y_pred,y_true):
        accuracy=np.sum(y_true==y_pred) / len(y_true)
        return accuracy
    
    print("final accuracy is ",accuracy(y_pred,y_test))
    regressor.plot_loss()   
            
        
