from operator import le
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

class Model:
    def __init__(self,X,y,polyno=1):
        self.X = X**polyno
        for i in range(polyno-1,0,-1):
            self.X = np.hstack((self.X,X**i))
        self.X = np.hstack((self.X,np.ones((X.shape[0],1))))
        self.y = y.reshape((y.shape[0],1))
        self.theta = np.random.randn(self.X.shape[1],1)

    def F(self):
        return self.X.dot(self.theta)

    def function_cout(self):
        return (1/(2*self.y.shape[0]))*np.sum(((self.X.dot(self.theta)-self.y)**2))

    def __gradient(self):
        return (1/self.y.shape[0])*self.X.T.dot((self.F()-self.y))

    def Gradient_Descent(self,learning_rate,n_iter):
        history = list()
        for i in range(n_iter):
            history.append(self.function_cout())
            self.theta -= learning_rate*self.__gradient()
        return history
    def coef_determination(self):
        return 1-((self.y-self.F())**2).sum()/((self.y-self.y.mean())**2).sum()
    def Equation_normale(self):
        self.theta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))
    
X ,y = make_regression(n_samples=100,n_features=2,noise=10)
model = Model(X,y)
print(model.X.shape)
plt.scatter(X[:,1],y)
history = model.Gradient_Descent(learning_rate=0.01,n_iter=1000)
#model.Equation_normale()
plt.scatter(X[:,1],model.F(),c='r')
print(f'{model.coef_determination()}')
plt.show()