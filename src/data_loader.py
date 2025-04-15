import pandas as pd
import numpy as np
import math
class Learning: 
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.__clean_data()
        self.df_copy=self.df.copy()
        self.discrete_data_ref = {}
        self.__format_dat()
        self.m= self.df.shape[0]
        self.n= self.df.shape[1]
        self.theta_m=np.zeros(self.n)
        self.alpha=0.000001
        self.__run_gradient_descent()
    def __clean_data(self):
        self.df.dropna
    
    def __format_dat(self):
        for col in self.df.columns:
            if isinstance(self.df[col].head().iloc[0],float):
              self.df[col] = self.df[col].apply(lambda x: (x - self.df[col].min())/(self.df[col].max() - self.df[col].min()) )
            if isinstance(self.df[col].head().iloc[0],str): 
              self.df[col], unique = pd.factorize(self.df[col])
              self.discrete_data_ref[col] = dict(enumerate(unique))
    def __sigmoid(self, theta, x):
        x = np.insert(x, 0, 1)
        z = np.dot(theta, x)
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
    def __gradient(self, param_index):
        gradient = 0
        for i in range(self.m):
            gradient = gradient + (self.__sigmoid(self.theta_m, self.df.iloc[i].to_numpy()[:-1])- self.df.iloc[i, self.n-1]) * self.df.iloc[i, param_index]
        return gradient                       
           
    def __param_computation(self):
        for index, theta in enumerate(self.theta_m):
            print(index)
            self.theta_m[index]= self.theta_m[index] - self.alpha * self.__gradient(index)
    def __compute_cost(self):
        cost = 0
        for i in range(self.m):
           cost = cost +  self.df.iloc[i, self.n-1] * math.log10(self.__sigmoid(self.theta_m,self.df.iloc[i].to_numpy()[:-1])) + (1 -self.df.iloc[i, self.n-1]) * math.log10(1 - self.__sigmoid(self.theta_m,self.df.iloc[i].to_numpy()[:-1])) 
        cost = (-1 / self.m) * cost
        return cost
    def __run_gradient_descent(self):
        for i in range(10):
            print(self.__compute_cost())
            self.__param_computation()
            print(self.__compute_cost())