import pandas as pd
import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, file_path, label_col, alpha=0.01, iterations=1000):
        self.df = pd.read_csv(file_path)
        self.label_col = label_col
        self.alpha = alpha
        self.iterations = iterations
        self.discrete_data_ref = {}

        self.__clean_data()
        self.__format_data()
        self.df = self.df.iloc[:-5000]
        self.test_df = self.df.iloc[-5000:]
        # Prepare features and labels
        self.X = self.df.drop(columns=[self.label_col]).to_numpy()
        self.testX = self.test_df.drop(columns=[self.label_col]).to_numpy()
        self.y = self.df[self.label_col].to_numpy()
        self.testy = self.test_df[self.label_col].to_numpy()
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]

        # Add bias term (column of 1s)
        self.X = np.hstack((np.ones((self.m, 1)), self.X))  # shape: (m, n+1)
        self.theta = np.zeros(self.n + 1)

        self.__run_gradient_descent()

    def __clean_data(self):
        self.df.dropna(inplace=True)

    def __format_data(self):
        for col in self.df.columns:
            if col == self.label_col:
                continue
            if np.issubdtype(self.df[col].dtype, np.number):
                self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
            else:
                self.df[col], unique = pd.factorize(self.df[col])
                self.discrete_data_ref[col] = dict(enumerate(unique))

    def __sigmoid(self, z):
        z = np.clip(z, -500, 500)  # avoid overflow
        return 1 / (1 + np.exp(-z))

    def __compute_cost(self):
        predictions = self.__sigmoid(np.dot(self.X, self.theta))
        # Avoid log(0) by clipping
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        cost = -np.mean(self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions))
        return cost

    def __gradient_step(self):
        predictions = self.__sigmoid(np.dot(self.X, self.theta))
        gradient = (1 / self.m) * np.dot(self.X.T, (predictions - self.y))
        self.theta -= self.alpha * gradient

    def __run_gradient_descent(self):
        for i in range(self.iterations):
            cost = self.__compute_cost()
            if i % 100 == 0 or i == self.iterations - 1:
                print(f"Iteration {i}, Cost: {cost:.6f}")
            self.__gradient_step()

    def predict_prob(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


