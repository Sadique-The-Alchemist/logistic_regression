from src.data_loader import LogisticRegressionFromScratch
import numpy as np

# data_loader = Learning(file_path="data/train.csv")
# print(data_loader.df.dtypes) 
# for col in data_loader.df.columns:
#   if isinstance(data_loader.df[col].head().iloc[0],float):
#     data_loader.df[col] = data_loader.df[col].apply(lambda x: (x - data_loader.df[col].min())/(data_loader.df[col].max() - data_loader.df[col].min()) )
#   if isinstance(data_loader.df[col].head().iloc[0],str): 
#     data_loader.df[col], unique = pd.factorize(data_loader.df[col])
#     print(dict(enumerate(unique)))
  
      
# print(data_loader.df) 
# Assuming your CSV has a column "Outcome" which is 0 or 1
model = LogisticRegressionFromScratch("data/train.csv", label_col="loan_status", alpha=0.1, iterations=1000)

# Predict on training data
predictions = model.predict(model.testX)  # remove bias term for input
accuracy = np.mean(predictions == model.testy)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
