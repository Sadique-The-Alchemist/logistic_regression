import pandas as pd
class DataLoader: 
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.__clean_data()
        self.df_clean=self.df.copy()

    def __clean_data(self):
        self.df.dropna
    