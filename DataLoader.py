import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, file_name: str):
        self.df = pd.read_csv(file_name)
        # self.Pregnancies = df["Pregnancies"].to_numpy()
        # self.Glucose = df["Glucose"].to_numpy()
        # self.BloodPressure = df["BloodPressure"].to_numpy()
        # self.SkinThickness = df["SkinThickness"].to_numpy()
        # self.Insulin = df["Insulin"].to_numpy()
        # self.BMI = df["BMI"].to_numpy()
        # self.DiabetesPedigreeFunction = df["DiabetesPedigreeFunction"].to_numpy()
        # self.Age = df["Age"].to_numpy()
        # self.Outcome = df["Outcome"].to_numpy()

    def get_data(self):
        return self.df[self.df.columns[:8]].values.copy(), self.df[self.df.columns[-1]].values.copy()

    def get_clear_data(self):
        return "clear_data"
