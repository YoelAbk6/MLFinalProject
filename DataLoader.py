import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Feature logical ranges - values out of this range are *definitely* corrupted :

(In general, need to clean all the missing values, meaning 0 values. we decided to set them to the mean and not delete all the row)

0 < Pregnancies 
0 < Glucose (We can maybe dig to a deeper resolution)
40 <= Blood Pressure (Disatolic) <=120
0 < Skin Thickness
0 < Insulin
10 < BMI < 42
0 < Age
Diabetes Pedigree Function(measure of the diabetes mellitus history in relatives and the genetic relationship of those relatives to the patient)
"""


class DataLoader:
    def __init__(self, file_name: str):
        self.df = pd.read_csv(file_name)
        self.df_Pregnancies = self.df["Pregnancies"]
        self.df_Glucose = self.df["Glucose"]
        self.df_BloodPressure = self.df["BloodPressure"]
        self.df_SkinThickness = self.df["SkinThickness"]
        self.df_Insulin = self.df["Insulin"]
        self.df_BMI = self.df["BMI"]
        self.df_DiabetesPedigreeFunction = self.df["DiabetesPedigreeFunction"]
        self.df_Age = self.df["Age"]
        self.df_Outcome = self.df["Outcome"]

    def get_data(self):
        cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.df[cols_to_replace] = self.df[cols_to_replace].replace(0, np.NaN)
        self.df.dropna(thresh=6, inplace=True)  # drop rows that contain 4 or more NaN values
        self.df['BMI'] = self.df['BMI'].apply(lambda x: np.NaN if (x >= 42 or x <= 10) else x)

        col_means = self.df.mean()  # Compute column-wise mean
        self.df.fillna(value=col_means, inplace=True)  # Replace NaN values with column-wise means

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # print(self.df)
        return self.df[self.df.columns[:8]].values.copy(), self.df[self.df.columns[-1]].values.copy()

    def visualize_features_density(self):
        fig, axs = plt.subplots(4, 2)
        axs = axs.flatten()
        sns.histplot(self.df['Pregnancies'], kde=True,
                     color='#38b000', ax=axs[0], stat="density", kde_kws=dict(cut=3))
        sns.histplot(self.df['Glucose'], kde=True,
                     color='#FF9933', ax=axs[1], stat="density", kde_kws=dict(cut=3))
        sns.histplot(self.df['BloodPressure'], kde=True,
                     color='#522500', ax=axs[2], stat="density", kde_kws=dict(cut=3))
        sns.histplot(self.df['SkinThickness'], kde=True,
                     color='#66b3ff', ax=axs[3], stat="density", kde_kws=dict(cut=3))
        sns.histplot(self.df['Insulin'], kde=True,
                     color='#FF6699', ax=axs[4], stat="density", kde_kws=dict(cut=3))
        sns.histplot(self.df['BMI'], color='#e76f51',
                     kde=True, ax=axs[5], stat="density", kde_kws=dict(cut=3))
        sns.histplot(self.df['DiabetesPedigreeFunction'],
                     color='#03045e', kde=True, ax=axs[6], stat="density", kde_kws=dict(cut=3))
        sns.histplot(self.df['Age'], kde=True,
                     color='#333533', ax=axs[7], stat="density", kde_kws=dict(cut=3))
        plt.show()

    def get_clean_data(self):
        return "clear_data"

    def clean_data(self):
        self.__replace_zeros_by_mean()
        self.__replace_OOR_by_mean()

    def __replace_OOR_by_mean(self):
        """
        Replace values that are Out Of Range by the mean
        """
        # print("Number of BMI <= 10 or BMI >= 42:", (self.df_BMI[~self.df_BMI.between(10, 42)]).shape[0])
        self.df_BMI = self.df_BMI.apply(lambda x: self.df_BMI.mean() if (x >= 42 or x <= 10) else x)

        # print("Number of Blood Pressure <= 40 or Blood Pressure >= 120:", (
        #     self.df_BloodPressure[~self.df_BloodPressure.between(40, 120)]).shape[0])
        self.df_BloodPressure = self.df_BloodPressure.apply(
            lambda x: self.df_BloodPressure.mean() if (x >= 120 or x <= 40) else x)

    def __replace_zeros_by_mean(self):
        """
        Replace zero values by the mean
        """
        print("Number of Pregnancies = 0:", (
            self.df_Pregnancies[self.df_Pregnancies == 0]).shape[0])
        self.df_Pregnancies = self.df_Pregnancies.apply(
            lambda x: self.df_Pregnancies.mean() if x == 0 else x)

        print("Number of Glucose = 0:", (
            self.df_Glucose[self.df_Glucose == 0]).shape[0])
        self.df_Glucose = self.df_Glucose.apply(
            lambda x: self.df_Glucose.mean() if x == 0 else x)

        print("Number of Insulin = 0:", (
            self.df_Insulin[self.df_Insulin == 0]).shape[0])
        self.df_Insulin = self.df_Insulin.apply(
            lambda x: self.df_Insulin.mean() if x == 0 else x)

        print("Number of BMI = 0:", (
            self.df_BMI[self.df_BMI == 0]).shape[0])
        self.df_BMI = self.df_BMI.apply(
            lambda x: self.df_BMI.mean() if x == 0 else x)

        print("Number of Blood Pressure = 0:", (
            self.df_BloodPressure[self.df_BloodPressure == 0]).shape[0])
        self.df_BloodPressure = self.df_BloodPressure.apply(
            lambda x: self.df_BloodPressure.mean() if x == 0 else x)

        print("Number of Skin Thickness = 0:", (
            self.df_SkinThickness[self.df_SkinThickness == 0]).shape[0])
        self.df_SkinThickness = self.df_SkinThickness.apply(
            lambda x: self.df_SkinThickness.mean() if x == 0 else x)
