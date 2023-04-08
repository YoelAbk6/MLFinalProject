import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE

"""
Feature logical ranges - values out of this range are *definitely* corrupted :

 
0 < Glucose (We can maybe dig to a deeper resolution)
40 <= Blood Pressure (Disatolic) <=120
0 < Skin Thickness
0 < Insulin
10 < BMI < 40
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
        self.averages = {}

    def get_data(self):
        cols_to_replace = ['Glucose', 'BloodPressure',
                           'SkinThickness', 'Insulin', 'BMI']
        self.df[cols_to_replace] = self.df[cols_to_replace].replace(0, np.NaN)
        # drop rows that contain 4 or more NaN values
        self.df.dropna(thresh=6, inplace=True)

        # # replace all the values that above the 95 percent line and down 5 percent line with NaN
        # for col in cols_to_replace:
        #     col_values = self.df[col].values
        #     lower_bound = np.percentile(col_values, 5)
        #     upper_bound = np.percentile(col_values, 95)
        #     self.df[col] = np.where((col_values < lower_bound) | (
        #         col_values > upper_bound), np.nan, col_values)

        # clean specific data
        self.df['BMI'] = self.df['BMI'].apply(
            lambda x: np.NaN if (x > 40 or x <= 10) else x)
        self.df['BloodPressure'] = self.df['BloodPressure'].apply(
            lambda x: np.NaN if (x >= 120 or x <= 40) else x)

        # save all the averages for further use
        self.averages = self.df.mean()
        # Replace NaN values with column averages
        self.df.fillna(value=self.averages, inplace=True)

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
        self.__replace_OOR_by_mean()

    def __replace_OOR_by_mean(self):
        """
        Replace values that are Out Of Range by the mean
        """
        # print("Number of BMI <= 10 or BMI >= 42:", (self.df_BMI[~self.df_BMI.between(10, 42)]).shape[0])
        self.df_BMI = self.df_BMI.apply(
            lambda x: self.df_BMI.mean() if (x >= 42 or x <= 10) else x)

        # print("Number of Blood Pressure <= 40 or Blood Pressure >= 120:", (
        #     self.df_BloodPressure[~self.df_BloodPressure.between(40, 120)]).shape[0])
        self.df_BloodPressure = self.df_BloodPressure.apply(
            lambda x: self.df_BloodPressure.mean() if (x >= 120 or x <= 40) else x)

    def get_train_test_norm(self, X, y):
        rs = check_random_state(42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rs)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)

        # return X_train_norm, X_test_norm, y_train, y_test, rs

        """
        - Synthetic Minority Over-sampling Technique
        - Can be used to deal with the outcome imbalance (65(Negative)/35(Positive))
        - Did't result better accuraccy overall, does improve the prediction of positives
        - Sampling_strategy is the new ratio between the minority and the majority
        - In our DS, after the cleaning, the minority ratio is 0.53
        """
        sm = SMOTE(sampling_strategy=0.8, random_state=rs)
        X_train_balanced, y_train_balanced = sm.fit_resample(
            X_train_norm, y_train)
        return X_train_balanced, X_test_norm, y_train_balanced, y_test, rs
