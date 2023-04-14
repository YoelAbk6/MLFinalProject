import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE

"""
Feature logical ranges - values out of this range are *definitely* corrupted:
 
0 < Glucose (We can maybe dig to a deeper resolution)
40 <= Blood Pressure (Disatolic) <=120
0 < Skin Thickness
0 < Insulin
10 < BMI < 40
0 < Age
Diabetes Pedigree Function(measure of the diabetes mellitus history in relatives and the genetic relationship of those relatives to the patient)
"""


class DataLoader:
    """
    A class that loads, cleans, and splits the Diabetes Dataset into training and testing data.
    """
    def __init__(self, file_name: str):
        self.df = pd.read_csv(file_name)
        self.averages = {}

    def get_raw_data(self):
        return self.df[self.df.columns[:8]].values.copy(), self.df[self.df.columns[-1]].values.copy()

    def get_clean_data(self):
        """
        Cleans the dataset by replacing zero values with NaN and dropping rows with four or more NaN values. Also,
        replaces out-of-range values with NaN and replaces remaining NaN values with the column averages.
        """
        cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.df[cols_to_replace] = self.df[cols_to_replace].replace(0, np.NaN)

        # drop rows that contain 4 or more NaN values
        self.df.dropna(thresh=6, inplace=True)

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

    def get_train_test_norm(self, X, y):
        """
        Splits the cleaned data into training and testing data, and standardizes the feature data. Also, uses SMOTE
        to balance the training data by oversampling the minority class.
        """
        rs = check_random_state(42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)

        """
        - Synthetic Minority Over-sampling Technique
        - Can be used to deal with the outcome imbalance (65(Negative)/35(Positive))
        - Did't result better accuraccy overall, does improve the prediction of positives
        - Sampling_strategy is the new ratio between the minority and the majority
        - In our DS, after the cleaning, the minority/majority ratio is 0.53/1
        """
        sm = SMOTE(random_state=rs)
        X_train_balanced, y_train_balanced = sm.fit_resample(
            X_train_norm, y_train)

        return X_train_balanced, X_test_norm, y_train_balanced, y_test, rs
