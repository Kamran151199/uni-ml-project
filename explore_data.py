
import pandas as pd

df = pd.read_csv('heart_attack_prediction_dataset.csv')

print('Dataset Info:')
df.info()

print('\nDataset Head:')
print(df.head())

print('\nDataset Description:')
print(df.describe())

print('\nMissing Values:')
print(df.isnull().sum())


