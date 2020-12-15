import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('weight-height.csv')

def display_data():
    print(data)

def clean_target(target):
    target.loc[target == 'Male'] = 0
    target.loc[target == 'Female'] = 1
    return target.astype(int)

def get_splitted_data(nsamples=None, imbalanced_samples = None):
    data.columns = ['gender', 'height', 'weight']

    sub_data = data if nsamples is None else data.sample(nsamples, random_state=45)

    if imbalanced_samples is not None:
        female_sample = data[data['gender'] == 'Female'].sample(imbalanced_samples, random_state=1)
        sub_data = pd.concat([sub_data, female_sample])
        sub_data.drop_duplicates(inplace=True)
    

    features = sub_data.drop('gender', axis=1)
    target = clean_target(sub_data['gender'])

    #features_train, features_valid, target_train, target_valid 
    return train_test_split(features, target, test_size=0.25)

