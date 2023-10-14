from Imports import *
from Load_Data import train
from Load_Data import train_orig
from Load_Data import test
from Utilities import print_sl
from Load_Data import num_cols



def preprocessing(df, le_cols, ohe_cols):
    le = LabelEncoder()

    for col in le_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, columns=ohe_cols)

    df["pain"] = df["pain"].replace('slight', 'moderate')
    df["peristalsis"] = df["peristalsis"].replace('distend_small', 'normal')
    df["rectal_exam_feces"] = df["rectal_exam_feces"].replace('serosanguious', 'absent')
    df["nasogastric_reflux"] = df["nasogastric_reflux"].replace('slight', 'none')

    df["temp_of_extremities"] = df["temp_of_extremities"].fillna("normal").map(
        {'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3})
    df["peripheral_pulse"] = df["peripheral_pulse"].fillna("normal").map(
        {'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3})
    df["capillary_refill_time"] = df["capillary_refill_time"].fillna("3").map(
        {'less_3_sec': 0, '3': 1, 'more_3_sec': 2})
    df["pain"] = df["pain"].fillna("depressed").map(
        {'alert': 0, 'depressed': 1, 'moderate': 2, 'mild_pain': 3, 'severe_pain': 4, 'extreme_pain': 5})
    df["peristalsis"] = df["peristalsis"].fillna("hypomotile").map(
        {'hypermotile': 0, 'normal': 1, 'hypomotile': 2, 'absent': 3})
    df["abdominal_distention"] = df["abdominal_distention"].fillna("none").map(
        {'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3})
    df["nasogastric_tube"] = df["nasogastric_tube"].fillna("none").map({'none': 0, 'slight': 1, 'significant': 2})
    df["nasogastric_reflux"] = df["nasogastric_reflux"].fillna("none").map(
        {'less_1_liter': 0, 'none': 1, 'more_1_liter': 2})
    df["rectal_exam_feces"] = df["rectal_exam_feces"].fillna("absent").map(
        {'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3})
    df["abdomen"] = df["abdomen"].fillna("distend_small").map(
        {'normal': 0, 'other': 1, 'firm': 2, 'distend_small': 3, 'distend_large': 4})
    df["abdomo_appearance"] = df["abdomo_appearance"].fillna("serosanguious").map(
        {'clear': 0, 'cloudy': 1, 'serosanguious': 2})

    df.drop('lesion_3', axis=1, inplace=True)

    return df


def features_engineering(df):
    df['lesion_2'] = df['lesion_2'].apply(lambda x: 1 if x > 0 else 0)
    data_preprocessed = df.copy()

    data_preprocessed["abs_rectal_temp"] = (data_preprocessed["rectal_temp"] - 37.8).abs()
    data_preprocessed.drop(columns=["rectal_temp"])

    return data_preprocessed

le_cols = ["surgery", "age", "surgical_lesion", "cp_data"]
ohe_cols = ["mucous_membrane"]

train = preprocessing(train, le_cols, ohe_cols)
test = preprocessing(test, le_cols, ohe_cols)
train_orig = preprocessing(train_orig, le_cols, ohe_cols)

#train = encode(train, categorical_cols)
#test = encode(test, categorical_cols)
#train_orig = encode(train_orig, categorical_cols)

total = pd.concat([train, train_orig], ignore_index=True)
total.drop_duplicates(inplace=True)

total = features_engineering(total)
test = features_engineering(test)

num_cols.remove('lesion_3')
num_cols.append('abs_rectal_temp')

# Initialize the KNNImputer with the desired number of neighbors
imputer = KNNImputer(n_neighbors=12) # 10 is good

# Perform KNN imputation
df_train_imputed = pd.DataFrame(imputer.fit_transform(total[num_cols]), columns=num_cols)
df_test_imputed = pd.DataFrame(imputer.transform(test[num_cols]), columns=num_cols)

# Check if there are still missing values in the train and test data sets
df_train_null = df_train_imputed[df_train_imputed.isnull().any(axis=1)]
df_test_null = df_test_imputed[df_test_imputed.isnull().any(axis=1)]

# Display the rows with null values
print('No. of records with missing value in Train data set after Imputation : {}'.format(df_train_null.shape[0]))
print('No. of records with missing value in Test data set after Imputation : {}'.format(df_test_null.shape[0]))

print_sl()

# Replace the imputed columns in the train data sets
total_2 = total.drop(num_cols, axis=1).reset_index()
total_2 = pd.concat([total_2, df_train_imputed], axis=1)

# Replace the imputed columns in the test data sets
test_2 = test.drop(num_cols, axis=1).reset_index()
test_2 = pd.concat([test_2, df_test_imputed], axis=1)

# Check the shape of the train and test data set
print('Shape of the Total data set : {}'.format(total_2.shape))
print('Shape of the Test data set : {}'.format(test_2.shape))

total_2.head(5)
total_2.head(5)
