from Imports import *
from Utilities import print_sl
from Utilities import summary



train = pd.read_csv(r'C:\Users\vaque\PycharmProjects\Github\Horse-Survival\Data_Inputs\train.csv')
test = pd.read_csv(r'C:\Users\vaque\PycharmProjects\Github\Horse-Survival\Data_Inputs\test.csv')

train_orig = pd.read_csv(r'C:\Users\vaque\PycharmProjects\Github\Horse-Survival\Data_Inputs\horse.csv')

train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

print('Data Loaded Succesfully!')

print_sl()

print(f'train shape: {train.shape}')
print(f'are there any null values in train: {train.isnull().any().any()}\n')

print(f'test shape: {test.shape}')
print(f'are there any null values in test: {test.isnull().any().any()}\n')

print(f'train_orig shape: {train_orig.shape}')
print(f'are there any null values in test: {train_orig.isnull().any().any()}\n')

categorical_cols = ['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time',
                   'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces',
                   'abdomen', 'abdomo_appearance', 'surgical_lesion', 'cp_data']

num_cols = ['hospital_number', 'rectal_temp', 'pulse', 'respiratory_rate', 'nasogastric_reflux_ph', 'packed_cell_volume', 'total_protein',
           'abdomo_protein', 'lesion_1', 'lesion_2', 'lesion_3']

target = 'outcome'

train.head()

styled = summary(train_orig, test).style.background_gradient(cmap='Blues')
styled.to_html("styled_output.html")


# Takeways
# Binary encoding for Lesson 2 (only 4 uniques)


