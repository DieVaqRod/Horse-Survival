from Data_Cleaning import total_2
from Data_Cleaning import test_2
from Load_Data import target

X_train = total_2.drop(columns=[target])
y_train = total_2[target].map({'died':0,'euthanized':1,'lived':2})
X_test = test_2

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
