from Imports import *
# Functions
def print_sl():
    print("=" * 50)
    print()

def show_na(df, column):
    sns.countplot(x='outcome', data=df[df[column].isnull()])
    plt.show()


def summary(train, test):
    train = train.drop(columns=["outcome"])
    sum = pd.DataFrame(train.dtypes, columns=['dtypes'])
    sum['Orig_missing#'] = train.isna().sum()
    sum['Test_missing#'] = test.isna().sum()
    sum['Orig_missing%'] = (train.isna().sum()) / len(train)
    sum['Test_missing%'] = (test.isna().sum()) / len(train)
    sum['Orig_uniques'] = train.nunique().values
    sum['Test_uniques'] = test.nunique().values
    sum['Orig_count'] = train.count().values
    sum['Test_count'] = test.count().values

    return sum


