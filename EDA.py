
from Imports import *
from Load_Data import train


# Create a copy of the dataframe
df_encoded = train.copy()

# Assuming these are your categorical variables, including 'outcome'
categorical_vars = ['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse',
                    'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis',
                    'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux',
                    'rectal_exam_feces', 'abdomen', 'abdomo_appearance', 'surgical_lesion',
                    'cp_data', 'outcome']

# Label encode categorical columns
label_encoders = {}
for column in categorical_vars:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(train[column])
    label_encoders[column] = le


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    corr = df[columns_without_excluded].corr()

    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrBr_r', annot=True, annot_kws={"size": 6})
    plt.title(title_name)
    plt.show()


# Plot correlation heatmap for encoded dataframe
plot_correlation_heatmap(df_encoded, 'Encoded Dataset Correlation')

# Takeaways
# lesion 3 is going to add noise, we remove it
# Absolute diference for abs_rectal_temp to quantify the deviation to highlight anomalies

