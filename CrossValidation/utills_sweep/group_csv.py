## Here we will group .csf file by "transforms"column

import pandas as pd

sweep_id = "j9h2x6w7"

df = pd.read_csv("../j9h2x6w7_runs.csv")
# Replace None or nan values in 'Transforms' with a placeholder
df['Transforms'] = df['Transforms'].fillna('No Transforms')

# Group by the 'Transforms' column and calculate the mean for each metric
grouped_df = df.groupby('Transforms').agg(
    {
        'Test F1': ['mean', 'max', 'min'],
        'Test Precision': ['mean', 'max', 'min'],
        'Test Recall': ['mean', 'max', 'min'],
        'Test Accuracy': ['mean', 'max', 'min']
    }
)

# Assuming there are specific columns for confusion matrix
confusion_matrix_columns = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
for col in confusion_matrix_columns:
    if col in df.columns:
        grouped_df[col] = df.groupby('Transforms')[col].mean()

# Flatten the columns
grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

# Save to CSV
grouped_csv_filename = f"{sweep_id}_grouped_runs.csv"
grouped_df.to_csv(grouped_csv_filename, index=True)