import pandas as pd


class SweepEditorDataFrame:
    def __init__(self, df: [pd.DataFrame, pd.DataFrame, ]):
        self.df = self.combine_dataframes(df)

    def init_edit(self) -> pd.DataFrame:
        df = self.df.copy()

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

        return grouped_df

    def combine_dataframes(self, dataframes: [pd.DataFrame, pd.DataFrame, ]):
        combined_dataframes = None

        for df in dataframes:
            combined_dataframes = pd.concat([combined_dataframes, df], axis=0)

        return combined_dataframes
