import wandb
import pandas as pd
import json
import shutil


def fetch_confusion_matrix(run_id, matrix_type, project_path="daswoldemar/bachelor"):
    # Initialize a client
    api = wandb.Api()

    try:
        artifact_name = f"{project_path}/run-{run_id}-{matrix_type}_confusion_matrix:v0"
        path = api.artifact(name=artifact_name).download()

        with open(f'artifacts/run-{run_id}-{matrix_type}_confusion_matrix:v0/{matrix_type}_confusion_matrix.table.json',
                  'r') as f:
            json_data = json.load(f)

        columns_json = json_data['columns']
        data_json = json_data['data'][0]

        # Fetch confusion matrix details
        run_data = {
            'Run ID': run_id,
            f"{columns_json[0]}": data_json[0],
            f"{columns_json[1]}": data_json[1],
            f"{columns_json[2]}": data_json[2],
            f"{columns_json[3]}": data_json[3],
        }

        shutil.rmtree('artifacts')
        return pd.DataFrame([run_data])

    except Exception as e:
        print(f"Run {run_id} has no {matrix_type}_confusion_matrix artifact: {e}")
        return None


def calculate_total_positives_negatives(dataframe):
    # Convert relevant columns to integers
    dataframe.iloc[:, 1:] = dataframe.iloc[:, 1:].astype(int)

    total_positives = dataframe.iloc[:, 3].sum() + dataframe.iloc[:, 4].sum()
    total_negatives = dataframe.iloc[:, 1].sum() + dataframe.iloc[:, 2].sum()

    return total_positives, total_negatives


# Example usage
run_ids = ["oi2a1svm", "wwen7gyv", "phm0ry3h", "3qsymi42", "od3n2vjg"]
matrix_types = ["train", "test", "val"]

results = []

for run_id in run_ids:
    for matrix_type in matrix_types:
        df = fetch_confusion_matrix(run_id, matrix_type)
        if df is not None:
            total_positives, total_negatives = calculate_total_positives_negatives(df)
            results.append({
                'Run ID': run_id,
                'Matrix Type': matrix_type,
                'Total Positives': total_positives,
                'Total Negatives': total_negatives
            })

# Create a DataFrame for pretty printing
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
