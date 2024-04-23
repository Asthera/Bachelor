import wandb
import pandas as pd
import json


def format_methods_from_list(methods_list):
    if methods_list == 'None' or methods_list == 'none':
        return 'None'

    if isinstance(methods_list, str):
        return methods_list

    # move to string
    methods_str = str(methods_list)
    # remove brackets
    methods_str = methods_str[1:-1]

    # "'," -> "\n"
    methods_str = methods_str.replace("', '", "\n")
    # "'" -> ""
    methods_str = methods_str.replace("'", "")

    return methods_str

# Initialize wandb
# Make sure you're logged in. You might need to do `wandb login` in your terminal.

def fetch_sweep_runs(sweep_id, project_path):
    # Initialize a client
    api = wandb.Api()

    # Fetch the sweep
    sweep = api.sweep(f"{project_path}/{sweep_id}")

    # List to hold data for each run
    data = []

    # Iterate through all runs in the sweep
    for run in sweep.runs:
        try:
            path = api.artifact(name=f"daswoldemar/bachelor/run-{run.id}-test_confusion_matrix:v0").download()

            with open(f'artifacts/run-{run.id}-test_confusion_matrix:v0/test_confusion_matrix.table.json', 'r') as f:
                json_data = json.load(f)

            columns_json = json_data['columns']
            data_json = json_data['data'][0]

            # Fetch desired details from the run. Adjust these fields based on your needs.
            run_data = {
                'Run ID': run.id,
                "Transform Init": format_methods_from_list(run.config.get("init_transform")),
                'Transforms': format_methods_from_list(run.config.get("transform")),
                'Output Transform': format_methods_from_list(run.config.get("output_transform")),
                "Fold": int(run.config.get("fold")[-1]),
                "Best Epoch": run.summary.get("best_val_loss_epoch"),
                'Test F1': run.summary.get('test_f1'),
                'Test Precision': run.summary.get('test_precision'),
                'Test Recall': run.summary.get('test_recall'),
                'Test Accuracy': run.summary.get('test_accuracy'),
                f"{columns_json[0]}": data_json[0],
                f"{columns_json[1]}": data_json[1],
                f"{columns_json[2]}": data_json[2],
                f"{columns_json[3]}": data_json[3],
            }
            data.append(run_data)
        except:
            print(f"Run {run.id} has no test_confusion_matrix artifact")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # delete artifacts folder
    import shutil
    shutil.rmtree('artifacts')

    return df


# Replace 'your_sweep_id' and 'your_project_path' with your specific details
sweep_id = 'torch'
project_path = 'daswoldemar/bachelor'

# Fetch data
df = fetch_sweep_runs(sweep_id, project_path)

# Save to Excel
#excel_filename = f"{sweep_id}_runs.xlsx"
#df.to_excel(excel_filename, index=False)

# Save to CSV
csv_filename = f"{sweep_id}_runs.csv"
df.to_csv(csv_filename, index=False)

print("Data saved to CSV file")
