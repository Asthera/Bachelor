import wandb
import pandas as pd
import json
import shutil


class SweepReader:
    def  __init__(self, sweep_id, project_path="daswoldemar/bachelor"):
        self.sweep_id = sweep_id
        self.project_path = project_path
        self.dataframe = None

    def fetch_sweep_runs(self) -> pd.DataFrame:
        # Initialize a client
        api = wandb.Api()

        # Fetch the sweep
        sweep = api.sweep(f"{self.project_path}/{self.sweep_id}")

        # List to hold data for each run
        data = []

        # Iterate through all runs in the sweep
        for run in sweep.runs:
            try:
                path = api.artifact(name=f"{self.project_path}/run-{run.id}-test_confusion_matrix:v0").download()

                with open(f'artifacts/run-{run.id}-test_confusion_matrix:v0/test_confusion_matrix.table.json',
                          'r') as f:
                    json_data = json.load(f)

                columns_json = json_data['columns']
                data_json = json_data['data'][0]

                # Fetch desired details from the run. Adjust these fields based on your needs.
                run_data = {
                    'Run ID': run.id,
                    "Transform Init": self.format_methods_from_list(run.config.get("init_transform")),
                    'Transforms': self.format_methods_from_list(run.config.get("transform")),
                    'Output Transform': self.format_methods_from_list(run.config.get("output_transform")),
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

        shutil.rmtree('artifacts')

        self.dataframe = pd.DataFrame(data)

        # Convert to DataFrame
        return pd.DataFrame(data)

    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe

    def read_from_csv(self, filename: str) -> pd.DataFrame:
        self.dataframe = pd.read_csv(filename)
        return self.dataframe

    def save_to_csv(self, filename: str) -> None:
        self.dataframe.to_csv(filename, index=False)
        print("Data saved to CSV file")

    def format_methods_from_list(self, methods_list: list or str) -> str:
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
