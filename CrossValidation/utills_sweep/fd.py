# read  general_single_higher_than_best_none_renamed.csv
# where column "Type of Augmentation" is NaN, change it to "No Augmentation"


import pandas as pd

df = pd.read_csv("general_single_higher_than_best_none_renamed.csv")
df["Type of Augmentation"] = df["Type of Augmentation"].fillna("No Augmentation")
df.to_csv("general_single_higher_than_best_none_renamed.csv", index=False)