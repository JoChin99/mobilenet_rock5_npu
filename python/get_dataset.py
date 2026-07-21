import kagglehub
from pathlib import Path

kaggle_dataset = "smeschke/four-shapes"
dataset_name = kaggle_dataset.split("/")[-1]    # To keep the dataset name 
save_dataset = Path.cwd() / "data" / dataset_name
save_dataset.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    kaggle_dataset,
    output_dir = str(save_dataset)
)

# print("File are here: ",path)
print("File are here: ", Path(path).resolve())    # To print the full path