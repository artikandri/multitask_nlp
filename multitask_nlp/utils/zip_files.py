from zipfile import ZipFile
import os
import shutil
from os.path import basename
from datetime import date
from multitask_nlp.settings import RESULTS_DIR, CHECKPOINTS_DIR
import pandas as pd

project_root = RESULTS_DIR

def zip_file(paths=[], file_name=None):
    if file_name is None:
        file_name = f"Zip-{date.today()}"
    if paths:
        zip_path = RESULTS_DIR / f"{file_name}.zip"
        with ZipFile(zip_path, 'w') as zipObj:
            for path in paths:
                if  os.path.exists(path):
                    for folderName, subfolders, filenames in os.walk(path):
                        for filename in filenames:
                            filePath = os.path.join(folderName, filename)
                            zipObj.write(filePath, basename(filePath))

        print(f"{file_name}.zip has been created. Check at {RESULTS_DIR}")
    else:
        print("No paths provided")

def get_df():
    path = RESULTS_DIR
    file_name = "Thesis Experiments - Results.tsv"
    file_path = path /  file_name
    df = pd.read_csv(file_path, sep="\t", decimal=",")
    df = df[df["is_hidden"] != 1 ]
    df = df.fillna(0)
    return df

def delete_unrelated_folders(checkpoint_names):
    for checkpoint_name in checkpoint_names:
        for folderName, subfolders, filenames in os.walk(CHECKPOINTS_DIR):
            path = CHECKPOINTS_DIR / folderName
            if checkpoint_name not in subfolders and checkpoint_name not in folderName:
                shutil.rmtree(path) 

if __name__ == "__main__":
    df = get_df()
    models = {}
    paths = []
    checkpoint_names = []
    for i, row in df.iterrows():
        models[row["short_task_name"]] = row["last_checkpoint_name"]
        paths.append(CHECKPOINTS_DIR / row["last_checkpoint_name"])
        checkpoint_names.append(row["last_checkpoint_name"])
    
    print(checkpoint_names)
        
    # delete_unrelated_folders(checkpoint_names)
    # zip_file(paths)
    
