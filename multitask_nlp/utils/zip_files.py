from zipfile import ZipFile
import os
import shutil
from datetime import date
from os.path import basename
from datetime import date
from multitask_nlp.settings import RESULTS_DIR, CHECKPOINTS_DIR
from multitask_nlp.utils.file_loading import write_as_txt_file
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
    checkpoints = list(os.walk(CHECKPOINTS_DIR))[1:]
    deleted_folders = ["List of deleted folders: "]
    for checkpoint_name in checkpoint_names:
        for folderName, subfolders, filenames in checkpoints:
            path = CHECKPOINTS_DIR / folderName
            if checkpoint_name not in subfolders and checkpoint_name not in folderName:
                print(f"---{folderName} on {path} will be deleted ---") 
                deleted_folders.append(folderName)              
                # shutil.rmtree(folderName) 
    print(f"{len(deleted_folders) - 1} folders have been deleted...")
    write_as_txt_file(deleted_folders, file_name=f"deleted_folders_{date.today()}")

if __name__ == "__main__":
    df = get_df()
    models = {}
    paths = []
    checkpoint_names = []
    for i, row in df.iterrows():
        models[row["short_task_name"]] = row["last_checkpoint_name"]
        paths.append(CHECKPOINTS_DIR / row["last_checkpoint_name"])
        checkpoint_names.append(row["last_checkpoint_name"])
    
    checkpoint_names = ['floral-cosmos-11' 'smooth-serenity-10' 'sweet-wave-2' 'dancing-tiger-1'
        'celestial-darkness-12' 'floating-fish-2' 'wise-bush-2'
        'gallant-forest-10' 'bright-peony-2' 'charmed-dust-5' 'zesty-planet-5'
        'stoic-puddle-6' 'distinctive-totem-3' 'genial-firefly-7'
        'vibrant-aardvark-4' 'dry-grass-12' 'balmy-dream-3' 'light-bee-25'
        'fanciful-donkey-7' 'bright-peony-2' 'distinctive-totem-3'
        'genial-firefly-7' 'brisk-resonance-1' 'frosty-universe-25'
        'dandy-totem-5' 'robust-salad-1' 'icy-wildflower-25' 'frosty-eon-1']
        
    delete_unrelated_folders(checkpoint_names)
    # zip_file(paths)
    
