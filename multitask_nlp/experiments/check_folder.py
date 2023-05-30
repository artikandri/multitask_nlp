import os

from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR

model_names = ["magic-brook-21", "serene-bee-6", "icy-wildflower-25"]


if __name__ == "__main__":
    for model_name in model_names: 
        path = CHECKPOINTS_DIR / model_name
        isExist = os.path.exists(path)
        if isExist:
            print("exists:", model_name)