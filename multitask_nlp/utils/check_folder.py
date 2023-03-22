import os

from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR

model_names = ["luminous-pig-5", "genial-aardvark-6", "lemon-bush-17", "vermilion-dragon-5", "vermilion-dog-1", \
              "dancing-tiger-1", "abundant-horse-1", "floating-fish-2", "beaming-dog-7", "bright-peony-2", \
              "vermilion-mandu-36", "cosmic-mountain-2", "charmed-dust-5", "zesty-planet-5", "stoic-puddle-6" ]



if __name__ == "__main__":
    for model_name in model_names: 
        path = CHECKPOINTS_DIR / model_name
        isExist = os.path.exists(path)
        if isExist:
            print("exists:", model_name)