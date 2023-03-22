import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
from multitask_nlp.models import models as models_dict
from multitask_nlp.settings import CHECKPOINTS_DIR, LOGS_DIR

path = CHECKPOINTS_DIR / "twilight-sound-6/epoch=2-step=639.ckpt"

model_type = "multitask_transformer"
model_name = "xlmr"
model_cls = models_dict[model_type]
model = model_cls(
    tasks_datamodules=data_module,
    model_name=model_name,
    max_length=max_length
)
model = model.load_from_checkpoint(path)
print(model.learning_rate)


# CHECKPOINT_PATH = "/mnt/big_one/persemo/mgruza/personalized-nlp/storage/checkpoints/drawn-sponge-6/epoch=1-step=2654.ckpt"
# TEXTS_PATH = "/mnt/big_one/persemo/mgruza/personalized-nlp/storage/other/some_texts.csv"
# PREDICTION_PATH = "/mnt/big_one/persemo/mgruza/personalized-nlp/storage/other/some_texts_with_predictions.csv"

# USE_CUDA = True


# def batch_forward(classifier, texts, batch_size=100):
#     def batch(iterable, n=batch_size):
#         l = len(iterable)
#         for ndx in range(0, l, n):
#             yield iterable[ndx : min(ndx + n, l)]

#     probabs = []
#     for text_batch in tqdm.tqdm(batch(texts)):
#         annotators = np.array([0] * text_batch.shape[0])
#         batch_probabs = classifier.forward(
#             {"raw_texts": text_batch, "annotator_ids": annotators}
#         )
#         probabs.extend(batch_probabs.unsqueeze(0))

#     return torch.cat(probabs, dim=0)


# if __name__ == "__main__":

#     map_location = "cuda" if USE_CUDA else "cpu"
#     classifier = Classifier.load_from_checkpoint(
#         CHECKPOINT_PATH, map_location=map_location
#     )
#     classifier.model.use_cuda = USE_CUDA
#     classifier = classifier.eval()

#     texts = pd.read_csv(TEXTS_PATH)["text"].values

#     with torch.no_grad():
#         probabs = batch_forward(classifier, texts)

#     predictions = classifier.decode_predictions(probabs).cpu().numpy()

#     result_df = pd.DataFrame({"text": texts})
#     for idx, col in enumerate(classifier.class_names):
#         result_df[col] = predictions[:, idx]

#     result_df.to_csv(PREDICTION_PATH, index=False)