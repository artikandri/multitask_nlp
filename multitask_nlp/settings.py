from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

GROUND_TRUTH_TYPE = Union[torch.Tensor, np.ndarray]
MODEL_INPUT_TYPE = Dict[str, Union[torch.Tensor, np.ndarray, str]]
MODEL_OUTPUT_TYPE = Union[torch.Tensor, List[torch.Tensor]]

PROJECT_DIR = Path(__file__).parent.parent.resolve()
STORAGE_DIR = PROJECT_DIR / 'storage'
LOGS_DIR = STORAGE_DIR / 'logs'
CHECKPOINTS_DIR = STORAGE_DIR / 'checkpoints'
RESULTS_DIR = STORAGE_DIR / 'results'
DATASETS_DIR = STORAGE_DIR / 'datasets'

MULTIEMO_DIR = DATASETS_DIR / 'multiemo'
ASPECT_EMO_DIR = DATASETS_DIR / 'aspectemo'
POLEVAL18_POS_DIR = DATASETS_DIR / 'poleval2018'
NKJP1M_DIR = DATASETS_DIR / 'nkjp1m'
CCPL_DIR = DATASETS_DIR / 'ccpl'
KPWR_N82_DIR = DATASETS_DIR / 'kpwr_n82'

MEASURING_HATE_SPEECH_DATA = DATASETS_DIR / 'measuring_hate_speech'

PEJORATIVE_DATA = DATASETS_DIR / 'pejorative_dataset'
CONVABUSE_DATA = DATASETS_DIR / 'convabuse'
GOEMOTION_DATA = DATASETS_DIR / 'goemotions'
STUDEMO_DATA = DATASETS_DIR / 'studemo'
SCITAIL_DATA = DATASETS_DIR / 'scitail'
MULTINLI_DATA = DATASETS_DIR / 'multinli'
SNLI_DATA = DATASETS_DIR / 'snli'
AG_NEWS_DATA = DATASETS_DIR / 'ag_news'
IMDB_DATA = DATASETS_DIR / 'imdb'
CONLL2003_DATA = DATASETS_DIR / 'conll2003_ner'

GLUE_DIR = DATASETS_DIR / 'glue'
RTE_DATA = GLUE_DIR / 'rte'
SST_2_DATA = GLUE_DIR / 'sst2'
STS_B_DATA = GLUE_DIR / 'stsb'

SUPERGLUE_DIR = DATASETS_DIR / 'superglue'
BOOLQ_DATA = SUPERGLUE_DIR / 'boolq'
COMMITMENT_BANK_DATA = SUPERGLUE_DIR / 'commitment_bank'
MULTIRC_DATA = SUPERGLUE_DIR / 'multirc'

KLEJ_DIR = DATASETS_DIR / 'klej'
ALLEGRO_DATA = KLEJ_DIR / 'allegro_reviews'
CBD_DATA = KLEJ_DIR / 'cbd'
CDSC_E_DATA = KLEJ_DIR / 'cdsc-e'
CDSC_R_DATA = KLEJ_DIR / 'cdsc-r'
DYK_DATA = KLEJ_DIR / 'dyk'
NKJP_NER_DATA = KLEJ_DIR / 'nkjp-ner'
POLEMO_IN_DATA = KLEJ_DIR / 'polemo2.0-in'
POLEMO_OUT_DATA = KLEJ_DIR / 'polemo2.0-out'
PSC_DATA = KLEJ_DIR / 'psc'

INDONLU_DIR = DATASETS_DIR/ 'indonlu'
CASA_ABSA_PROSA_DATA = INDONLU_DIR / 'casa_absa_prosa'
HOASA_ABSA_AIRY_DATA = INDONLU_DIR / 'hoasa_absa_airy'
FACQA_QA_FACTOID_ITB_DATA = INDONLU_DIR / 'facqa_qa_factoid_itb'
WRETE_ENTAILMENT_DATA = INDONLU_DIR / 'wrete_entailment_ui'
EMOT_EMOTION_TWITTER_DATA = INDONLU_DIR / 'emot_emotion_twitter'
NERGRIT_NER_GRIT_DATA = INDONLU_DIR / 'nergrit_ner_grit'
KEPS_KEYWORD_EXTRACTION_PROSA_DATA = INDONLU_DIR / 'keps_keyword_extraction_prosa'
NERP_NER_PROSA_DATA = INDONLU_DIR / 'nerp_ner_prosa'
SMSA_DOC_SENTIMENT_PROSA_DATA = INDONLU_DIR / 'smsa_doc_sentiment_prosa'



INDONESIAN_EMOTION_DATASET_DIR = DATASETS_DIR/ 'indonesian_emotion_dataset'


TRANSFORMER_MODEL_STRINGS = {
    'xlmr': 'xlm-roberta-base',
    'xlmr-large': 'xlm-roberta-large',
    'bert': 'bert-base-cased',
    'distilbert': 'distilbert-base-cased',
    'deberta': 'microsoft/deberta-large',
    'roberta': 'roberta-base',
    'herbert': 'allegro/herbert-base-cased',
    'polish-distilroberta': 'sdadas/polish-distilroberta',
    'polish-roberta': 'sdadas/polish-roberta-base-v2',
    'indo-roberta': 'flax-community/indonesian-roberta-base',
    'indo-bert': 'indolem/indobert-base-uncased',
    'labse': 'sentence-transformers/LaBSE',
    'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'xtremedistil-l6-h256': 'microsoft/xtremedistil-l6-h256-uncased',
}
