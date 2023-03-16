from typing import List, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.settings import MODEL_INPUT_TYPE, MODEL_OUTPUT_TYPE, TRANSFORMER_MODEL_STRINGS


class MultitaskTransformer(nn.Module):
    """Class of Transformer model with many task heads.

    The model similar to one proposed in "Multi-Task Deep Neural Networks for Natural Language
    Understanding" by Liu et al. (2019).

    There is one backbone transformer model, which outputs are processed by heads for a specific
    task.

    Attributes:
        tokenizer: Hugging Face tokenizer used to tokenize text.
        seq_labeling_tokenizer: Tokenizer used for sequence labelling task. For non-RoBERTa models
            the same as tokenizer.
        model: Transformer model loaded from Hugging Face repository.
        model_name (str): Transformer model name, should exist in HF repository. Defaults to "bert".
        max_length (int, optional): Maximum length of transformer input, maximum tokens which can be
            processed. Defaults to 256.
        text_embedding_dim (int): Hidden dimension of model.
        task_heads (torch,nn.ModuleDict): A dictionary mapping task name (or task category) with
            appropriate module head which are fully-connected layers.
    """
    def __init__(self, tasks_datamodules: Union[List[BaseDataModule], BaseDataModule],
                 model_name='bert', max_length=256, **kwargs):
        """Initializes Multitask transformer model.

        Args:
            tasks_datamodules (list of BaseDataModule or BaseDataModule): datamodules for task which
                are to be modelled by transformer.
            model_name (str): Transformer model name, should exist in HF repository. Defaults to
                "bert".
            max_length (int, optional): Maximum length of transformer input, maximum tokens which
                can be processed. Defaults to 256.
            **kwargs: Additional keyword arguments. Not used.

        Raises:
            ValueError: When task type is incorrect.
        """
        super().__init__()
        self.save_hyperparameters()

        if model_name in TRANSFORMER_MODEL_STRINGS:
            model_name = TRANSFORMER_MODEL_STRINGS[model_name]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name in ['roberta-base', 
                            'flax-community/indonesian-roberta-base', 
                            'sdadas/polish-distilroberta',
                          'sdadas/polish-roberta-base-v2']:
            self.seq_labeling_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                add_prefix_space=True
            )
        else:
            self.seq_labeling_tokenizer = self.tokenizer

        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length

        self.text_embedding_dim = self.model.config.hidden_size

        task_heads = nn.ModuleDict()

        if not isinstance(tasks_datamodules, list):
            tasks_datamodules = [tasks_datamodules]

        for task_dm in tasks_datamodules:
            task_kind = task_dm.task_type
            task_category = task_dm.task_category

            if task_category not in task_heads:
                if task_kind == 'classification':
                    output_dim = sum(task_dm.class_dims)
                    task_heads[task_category] = nn.Linear(self.text_embedding_dim, output_dim)
                elif task_kind == 'regression':
                    output_dim = len(task_dm.class_dims)
                    task_heads[task_category] = nn.Linear(self.text_embedding_dim, output_dim)
                elif task_kind == 'sequence labeling':
                    output_dim = sum(task_dm.class_dims)
                    task_heads[task_category] = nn.Sequential(
                        nn.Dropout(p=0.3),
                        nn.Linear(self.text_embedding_dim, output_dim)
                    )
                else:
                    raise ValueError(f"Error, {task_kind} is incorrect task kind.")

        self.task_heads = task_heads

    def forward(self, features: MODEL_INPUT_TYPE) -> MODEL_OUTPUT_TYPE:
        if features['task_type'] == 'sequence labeling':
            return self._process_for_sequence_labeling(features)
        else:
            return self._process_batch(features)

    def _process_batch(self, features: MODEL_INPUT_TYPE) -> torch.Tensor:
        """Processes batches for classification or regression tasks."""

        texts_raw = features['raw_texts'].tolist()
        if 'raw_2nd_texts' in features.keys():
            texts_2nd_raw = features['raw_2nd_texts'].tolist()
            assert len(texts_raw), len(texts_2nd_raw)
            texts_raw = [(text_a, text_b) for text_a, text_b in zip(texts_raw, texts_2nd_raw)]

        tokenizer = self.tokenizer
        model = self.model
        task_head = self.task_heads[features['task_category']]

        batch_encoding = tokenizer.batch_encode_plus(
            texts_raw,
            padding='longest',
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        batch_encoding = batch_encoding.to(model.device)

        model_output = model(**batch_encoding)

        # If transformer model has pooler layer, its output is used as an input to task heads. In
        # the other case, the output of the last hidden layer is used.
        if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
            emb = model_output.pooler_output
        else:
            emb = model_output.last_hidden_state[:, 0, :]

        x = emb
        x = x.view(-1, self.text_embedding_dim)
        output = task_head(x)
        return output

    def _process_for_sequence_labeling(self, features: MODEL_INPUT_TYPE) -> List[torch.Tensor]:
        """Processes batches for sequence labelling tasks."""
        tokenizer = self.seq_labeling_tokenizer
        model = self.model
        task_head = self.task_heads[features['task_category']]

        assert 'tokens' in features
        tokens_raw = features['tokens'].tolist()

        ids_replace_with_unk = []
        for text_id, text_words in enumerate(tokens_raw):
            for wid, word in enumerate(text_words):
                subwords = tokenizer.tokenize(word)

                # when tokenizer does not return any subwords it means that a word is unknown
                if len(subwords) == 0:
                    ids_replace_with_unk.append((text_id, wid))

        for tid, wid in ids_replace_with_unk:
            tokens_raw[tid][wid] = tokenizer.unk_token

        batch_encoding = tokenizer.batch_encode_plus(
            tokens_raw,
            padding='longest',
            add_special_tokens=True,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        # Since transformer tokenizers return subwords tokens but ground truth labels are with
        # respect to full words, it is needed to determine which tokenized subwords come from a
        # particular word.
        words_bounds = []
        for encoding in batch_encoding.encodings:
            example_words_bounds = []
            word_ids = encoding.word_ids

            previous_word_idx = None
            word_left_bound = None
            for i, word_idx in enumerate(word_ids):
                if i == len(word_ids) - 1:
                    if word_left_bound is not None:
                        word_right_bound = i
                        example_words_bounds.append([word_left_bound, word_right_bound])

                elif word_idx != previous_word_idx:
                    if word_left_bound is not None:
                        word_right_bound = i
                        example_words_bounds.append([word_left_bound, word_right_bound])

                    if word_idx is not None:
                        word_left_bound = i
                    else:
                        word_left_bound = None

                previous_word_idx = word_idx

            words_bounds.append(example_words_bounds)

        batch_encoding = batch_encoding.to(model.device)
        outputs = model(**batch_encoding).last_hidden_state

        tokens_embeddings = []
        examples_bounds = []
        e_id = 0
        for i, example_words_bounds in enumerate(words_bounds):
            example_output = outputs[i]
            example_output_length = len(example_output)
            example_tokens_embedding = []

            # As token (word) embedding we take an embedding of its first subword.
            for bound in example_words_bounds:
                start_ind = bound[0]
                if start_ind < example_output_length:
                    example_tokens_embedding.append(
                        example_output[start_ind]
                    )

            example_tokens_embedding = torch.stack(example_tokens_embedding, dim=0)
            tokens_embeddings.append(example_tokens_embedding)

            examples_bounds.append((e_id, e_id + len(example_tokens_embedding)))
            e_id += len(example_tokens_embedding)

        condensed_tokens = torch.cat(tokens_embeddings).view(-1, self.text_embedding_dim)
        condensed_output = task_head(condensed_tokens)

        output = []
        for e_start_ind, e_end_ind in examples_bounds:
            output.append(condensed_output[e_start_ind: e_end_ind])

        return output
