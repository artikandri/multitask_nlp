from typing import List, Tuple

import pandas as pd

from multitask_nlp.datasets.base_datamodule import BaseDataModule
from multitask_nlp.kpwr_ner_settings import ANNOTATION_COLUMNS, IOB_LABELS_MAPPING
from multitask_nlp.settings import KPWR_N82_DIR
from multitask_nlp.utils.kpwr_ner_tagging import KPWr_NER_ParsedTag

VAL_DATA_SIZE = 300

NER_TAGS = [
    'nam_adj', 'nam_adj_city', 'nam_adj_country', 'nam_adj_person', 'nam_eve', 'nam_eve_human',
    'nam_eve_human_cultural', 'nam_eve_human_holiday', 'nam_eve_human_sport', 'nam_fac_bridge',
    'nam_fac_goe', 'nam_fac_goe_stop', 'nam_fac_park', 'nam_fac_road', 'nam_fac_square',
    'nam_fac_system', 'nam_liv_animal', 'nam_liv_character', 'nam_liv_god', 'nam_liv_habitant',
    'nam_liv_person', 'nam_loc', 'nam_loc_astronomical', 'nam_loc_country_region',
    'nam_loc_gpe_admin1', 'nam_loc_gpe_admin2', 'nam_loc_gpe_admin3', 'nam_loc_gpe_city',
    'nam_loc_gpe_conurbation', 'nam_loc_gpe_country', 'nam_loc_gpe_district',
    'nam_loc_gpe_subdivision', 'nam_loc_historical_region', 'nam_loc_hydronym',
    'nam_loc_hydronym_lake', 'nam_loc_hydronym_ocean', 'nam_loc_hydronym_river',
    'nam_loc_hydronym_sea', 'nam_loc_land', 'nam_loc_land_continent', 'nam_loc_land_island',
    'nam_loc_land_mountain', 'nam_loc_land_peak', 'nam_loc_land_region', 'nam_num_house',
    'nam_num_phone', 'nam_org_company', 'nam_org_group', 'nam_org_group_band',
    'nam_org_group_team', 'nam_org_institution', 'nam_org_nation', 'nam_org_organization',
    'nam_org_organization_sub', 'nam_org_political_party', 'nam_oth', 'nam_oth_currency',
    'nam_oth_data_format', 'nam_oth_license', 'nam_oth_position', 'nam_oth_tech', 'nam_oth_www',
    'nam_pro', 'nam_pro_award', 'nam_pro_brand', 'nam_pro_media', 'nam_pro_media_periodic',
    'nam_pro_media_radio', 'nam_pro_media_tv', 'nam_pro_media_web', 'nam_pro_model_car',
    'nam_pro_software', 'nam_pro_software_game', 'nam_pro_title', 'nam_pro_title_album',
    'nam_pro_title_article', 'nam_pro_title_book', 'nam_pro_title_document',
    'nam_pro_title_song', 'nam_pro_title_treaty', 'nam_pro_title_tv', 'nam_pro_vehicle'
]


class KPWR_N82_DataModule(BaseDataModule):
    """Datamodule for KPWr NER task.

    It can be processed in tow manners:
     - standard -- NER tags are not split, treated as whole tag.
     - hierarchical -- hierarchy within NER tags is considered. Tags are split into detail levels,
        e.g. tag "nam_eve_human_sport" will be split into: "nam_eve", "human", "sport".
    """
    def __init__(
        self,
        hierarchical: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hierarchical = hierarchical

        self.data_dir = KPWR_N82_DIR

        if hierarchical:
            self.annotation_column = ANNOTATION_COLUMNS
        else:
            self.annotation_column = ['ner']

        self.text_column = 'text'
        self.tokens_column = 'tokens'

        self.train_split_names = ['train']
        self.val_split_names = ['dev']
        self.test_split_names = ['test']

    @property
    def class_dims(self):
        return [len(label_map) for label_map in self.label_maps]

    @property
    def task_name(self):
        task_name = 'KPWr_n82'
        if self.hierarchical:
            task_name += '_hierarchical'
        return task_name

    @property
    def task_type(self):
        return 'sequence labeling'

    @property
    def texts_clean(self):
        return self.data[self.text_column].to_list()

    def prepare_data(self) -> None:
        text_ids, texts, texts_tokens, tags_per_categories, splits = self._get_data_from_files()

        self.data = pd.DataFrame({
            'text_id': text_ids,
            self.text_column: texts,
            self.tokens_column: texts_tokens,
            'split': splits
        })
        self.data[self.tokens_column] = self.data[self.tokens_column].to_numpy()

        if self.hierarchical:
            self.label_maps = IOB_LABELS_MAPPING
        else:
            label_map = {label: i for i, label in enumerate(self._get_labels())}
            self.label_maps = [label_map]

        encoded_tags_per_categories = []
        for tags_per_category, label_map in zip(tags_per_categories, self.label_maps):
            labels = [[label_map[tag] for tag in tags] for tags in tags_per_category]
            encoded_tags_per_categories.append(labels)

        data_for_annotations_df = {
            'text_id': text_ids,
            'annotator_id': 0
        }
        for annotation_column, labels in zip(self.annotation_column, encoded_tags_per_categories):
            data_for_annotations_df[annotation_column] = labels

        self.annotations = pd.DataFrame(data_for_annotations_df)

    def _get_data_from_files(self) -> Tuple[List, ...]:
        train_val_data = self._read_file(self.data_dir / 'kpwr-ner-n82-train-tune.iob')
        train_data = train_val_data[:-VAL_DATA_SIZE]
        val_data = train_val_data[-VAL_DATA_SIZE:]
        test_data = self._read_file(self.data_dir / 'kpwr-ner-n82-test.iob')

        all_data = (
            (train_data, 'train'),
            (val_data, 'dev'),
            (test_data, 'test')
        )

        text_ids, texts, texts_tokens, tags, splits = [], [], [], [], []
        for split_documents_data, split in all_data:
            for document_id, document_data in enumerate(split_documents_data, 1):
                for sentence_id, sentence_data in enumerate(document_data, 1):
                    (sentence, sentence_tokens, sentence_tags) = sentence_data

                    texts.append(sentence)
                    texts_tokens.append(sentence_tokens)
                    tags.append(sentence_tags)
                    text_ids.append(f"{split}_{document_id}_{sentence_id}")
                    splits.append(split)

        assert all([len(tokens) == len(texts_tags) for tokens, texts_tags
                    in zip(texts_tokens, tags)])

        if self.hierarchical:
            parsed_tags = [[KPWr_NER_ParsedTag.from_tag(t).to_list() for t in text_tags] for
                           text_tags in tags]
            tags_per_categories = [list(z) for z in
                                   zip(*[[list(z) for z in zip(*sent_parsed_tags)] for
                                         sent_parsed_tags in parsed_tags])]
        else:
            tags_per_categories = [tags]
        return text_ids, texts, texts_tokens, tags_per_categories, splits

    def _get_labels(self) -> List[str]:
        labels = []
        for t in NER_TAGS:
            labels.append(f'B-{t}')
            labels.append(f'I-{t}')
        labels.append('O')
        return labels

    @staticmethod
    def _read_file(filepath: str) -> List[List[Tuple[str, List[str], List[str]]]]:
        all_documents_data = []
        document_data = []
        sentence_tokens = []
        tags = []

        f = open(filepath, encoding='UTF-8')
        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or line[0] == "\n":
                if len(sentence_tokens) > 0:
                    sentence = ' '.join(sentence_tokens)
                    document_data.append((sentence, sentence_tokens, tags))
                    sentence_tokens = []
                    tags = []
                continue

            elif line.startswith('-DOCSTART'):
                if len(document_data) > 0:
                    all_documents_data.append(document_data)
                    document_data = []
                continue

            splits = line.split('\t')
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, ner_tag = splits[0], splits[3]
            sentence_tokens.append(word.strip())
            tags.append(ner_tag.strip())

        if len(sentence_tokens) > 0:
            sentence = ' '.join(sentence_tokens)
            document_data.append((sentence, sentence_tokens, tags))

        if len(document_data) > 0:
            all_documents_data.append(document_data)

        f.close()
        return all_documents_data
