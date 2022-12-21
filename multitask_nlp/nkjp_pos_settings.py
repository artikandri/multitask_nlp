from multitask_nlp.utils.iob_tagging import to_iob_tag

GRAMMATICAL_CATEGORIES_NUM = 13

ANNOTATION_COLUMNS = [
    'grammatical_class', 'number', 'case', 'gender', 'person',
    'degree', 'aspect', 'negation', 'accentity', 'postprepositionity',
    'accommodativity', 'agglutinativity', 'vocalicity', 'period'
]

# In order of grammatical categories. Values taken from NKJP guidelines:
# http://nkjp.pl/poliqarp/help/plse2.html
LABELS_MAPPING = [
    {
        'adj': 0, 'adja': 1, 'adjc': 2, 'adjp': 3, 'adv': 4, 'aglt': 5, 'bedzie': 6, 'brev': 7,
        'burk': 8, 'comp': 9, 'conj': 10, 'depr': 11, 'fin': 12, 'ger': 13, 'imps': 14, 'impt': 15,
        'inf': 16, 'interj': 17, 'interp': 18, 'num': 19, 'numcol': 20,'pact': 21, 'pant': 22,
        'pcon': 23, 'ppas': 24, 'ppron12': 25, 'ppron3': 26, 'praet': 27, 'pred': 28, 'prep': 29,
        'qub': 30, 'siebie': 31, 'subst': 32, 'winien': 33, 'xxx': 34, 'ign': 35
    },
    {'pl': 0, 'sg': 1, 'absent': 2, None: 2},
    {'acc': 0, 'dat': 1, 'gen': 2, 'inst': 3, 'loc': 4, 'nom': 5, 'voc': 6, 'absent': 7, None: 7},
    {'f': 0, 'm1': 1, 'm2': 2, 'm3': 3, 'n': 4, 'absent': 5, None: 5},
    {'pri': 0, 'sec': 1, 'ter': 2, 'absent': 3, None: 3},
    {'com': 0, 'pos': 1, 'sup': 2, 'absent': 3, None: 3},
    {'imperf': 0, 'perf': 1, 'absent': 2, None: 2},
    {'aff': 0, 'neg': 1, 'absent': 2, None: 2},
    {'akc': 0, 'nakc': 1, 'absent': 2, None: 2},
    {'npraep': 0, 'praep': 1, 'absent': 2, None: 2},
    {'congr': 0, 'rec': 1, 'absent': 2, None: 2},
    {'agl': 0, 'nagl': 1, 'absent': 2, None: 2},
    {'nwok': 0, 'wok': 1, 'absent': 2, None: 2},
    {'npun': 0, 'pun': 1, 'absent': 2, None: 2}
]

IOB_LABELS_MAPPING = []
for label_mapping in LABELS_MAPPING:
    iob_label_mapping = {
        to_iob_tag(k): v for k, v in label_mapping.items()
    }
    IOB_LABELS_MAPPING.append(iob_label_mapping)
