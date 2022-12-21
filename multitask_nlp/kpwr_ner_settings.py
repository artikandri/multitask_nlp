from copy import copy

ANNOTATION_COLUMNS = [
    'iob_tag', 'category', 'subcategory', 'subsubcategory'
]

# In order given in ANNOTATION_COLUMNS
# Values taken from KPWr guidelines:
# https: // clarin - pl.eu / dspace / bitstream / handle / 11321 / 294 / WytyczneKPWr - jednostkiidentyfikacyjne.pdf
LABELS_MAPPING = [
    {'B': 0, 'I': 1, 'O': 2},
    {'nam_adj': 0, 'nam_eve': 1, 'nam_fac': 2, 'nam_liv': 3, 'nam_loc': 4, 'nam_num': 5,
     'nam_org': 6, 'nam_oth': 7, 'nam_pro': 8, None: 9},
    {'person': 0, 'country': 1, 'city': 2, 'human': 3, 'natural_phenomenom': 4, 'bridge': 5,
     'cossroad': 6, 'goe': 7, 'park': 8, 'road': 9, 'square': 10, 'system': 11, 'animal': 12,
     'character': 13, 'god': 14, 'habitant': 15, 'plant': 16, 'astronomical': 17,
     'country_region': 18, 'gpe': 19, 'historical_region': 20, 'hydronym': 21,
     'land': 22, 'company': 23, 'group': 24, 'nation': 25, 'organization': 26, 'institution': 27,
     'institution_full': 28, 'organization_sub': 29, 'political_party': 30, 'postal_code': 31,
     'house': 32, 'flat': 33, 'phone': 34, 'address_street': 35, 'currency': 36, 'data_format': 37,
     'ip': 38, 'license': 39, 'mail': 40, 'position': 41, 'stock_index': 42, 'tech': 43, 'www': 44,
     'award': 45, 'brand': 46, 'media': 47, 'model': 48, 'software': 49, 'title': 50, 'vehicle': 51,
     None: 52},
    {'aniversary': 0, 'holiday': 1, 'sport': 2, 'cultural': 3, 'market': 4, 'stop': 5, 'add': 6,
     'first': 7, 'last': 8, 'admin': 9, 'admin1': 10, 'admin2': 11, 'admin3': 12, 'city': 13,
     'conurbation': 14, 'country': 15, 'district': 16, 'subdivision': 17, 'bay': 18, 'lagoon': 19,
     'lake': 20, 'ocean': 21, 'river': 22, 'sea': 23, 'cape': 24, 'continent': 25, 'desert': 26,
     'island': 27, 'mountain': 28, 'peak': 29, 'peninsula': 30, 'protected_area': 31, 'region': 32,
     'sandspit': 33, 'band': 34, 'team': 35, 'tv': 36, 'radio': 37, 'periodic': 38, 'web': 39,
     'car': 40, 'phone': 41, 'plane': 42, 'ship': 43, 'os': 44, 'game': 45, 'version': 46,
     'album': 47, 'article': 48, 'boardgame': 49, 'book': 50, 'document': 51, 'painting': 52,
     'movie': 53, 'song': 54, 'treaty': 55, None: 56},
]

IOB_LABELS_MAPPING = []
for i, label_mapping in enumerate(LABELS_MAPPING):
    new_label_mapping = copy(label_mapping)
    IOB_LABELS_MAPPING.append(new_label_mapping)
