from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ParsedTag:
    """NKJP parsed tag class.

    It breaks a tag into particular grammatical class and categories which stored in separate
    attributes.
    """
    grammatical_class: str
    number: str = None
    case: str = None
    gender: str = None
    person: str = None
    degree: str = None
    aspect: str = None
    negation: str = None
    accentity: str = None
    postprepositionity: str = None
    accommodativity: str = None
    agglutinativity: str = None
    vocalicity: str = None
    period: str = None

    def to_tuple(self) -> Tuple[str, ...]:
        return (
            self.grammatical_class, self.number, self.case, self.gender,
            self.person, self.degree, self.aspect, self.negation,
            self.accentity, self.postprepositionity, self.accommodativity,
            self.agglutinativity, self.vocalicity, self.period
        )

    def to_list(self) -> List[str]:
        return [
            self.grammatical_class, self.number, self.case, self.gender,
            self.person, self.degree, self.aspect, self.negation,
            self.accentity, self.postprepositionity, self.accommodativity,
            self.agglutinativity, self.vocalicity, self.period
        ]

    def to_full_tag(self) -> str:
        """Reconstruct full POS tag.

        Returns:
            str: Reconstructed POS tag.
        """
        if self.grammatical_class == 'adj':
            return f'{self.grammatical_class}:{self.number}:{self.case}:{self.gender}:{self.degree}'
        elif self.grammatical_class == 'adv':
            full_tag = self.grammatical_class
            if self.degree is not None:
                full_tag += f':{self.degree}'
            return full_tag
        elif self.grammatical_class == 'aglt':
            return f'{self.grammatical_class}:{self.number}:{self.person}' \
                   f':{self.aspect}:{self.vocalicity}'
        elif self.grammatical_class in ['bedzie', 'fin', 'impt']:
            return f'{self.grammatical_class}:{self.number}:{self.person}:{self.aspect}'
        elif self.grammatical_class == 'brev':
            return f'{self.grammatical_class}:{self.period}'
        elif self.grammatical_class in ['ger', 'pact', 'ppas']:
            return f'{self.grammatical_class}:{self.number}:{self.case}:{self.gender}' \
                   f':{self.aspect}:{self.negation}'
        elif self.grammatical_class in ['imps', 'inf', 'pant', 'pcon']:
            return f'{self.grammatical_class}:{self.aspect}'
        elif self.grammatical_class in ['num', 'numcol']:
            full_tag = f'{self.grammatical_class}:{self.number}:{self.case}:{self.gender}'
            if self.accommodativity is not None:
                full_tag += f':{self.accommodativity}'
            return full_tag
        elif self.grammatical_class == 'ppron12':
            full_tag = f'{self.grammatical_class}:{self.number}:{self.case}:{self.gender}:{self.person}'
            if self.accentity is not None:
                full_tag += f':{self.accentity}'
            return full_tag
        elif self.grammatical_class == 'ppron3':
            full_tag = f'{self.grammatical_class}:{self.number}:{self.case}:{self.gender}' \
                       f':{self.person}'
            if self.accentity is not None:
                full_tag += f':{self.accentity}'
            if self.postprepositionity is not None:
                full_tag += f':{self.postprepositionity}'
            return full_tag
        elif self.grammatical_class == 'praet':
            full_tag = f'{self.grammatical_class}:{self.number}:{self.gender}:{self.aspect}'
            if self.agglutinativity is not None:
                full_tag += f':{self.agglutinativity}'
            return full_tag
        elif self.grammatical_class == 'prep':
            full_tag = f'{self.grammatical_class}:{self.case}'
            if self.vocalicity is not None:
                full_tag += f':{self.vocalicity}'
            return full_tag
        elif self.grammatical_class == 'qub':
            full_tag = self.grammatical_class
            if self.vocalicity is not None:
                full_tag += f':{self.vocalicity}'
            return full_tag
        elif self.grammatical_class == 'siebie':
            return f'{self.grammatical_class}:{self.case}'
        elif self.grammatical_class == 'winien':
            return f'{self.grammatical_class}:{self.number}:{self.gender}:{self.aspect}'
        elif self.grammatical_class in ['subst', 'depr']:
            return f'{self.grammatical_class}:{self.number}:{self.case}:{self.gender}'
        elif self.grammatical_class in ['adja', 'adjc', 'adjp', 'burk', 'comp', 'conj',
                                        'interj', 'interp', 'pred', 'qub', 'xxx', 'ign']:
            return self.grammatical_class
        else:
            raise ValueError

    @classmethod
    def from_tag(cls, tag: str):
        """Create Parsed tag object for given POS tag.

        Parsed POS tag object is obtained by following those guidelines:
        http://nkjp.pl/poliqarp/help/plse2.html

        Args:
            tag (str): Full POS tag.

        Returns:
            ParsedTag: Parsed tag object.
        """
        tag_elements = tag.split(':')
        tag_grammatical_class = tag_elements[0]
        if tag_grammatical_class == 'adj':
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                case=tag_elements[2],
                gender=tag_elements[3],
                degree=tag_elements[4]
            )
        elif tag_grammatical_class == 'adv':
            return cls(
                grammatical_class=tag_elements[0],
                degree=tag_elements[1] if len(tag_elements) == 2 else None
            )
        elif tag_grammatical_class == 'aglt':
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                person=tag_elements[2],
                aspect=tag_elements[3],
                vocalicity=tag_elements[4]
            )
        elif tag_grammatical_class in ['bedzie', 'fin', 'impt']:
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                person=tag_elements[2],
                aspect=tag_elements[3]
            )
        elif tag_grammatical_class == 'brev':
            return cls(
                grammatical_class=tag_elements[0],
                period=tag_elements[1]
            )
        elif tag_grammatical_class in ['ger', 'pact', 'ppas']:
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                case=tag_elements[2],
                gender=tag_elements[3],
                aspect=tag_elements[4],
                negation=tag_elements[5]
            )
        elif tag_grammatical_class in ['imps', 'inf', 'pant', 'pcon']:
            return cls(
                grammatical_class=tag_elements[0],
                aspect=tag_elements[1]
            )
        elif tag_grammatical_class in ['num', 'numcol']:
            tag_obj = cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                case=tag_elements[2],
                gender=tag_elements[3]
            )
            if len(tag_elements) == 5:
                tag_obj.accommodativity = tag_elements[4]

            return tag_obj
        elif tag_grammatical_class == 'ppron12':
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                case=tag_elements[2],
                gender=tag_elements[3],
                person=tag_elements[4],
                accentity=tag_elements[5] if len(tag_elements) == 6 else None
            )
        elif tag_grammatical_class == 'ppron3':
            tag_obj = cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                case=tag_elements[2],
                gender=tag_elements[3],
                person=tag_elements[4]
            )
            if len(tag_elements) == 6:
                tag_obj.accentity = tag_elements[5]
            if len(tag_elements) == 7:
                tag_obj.postprepositionity = tag_elements[6]

            return tag_obj
        elif tag_grammatical_class == 'praet':
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                gender=tag_elements[2],
                aspect=tag_elements[3],
                agglutinativity=tag_elements[4] if len(tag_elements) == 5 else None
            )
        elif tag_grammatical_class == 'prep':
            return cls(
                grammatical_class=tag_elements[0],
                case=tag_elements[1],
                vocalicity=tag_elements[2] if len(tag_elements) == 3 else None
            )
        elif tag_grammatical_class == 'qub':
            return cls(
                grammatical_class=tag_elements[0],
                vocalicity=tag_elements[1] if len(tag_elements) == 2 else None
            )
        elif tag_grammatical_class == 'siebie':
            return cls(
                grammatical_class=tag_elements[0],
                case=tag_elements[1]
            )
        elif tag_grammatical_class == 'winien':
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                gender=tag_elements[2],
                aspect=tag_elements[3]
            )
        elif tag_grammatical_class in ['subst', 'depr']:
            return cls(
                grammatical_class=tag_elements[0],
                number=tag_elements[1],
                case=tag_elements[2],
                gender=tag_elements[3]
            )
        elif tag_grammatical_class in ['adja', 'adjc', 'adjp', 'burk', 'comp', 'conj',
                                       'interj', 'interp', 'pred', 'qub', 'xxx', 'ign']:
            return cls(tag)
        else:
            print(tag)
            raise ValueError

    @classmethod
    def create_consistent_tag(cls, tagset: Dict[str, str]):
        temp_tag = cls(**tagset)
        tag = cls.from_tag(temp_tag.to_full_tag())
        return tag
