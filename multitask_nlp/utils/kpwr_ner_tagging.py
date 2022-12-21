from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class KPWr_NER_ParsedTag:
    """KPWr parsed tag class.

    It breaks a tag into categories level which are stored in separate attributes.

    Attributes:
        iob_tag (str): IOB tag.
        category (str, optional): Main NER category. Defaults to None. It is None when there is
            no NER tag.
        subcategory (str, optional) NER subcategory. Defaults to None. It is None when NER tag doeas
            not have detailed subcategory.
        subsubcategory (str, optional): The lowest NER category. Defaults to None. It is None when
            NER tag does not have detailed subsubcategory.
    """
    iob_tag: str
    category: str = None
    subcategory: str = None
    subsubcategory: str = None

    def to_tuple(self) -> Tuple[str, ...]:
        return (
            self.iob_tag, self.category, self.subcategory, self.subsubcategory
        )

    def to_list(self) -> List[str]:
        return [
            self.iob_tag, self.category, self.subcategory, self.subsubcategory
        ]

    def to_full_tag(self, level: int = 3) -> str:
        """Reconstruct full NER tag at a given level of depth.

        Args:
            level (int, optional): Reconstruction of NER tag depth level. Defaults to 3 which is
                the maximum depth.

        Returns:
            str: Reconstructed NER tag.
        """
        full_tag = self.iob_tag

        if level >= 1 and self.category is not None:
            full_tag += f'-{self.category}'
            if level >= 2 and self.subcategory is not None:
                full_tag += f'_{self.subcategory}'
                if level >= 3 and self.subsubcategory is not None:
                    full_tag += f'_{self.subsubcategory}'

        return full_tag

    @classmethod
    def from_tag(cls, tag: str):
        """Create Parsed tag object for given NER tag.

        Parsed NER tag object is obtained by following those guidelines:
        https://clarin-pl.eu/dspace/bitstream/handle/11321/294/WytyczneKPWr-jednostkiidentyfikacyjne.pdf

        Args:
            tag (str): NER tag.

        Returns:
            KPWr_NER_ParsedTag: Parsed tag object.
        """
        tag_split = tag.split('-')
        iob_tag = tag_split[0]
        if iob_tag == 'O' or len(tag_split) == 1:
            return cls(
                iob_tag=iob_tag
            )
        else:
            tag = tag_split[1]
            tag_elements = tag.split('nam_')[-1].split('_')
            tag_category = 'nam_' + tag_elements[0]

            if len(tag_elements) == 1:
                return cls(
                    iob_tag=iob_tag,
                    category=tag_category
                )
            elif tag_category == 'nam_adj':
                return cls(
                    iob_tag=iob_tag,
                    category=tag_category,
                    subcategory=tag_elements[1] if len(tag_elements) == 2 else None
                )
            elif tag_category == 'nam_eve':
                if tag_elements[1] == 'human':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=tag_elements[2] if len(tag_elements) == 3 else None
                    )
                elif tag_elements[1] == 'natural':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory='natural_phenomenom'
                    )
                else:
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1]
                    )
            elif tag_category == 'nam_fac':
                return cls(
                    iob_tag=iob_tag,
                    category=tag_category,
                    subcategory=tag_elements[1],
                    subsubcategory=tag_elements[2] if tag_elements[1] == 'goe' and len(
                        tag_elements) == 3 else None
                )
            elif tag_category == 'nam_liv':
                return cls(
                    iob_tag=iob_tag,
                    category=tag_category,
                    subcategory=tag_elements[1],
                    subsubcategory=tag_elements[2] if tag_elements[1] == 'person' and len(
                        tag_elements) == 3 else None
                )
            elif tag_category == 'nam_loc':
                if tag_elements[1] in 'gpe':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=tag_elements[2] if len(tag_elements) == 3 else None
                    )
                elif tag_elements[1] in 'hydronym':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=tag_elements[2] if len(tag_elements) == 3 else None
                    )
                elif tag_elements[1] in 'land':
                    location_land_category = None
                    if len(tag_elements) == 3:
                        location_land_category = tag_elements[2]
                    if location_land_category == 'protected':
                        location_land_category = 'protected_area'
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=location_land_category
                    )
                else:
                    location_category = tag_elements[1]
                    if location_category == 'country':
                        location_category = 'country_region'
                    elif location_category == 'historical':
                        location_category = 'historical_region'
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=location_category
                    )
            elif tag_category == 'nam_num':
                numex_category = tag_elements[1]
                if numex_category == 'postal':
                    numex_category = 'postal_code'
                return cls(
                    iob_tag=iob_tag,
                    category=tag_category,
                    subcategory=numex_category
                )
            elif tag_category == 'nam_org':
                organization_category = tag_elements[1]
                if organization_category == 'political':
                    organization_category = 'political_party'
                elif organization_category in ['institution', 'organization']:
                    if len(tag_elements) == 3:
                        organization_category += f'_{tag_elements[2]}'

                return cls(
                    iob_tag=iob_tag,
                    category=tag_category,
                    subcategory=organization_category,
                    subsubcategory=tag_elements[2]
                    if organization_category == 'group' and len(tag_elements) == 3 else None
                )
            elif tag_category == 'nam_oth':
                subcategory = tag_elements[1]
                if subcategory == 'address':
                    subcategory = 'address_street'
                elif subcategory == 'data':
                    subcategory = 'data_format'
                elif subcategory == 'stock':
                    subcategory = 'stock_index'

                return cls(
                    iob_tag=iob_tag,
                    category=tag_category,
                    subcategory=subcategory
                )
            elif tag_category == 'nam_pro':
                if tag_elements[1] == 'media':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=tag_elements[2] if len(
                            tag_elements) == 3 else None
                    )
                elif tag_elements[1] == 'model':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=tag_elements[2] if len(
                            tag_elements) == 3 else None
                    )
                elif tag_elements[1] == 'software':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=tag_elements[2] if len(
                            tag_elements) == 3 else None
                    )
                elif tag_elements[1] == 'title':
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1],
                        subsubcategory=tag_elements[2] if len(
                            tag_elements) == 3 else None
                    )
                else:
                    return cls(
                        iob_tag=iob_tag,
                        category=tag_category,
                        subcategory=tag_elements[1]
                    )
            else:
                print(tag)
                raise ValueError

    @classmethod
    def create_consistent_tag(cls, tagset: Dict[str, str]):
        temp_tag = cls(**tagset)
        tag = cls.from_tag(temp_tag.to_full_tag())
        return tag
