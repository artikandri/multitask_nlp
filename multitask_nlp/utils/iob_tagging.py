from typing import Optional


def to_iob_tag(tag: Optional[str], iob_prefix: str = 'B') -> str:
    """Transform tag to IOB format with given prefix.

    Args:
        tag (str): Raw tag value.
        iob_prefix (str, optional): either 'B' or 'I'. Defaults to 'B'.

    Returns:
        str: IOB tag.
    """
    if tag is None or tag == 'absent':
        return 'O'
    else:
        return f'{iob_prefix}-{tag}'


def from_iob_tag(tag: str) -> Optional[str]:
    """Transform tag from IOB tag to raw tag.

    Args:
        tag (str): IOB tag.

    Returns:
        str: Raw tag.
    """
    if tag == 'O':
        return None
    elif '-' in tag:
        return tag.split('-')[1]
    else:
        return tag
