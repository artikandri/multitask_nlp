from pathlib import Path
from typing import List, Union


def read_lines_from_txt_file(file_path: Union[str, Path]) -> List[str]:
    """Reads lines from given text file.

    Returns:
        list of str: Lines in the file.
    """
    with open(file_path, "r", encoding='UTF-8') as f:
        lines = f.read().splitlines()
    return lines
