from pathlib import Path
from typing import List, Union
from multitask_nlp.settings import RESULTS_DIR


def read_lines_from_txt_file(file_path: Union[str, Path]) -> List[str]:
    """Reads lines from given text file.

    Returns:
        list of str: Lines in the file.
    """
    with open(file_path, "r", encoding='UTF-8') as f:
        lines = f.read().splitlines()
    return lines

def write_as_txt_file(txt="", file_name=""):
    if file_name is None:
        file_name = f"experiment-{date.today()}"
    path = RESULTS_DIR / f'{file_name}.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(txt))
    print(f"{file_name}.txt has been created. Check at {RESULTS_DIR}")
