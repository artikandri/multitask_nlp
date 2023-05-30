from pathlib import Path
from typing import List, Union
from multitask_nlp.settings import RESULTS_DIR

from datetime import date, datetime


def read_lines_from_txt_file(file_path: Union[str, Path]) -> List[str]:
    """Reads lines from given text file.

    Returns:
        list of str: Lines in the file.
    """
    with open(file_path, "r", encoding='UTF-8') as f:
        lines = f.read().splitlines()
    return lines

def write_as_txt_file(txt="", file_name="", add_datetime=True):
    if file_name is None:
        file_name = f"experiment-{date.today()}"
    if add_datetime:
        current_datetime = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        txt.append(f"datetime: {current_datetime}")
        file_name = f"{file_name}-{current_datetime}"
    
    path = RESULTS_DIR / f'{file_name}.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(txt))
    print(f"{file_name}.txt has been created. Check at {RESULTS_DIR}")
