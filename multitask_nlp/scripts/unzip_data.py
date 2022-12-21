import os
from zipfile import ZipFile

file_path = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
dataset_path = os.path.join(os.path.join(project_path, 'storage'), 'datasets')

aspectemo_path = os.path.join(dataset_path, 'aspectemo')
with ZipFile(os.path.join(aspectemo_path, 'documents.zip'), 'r') as zip_ref:
    zip_ref.extractall(path=aspectemo_path)

ccpl_path = os.path.join(dataset_path, 'ccpl')
with ZipFile(os.path.join(ccpl_path, 'anonimizacja_xml_out_ver(3.04).zip'), 'r') as zip_ref:
    zip_ref.extractall(path=ccpl_path)

poleval2018_path = os.path.join(dataset_path, 'poleval2018')
with ZipFile(os.path.join(poleval2018_path, 'poleval2018.zip'), 'r') as zip_ref:
    zip_ref.extractall(path=poleval2018_path)
