r"""自動對路徑下的所有py檔執行 autopep8 --in-palce
Usage:
    python auto_format.py
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(__file__),
    os.pardir
))

for root, dirs, files in os.walk(PROJECT_ROOT):
    for name in files:
        if name[-3:] == '.py':
            file_path = os.path.join(root, name)
            os.system(f'autopep8 --in-place {file_path}')
