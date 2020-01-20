import os


PT_OUTPUT_DIR = os.environ.get('PT_OUTPUT_DIR')
OUTPUT_DIR = PT_OUTPUT_DIR if PT_OUTPUT_DIR is not None else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
