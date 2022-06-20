import matplotlib.pyplot as plt
import os

def save_image_to_file(image, filename):
    image.save(filename)

def create_directory(directory_path):
    os.mkdir(directory_path)

def init_jupyter_notebook():
    ### Add libraries path to system path so python can easily access it ###
    import os, sys

    currentdir = os.getcwd()
    parentdir = os.path.dirname(currentdir)
    sys.path.append(f"{parentdir}/new_code_rewrite")

    ### Save conda environment to requirements file ###
    os.system("conda list -e > requirements.txt")

    ### Print run time of jupyter notebook first cell run ###
    from datetime import datetime

    print(f"Cell run at {datetime.today()}")