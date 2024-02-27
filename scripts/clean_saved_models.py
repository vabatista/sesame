## Code to remove recursevly all saved-model folders in results folder
## Usage: python clean_saved_models.py

import os
import shutil

def clean_saved_models(path):
    for root, dirs, files in os.walk(path):
        for name in dirs:
            if name == 'saved-models':
                shutil.rmtree(os.path.join(root, name))
        for file in files:
            if file.find('tokenizer') > 0 or file.find('config') > 0 or file.endswith('.bin') or file.endswith('.safetensors'):
                os.remove(os.path.join(root, file))
                #print(os.path.join(root, file))
if __name__ == '__main__':  
    clean_saved_models('./results')

