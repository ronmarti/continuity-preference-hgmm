import json
import pickle
# from io import StringIO
from pathlib import Path
import os


def save_json(obj, filename):
    directory, name = os.path.split(os.path.abspath(filename))
    filename = Path(filename)
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    file_path = Path(filename).absolute()
    assert file_path.is_file(), f'No such file as {file_path}.'
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def save_pickle(obj, filename):
    directory, name = os.path.split(os.path.abspath(filename))
    filename = Path(filename)
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    file_path = Path(filename).absolute()
    assert file_path.is_file(), f'No such file as {file_path}.'
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == "__main__":

    
    N_epochs = 40
    batch_size = 300

    
    N_states = 2
    cont_precision = 10

    usecols_kwargs = {'usecols': ['IW_508', 'IW_512']}

    listOfDicts = {
        'batch_size': batch_size,
        'N_epochs': N_epochs,
        'N_states': N_states,
        'cont_precision': cont_precision,
        'usecols_kwargs': usecols_kwargs
    }

    config_filename = r'out/config.json'
    save_json(listOfDicts, config_filename)

    obj = load_json(config_filename)
    assert listOfDicts == obj, 'Save and load are not equal.'

else:
    print('Importing persistence')
