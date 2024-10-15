from brainscore_language import data_registry
from brainscore_language.data.t17speech2024.data_packaging import load_t17speech2024_reference
#from datasets import load_dataset


def load_assembly():
    #dataset = load_dataset('willett2023')
    # lines = dataset['text']
    assembly = load_t17speech2024_reference()
    #lines = dataset['text']
    return assembly

data_registry['T17speech2024-linear'] = load_assembly
