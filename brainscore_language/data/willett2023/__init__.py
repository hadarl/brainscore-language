from brainscore_language import data_registry
from brainscore_language.data.willett2023.data_packaging  import load_willett2023_reference
#from datasets import load_dataset


def load_assembly():
    #dataset = load_dataset('willett2023')
    # lines = dataset['text']
    assembly = load_willett2023_reference()
    #lines = dataset['text']
    return assembly

data_registry['Willett2023-linear'] = load_assembly
