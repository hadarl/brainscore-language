from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

from tqdm import tqdm
from brainio.assemblies import merge_data_arrays
from brainscore_language import load_benchmark
import warnings
warnings.filterwarnings("ignore", message="FutureWarning: unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.")

import time

st = time.time()

model_identifier = 'distilgpt2'

model = HuggingfaceSubject(model_id=model_identifier, region_layer_mapping={})

benchmark = load_benchmark('Willett2023-linear')

layer_scores = []

from brainscore_language import score

for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(5)], desc='layers'):
    print(layer)
    layer_model = HuggingfaceSubject(model_id=model_identifier, region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: layer})
    #print(layer_model.basemodel)
    layer_score = benchmark(layer_model)
    # package for xarray
    layer_score = layer_score.expand_dims('layer')
    layer_score['layer'] = [layer]
    layer_scores.append(layer_score)
layer_scores = merge_data_arrays(layer_scores)
print(layer_scores)

et = time.time()

elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')