from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

from tqdm import tqdm
from brainio.assemblies import merge_data_arrays
from brainscore_language import load_benchmark
import warnings
warnings.filterwarnings("ignore", message="FutureWarning: unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.")

import time

st = time.time()

# model_identifier = 'distilgpt2'
model_identifier = 'gpt2-large'

model = HuggingfaceSubject(model_id=model_identifier, region_layer_mapping={})
print(model.basemodel)

benchmark = load_benchmark('Willett2023-linear')

#########

layer_scores = []

from brainscore_language import score

for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(36)], desc='layers'):
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


#########

layer_names = []
for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(36)], desc='layers'):
    layer_names.append(layer)

print(layer_names)

layer_scores_ordered = []
for layer in layer_names:
    layer_scores_ordered.append(layer_scores.sel(layer=layer).data)

print(layer_scores_ordered)

import numpy as np
from matplotlib import pyplot

fig, ax = pyplot.subplots()
x = np.arange(len(layer_scores_ordered))
ax.scatter(x, layer_scores_ordered)
ax.set_xticks(x)
ax.set_xticklabels(layer_names, rotation=90)
ax.set_ylabel('score')

fig.savefig('Willett2023_gpt2-large_10Feb.png')


def plot_activations_per_word(assembly):
    import matplotlib.pyplot as plt

    words = assembly.word.values

    # Create subplots for each word
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    for i, word in enumerate(words):
        ax = axes[i]
        data_for_word = assembly.sel(word=word)

        # Plot the data as a colored map
        im = ax.imshow(data_for_word.values.T, cmap='viridis', aspect='auto', origin='lower')

        # Add labels and title
        ax.set_title(f"Word: {word}")
        ax.set_xlabel('Neuroid Index')
        ax.set_ylabel('Time Bin')

    # Add a colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02)
    cbar.set_label('Mean Activity')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_histogram_neuroid(assembly, index):

    import matplotlib.pyplot as plt

    # Extract values for the 3rd neuroid
    neuroid_values = assembly.sel(neuroid=index).values.flatten()

    # Create a histogram
    plt.hist(neuroid_values, bins=50, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Neuroid Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Values for Neuroid {index}')

    # Show the plot
    plt.show()