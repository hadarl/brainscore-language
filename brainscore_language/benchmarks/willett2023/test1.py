import matplotlib.pyplot as plt

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
layer_scores_backup = layer_scores
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

layer_scores_std_ordered = []
for layer in layer_names:
    layer_scores_std_ordered.append(np.std(layer_scores.sel(layer=layer).raw.mean('neuroid')).data)

print(layer_scores_ordered)
print(layer_scores_std_ordered)

def plot_score_vs_layer(layer_scores_ordered, layer_scores_std_ordered):

    import numpy as np
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(figsize=(10,10))
    x = np.arange(len(layer_scores_ordered))
    #ax.scatter(x, layer_scores_ordered)
    ax.errorbar(x, layer_scores_ordered, yerr=layer_scores_std_ordered, fmt='o')
    ax.set_xticks(x)
    fig.subplots_adjust(bottom=0.3, left=0.2)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('score')
    # ax.legend()
    #plt.show()

    #fig.savefig('Willett2023_gpt2-large.png')

#####

layer_scores_per_neuroid = []
layer_scores_per_neuroid_reordered = []
for layer in layer_names:
    data = db['layer_scores'].sel(layer=layer).raw.mean('split').data
    data = data.reshape(256,1)
    data_reordered = data[electrode_mapping]
    print(np.shape(data))
    layer_scores_per_neuroid.append(data)
    layer_scores_per_neuroid_reordered.append(data_reordered)

plt.imshow(layer_scores_per_neuroid_reordered[0].reshape(16,16))
plt.colorbar()
plt.show()

#############

layer_scores_per_neuroid = []
layer_scores_per_neuroid_reordered = []
for layer in range(36):
    data = layer_scores_backup[layer].raw.mean('split').data
    data = data.reshape(256,1)
    data_reordered = data[electrode_mapping]
    layer_scores_per_neuroid.append(data)
    layer_scores_per_neuroid_reordered.append(data_reordered)

plt.imshow(layer_scores_per_neuroid_reordered[8].reshape(16,16))
plt.colorbar()
plt.show()
##############


layer_scores_per_array = layer_scores_per_neuroid_reordered[8].reshape(16,16)[8:16,8:16]
plt.imshow(layer_scores_per_array)
plt.colorbar()
plt.show()

###################

layer_scores_per_neuroid_reordered_reshaped = []

for layer in range(36):
    arr_all = []
    arr = layer_scores_per_neuroid_reordered[layer].reshape(16, 16)[0:8,0:8]
    arr_all.append(arr.reshape(64,1))
    arr = layer_scores_per_neuroid_reordered[layer].reshape(16, 16)[0:8,8:16]
    arr_all.append(arr.reshape(64,1))
    arr = layer_scores_per_neuroid_reordered[layer].reshape(16, 16)[8:16, 0:8]
    arr_all.append(arr.reshape(64,1))
    arr = layer_scores_per_neuroid_reordered[layer].reshape(16, 16)[8:16, 8:16]
    arr_all.append(arr.reshape(64,1))
    layer_scores_per_neuroid_reordered_reshaped.append(arr_all)

layer_scores_per_neuroid_reordered_reshaped = np.squeeze(layer_scores_per_neuroid_reordered_reshaped)

array_pos = ["IFG_sup", "6v_sup", "IFG_inf", "6v_inf"]
layer_scores_per_array = xr.DataArray(layer_scores_per_neuroid_reordered_reshaped,
                                      coords = [layer_names, array_pos, range(64)], dims = ["layers", "array", "neuroid"])

means = layer_scores_per_array.mean('neuroid')
stds = layer_scores_per_array.std('neuroid')

fig, ax = plt.subplots(figsize=(10, 10))
for array_ind in range(4):
    plot_score_vs_layer(means[:,array_ind], stds[:,array_ind])
ax.legend(array_pos)
plt.show()
#fig.show()








#######################
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


####

ceilings = []
for i in range(11):
    ceilings.append(benchmark.ceiling[i].raw)
ceilings = sum(ceilings, [])

binwidth = (max(ceilings) - min(ceilings))/50
for i in range(11):
    plt.hist(benchmark.ceiling[i].raw, bins=np.arange(min(ceilings), max(ceilings) + binwidth, binwidth))
    # plt.pause(0.5)
n = [num for num in range(2, 24) if num % 2 == 0]
plt.legend(n)
plt.show()

plt.hist(ceilings,50)


########

# fig, axs = plt.subplots(2, 2)
# plt.imshow(layer_scores_per_neuroid[0][0:64].reshape(8,8))

def plot_scores_per_array(layer_scores_per_neuroid, electrode_mapping):

    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import colors

    Nr = 2
    Nc = 2

    fig, axs = plt.subplots(Nr, Nc)
    fig.suptitle('Scores per array')


    images = []
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            array_ind = i*2 + j
            #data = electrode_mapping
            data = layer_scores_per_neuroid[array_ind][(array_ind*64):(array_ind+1)*64].reshape(8,8)
            images.append(axs[i, j].imshow(data))
            axs[i, j].label_outer()

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())


    for im in images:
        im.callbacks.connect('changed', update)

    plt.show()



def plot_scores_per_layer(layer_scores_per_neuroid_reordered, electrode_mapping):

    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import colors

    Nr = 2
    Nc = 3

    fig, axs = plt.subplots(Nr, Nc)
    fig.suptitle('Scores per layer')


    images = []
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            layer_ind = i*Nc + j
            #data = electrode_mapping
            data = layer_scores_per_neuroid_reordered[layer_ind].reshape(16,16)
            images.append(axs[i, j].imshow(data))
            axs[i, j].label_outer()


    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())


    for im in images:
        im.callbacks.connect('changed', update)

    plt.show()






#####

electrode_mapping = \
               [192, 193, 208, 216, 160, 165, 178, 185,      62,  51,  43,  35,  94,  87,  79,  78,
                194, 195, 209, 217, 162, 167, 180, 184,      60,  53,  41,  33,  95,  86,  77,  76,
         		196, 197, 211, 218, 164, 170, 177, 189,      63,  54,  47,  44,  93,  84,  75,  74,
				198, 199, 210, 219, 166, 174, 173, 187,      58,  55,  48,  40,  92,  85,  73,  72,
				200, 201, 213, 220, 168, 176, 183, 186,      59,  45,  46,  38,  91,  82,  71,  70,
				202, 203, 212, 221, 172, 175, 182, 191,      61,  49,  42,  36,  90,  83,  69,  68,
				204, 205, 214, 223, 161, 169, 181, 188,      56,  52,  39,  34,  89,  81,  67,  66,
				206, 207, 215, 222, 163, 171, 179, 190,      57,  50,  37,  32,  88,  80,  65,  64,
                129, 144, 150, 158, 224, 232, 239, 255,     125, 126, 112, 103,  31,  28,  11,   8,
				128, 142, 152, 145, 226, 233, 242, 241,     123, 124, 110, 102,  29,  26,   9,   5,
				130, 135, 148, 149, 225, 234, 244, 243,     121, 122, 109, 101,  27,  19,  18,   4,
				131, 138, 141, 151, 227, 235, 246, 245,     119, 120, 108, 100,  25,  15,  12,   6,
				134, 140, 143, 153, 228, 236, 248, 247,     117, 118, 107,  99,  23,  13,  10,   3,
				132, 146, 147, 155, 229, 237, 250, 249,     115, 116, 106,  97,  21,  20,   7,   2,
				133, 137, 154, 157, 230, 238, 252, 251,     113, 114, 105,  98,  17,  24,  14,   0,
				136, 139, 156, 159, 231, 240, 254, 253,     127, 111, 104,  96,  30,  22,  16,   1]


db = {}
db['model'] = model
db['benchmark'] = benchmark
db['layer_scores'] = layer_scores
db['layer_score'] = layer_score
db['layer_scores_backup'] = layer_scores_backup
db['model_identifier'] = model_identifier
db['ceilings'] = ceilings

dbfile = open('distilgpt2_Willett2023.pkl', 'ab')

import pickle
# source, destination
pickle.dump(db, dbfile)
dbfile.close()

#
dbfile = open('gpt2-large_Willett2023.pkl', 'rb')
    db = pickle.load(dbfile)
    for keys in db:
        print(keys, '=>', db[keys])
    dbfile.close()