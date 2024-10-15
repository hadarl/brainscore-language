import matplotlib.pyplot as plt

from brainscore_language.model_helpers.huggingface import HuggingfaceSubject
from brainscore_language import ArtificialSubject

from tqdm import tqdm
from brainio.assemblies import merge_data_arrays
from brainscore_language import load_benchmark
import warnings
warnings.filterwarnings("ignore", message="FutureWarning: unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.")

import time

import numpy as np
import xarray as xr
from brainscore_language.benchmarks.willett2023.plot_utils import *


st = time.time()

model_identifier = 'distilgpt2'
# model_identifier = 'gpt2-large'

model = HuggingfaceSubject(model_id=model_identifier, region_layer_mapping={})
print(model.basemodel)

#########

benchmark = load_benchmark('Willett2023-linear')

#########

layer_scores = []

from brainscore_language import score

for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(6)], desc='layers'):
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
for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(6)], desc='layers'):
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

##############

layer_scores_per_neuroid = []
layer_scores_per_neuroid_reordered = []
for layer in layer_names:
    data = db['layer_scores'].sel(layer=layer).raw.mean('split').data
    data = data.reshape(256,1)
    data_reordered = data[electrode_mapping]
    print(np.shape(data))
    layer_scores_per_neuroid.append(data)
    layer_scores_per_neuroid_reordered.append(data_reordered)

plt.imshow(layer_scores_per_neuroid_reordered[4].reshape(16,16))
plt.colorbar()
plt.show()

#############

tmp = db['layer_scores'].sel(layer='transformer.h.35.ln_1').raw.mean('split').data
plt.imshow(tmp)



################ # If db is not in variables
layer_scores_per_neuroid = []
layer_scores_per_neuroid_reordered = []
for layer in range(36):
    data = layer_scores_backup[layer].raw.mean('split').data
    data = data.reshape(256,1)
    data_reordered = data[electrode_mapping]
    layer_scores_per_neuroid.append(data)
    layer_scores_per_neuroid_reordered.append(data_reordered)

plt.imshow(layer_scores_per_neuroid_reordered[0].reshape(16,16))
plt.colorbar()
plt.show()
##############


layer_scores_per_array = layer_scores_per_neuroid_reordered[2].reshape(16,16)[8:16,8:16]
plt.imshow(layer_scores_per_array)
plt.colorbar()
plt.show()

###################

layer_scores_per_neuroid_reordered_reshaped = []

for layer in range(6):
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

import plot_utils

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






plot_scores_per_layer(layer_scores_per_neuroid_reordered, electrode_mapping)

#####



db = {}
db['model'] = model
db['benchmark'] = benchmark
db['layer_scores'] = layer_scores
db['layer_score'] = layer_score
db['layer_scores_backup'] = layer_scores_backup
db['model_identifier'] = model_identifier
db['ceilings'] = ceilings


#dbfile = open('gpt2-large_Willett2023_07Mar2024_768features.pkl', 'ab')
dbfile = open('distilgpt2_Willett2023_01Oct2024_768features.pkl', 'ab')
import pickle
# source, destination
pickle.dump(db, dbfile)
dbfile.close()

#
import pickle
dbfile = open('gpt2-large_Willett2023_07Mar2024_768features.pkl', 'rb')
db = pickle.load(dbfile)
for keys in db:
    print(keys, '=>', db[keys])
dbfile.close()


#
# import pickle
# dbfile = open('/home/hadarla/Projects/brainscore-language-fork/gpt2-large_Willett2023_07Mar2024_768features.pkl', 'rb')
# db = pickle.load(dbfile)
# for keys in db:
#     print(keys, '=>', db[keys])
# dbfile.close()


####################################


fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

for array_ind in range(4):

    x = np.arange(36)
    #linestyle = {"linestyle": "-", "linewidth": 4, "markeredgewidth": 5, "elinewidth": 5, "capsize": 10}
    ax.plot(x,means[:,array_ind],linewidth=4)


    # # ax.scatter(x, layer_scores_ordered)
    # # ax.errorbar(x, layer_scores_ordered, yerr=layer_scores_std_ordered, fmt='o',zorder=array_ind)
    # # ax.set_xticks(x)
    fig.subplots_adjust(bottom=0.3, left=0.2)
    # # ax.set_xticklabels(layer_names, rotation=45, ha='right')
    # # ax.set_ylabel('score')
    #
    # #ax.axis([0, 5, 0, 35])
    #linestyle = {"linestyle": "-", "linewidth": 4, "markeredgewidth": 5, "elinewidth": 5, "capsize": 10}
    # ax.errorbar(x, layer_scores_ordered, yerr=layer_scores_std_ordered, color=color[array_ind], **linestyle)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('score')

plt.xticks(np.arange(0,36))
plt.rcParams.update({'font.size': 20})

plt.legend(array_pos,loc=7)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
#fig.savefig('/home/hadarla/Projects/brainscore-language-fork/results/Brainscore_t17_meanLayerScore_allArrays.svg') #, bbox_inches='tight')





###############################


fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

for array_ind in range(4):

    x = np.arange(len(layer_scores_ordered))
    #linestyle = {"linestyle": "-", "linewidth": 4, "markeredgewidth": 5, "elinewidth": 5, "capsize": 10}
    ax.plot(x,means[:,array_ind],linewidth=4)


    # # ax.scatter(x, layer_scores_ordered)
    # # ax.errorbar(x, layer_scores_ordered, yerr=layer_scores_std_ordered, fmt='o',zorder=array_ind)
    # # ax.set_xticks(x)
    fig.subplots_adjust(bottom=0.3, left=0.2)
    # # ax.set_xticklabels(layer_names, rotation=45, ha='right')
    # # ax.set_ylabel('score')
    #
    # #ax.axis([0, 5, 0, 35])
    #linestyle = {"linestyle": "-", "linewidth": 4, "markeredgewidth": 5, "elinewidth": 5, "capsize": 10}
    # ax.errorbar(x, layer_scores_ordered, yerr=layer_scores_std_ordered, color=color[array_ind], **linestyle)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('score')

plt.xticks(np.arange(0,6))
plt.rcParams.update({'font.size': 20})

plt.legend(array_pos,loc=7)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
#fig.savefig('/home/hadarla/Projects/brainscore-language-fork/results/Brainscore_t17_meanLayerScore_allArrays.svg') #, bbox_inches='tight')
