import logging
import numpy as np
import os
import re
import scipy.io
import scipy.stats
import sys
from pathlib import Path
from tqdm import tqdm

from brainio.assemblies import NeuroidAssembly, walk_coords, merge_data_arrays
# from brainscore_language.utils.s3 import upload_data_assembly

_logger = logging.getLogger(__name__)

"""
The code in this file was run only once to initially upload the data, and is kept here for reference.
"""


# def upload_pereira2018(atlas):
#     assembly = load_pereira2018(atlas=atlas)
#     upload_data_assembly(assembly,
#                          assembly_identifier=f"Pereira2018.{atlas}")


# adapted from
# https://github.com/mschrimpf/neural-nlp/blob/cedac1f868c8081ce6754ef0c13895ce8bc32efc/neural_nlp/neural_data/fmri.py

def load_willett2023_reference():
    # assembly = load_willett2023_reference()

    data_dir = "../../../DATA/willett2023/competitionData/train/"
    # data_dir = "../DATA/willett2023/competitionData/train/"
    mat_files = [file for file in os.listdir(data_dir) if file.endswith(".mat")]

    counter = 0
    # stimuli = {}  # experiment -> stimuli
    assemblies = []
    for data_file_name in tqdm(mat_files, desc="Session"):
        print(data_file_name)
        # Construct the full path to the current file
        data_file = os.path.join(data_dir, data_file_name)
        data = scipy.io.loadmat(str(data_file))

        # spike_power = np.transpose(data['spikePow'])


        sentences = [sentence.item() for sentence in data['sentenceText']]
        stim = load_willett2023_stimuli(sentences)

        # assembly
        #data['examples'] = data['spikePow']
        spike_power_data = load_willett2023_spikePow(data, sentences, stim)
        neural_data_per_word = spike_power_data['word_neural_data']

        # meta = data['__header__']

        electrode_numbers = list(range(np.shape(data['spikePow'][0][0])[1]))

        sentence_for_word = []
        block_index_for_sentence = []
        for i in range(np.shape(sentences)[0]):
            nWords = len(stim['word_nums'][i])
            sen = sentences[i]
            sentence_for_word.extend(np.tile(sen, nWords))
            block_index_for_sentence.extend([data['blockIdx'][i, 0]]*nWords)

        all_words = [element for sublist in stim['sentence_words'] for element in sublist]

        neural_data_per_word_T = np.transpose(neural_data_per_word,(0,2,1))

        assembly = (NeuroidAssembly
                    (neural_data_per_word_T,
                     dims=('presentation', 'neuroid', 'time_bin'),
                     coords={
                         # 'experiment': ('presentation', ['T12 competition']),
                         'stimulus_index': ('presentation', range(len(neural_data_per_word))),
                         'stimulus_id': ('presentation', range(len(neural_data_per_word))),
                         'sentence_index': ('presentation', [element for sublist in stim['sentence_id'] for element in sublist]),
                         'word_index_in_sentence': ('presentation', [element for sublist in stim['word_nums'] for element in sublist]),
                         'sentence': ('presentation', sentence_for_word),
                         'word': ('presentation', all_words),
                         'passage_index': ('presentation', block_index_for_sentence),
                         #'passage_label': ('presentation', [1] * np.shape(sentences)[0]),
                         #'passage_category': ('presentation', [1] * np.shape(sentences)[0]),
                         'session_num': ('presentation', [os.path.splitext(data_file_name)[0]] * len(neural_data_per_word)),
                         'session_index': ('presentation', [mat_files.index(data_file_name)] * len(neural_data_per_word)),
                         'subject': ('presentation', [os.path.splitext(data_file_name)[0]] * len(neural_data_per_word)),

                         'electrode': ('neuroid', electrode_numbers),
                         'neuroid_id': ('neuroid', electrode_numbers),

                         # 'time_bin_start': ('time_bin', spike_power_data['time_bin_start']),
                         # 'time_bin_end': ('time_bin', spike_power_data['time_bin_end']),
                         'time_bin': ('time_bin', range(np.shape(neural_data_per_word)[1])),
                     }))

        # stimulus_id = _build_id(assembly, ['experiment', 'stimulus_num'])
        # assembly['stimulus_id'] = 'presentation', stimulus_id
        # # set story for compatibility
        # assembly['story'] = 'presentation', _build_id(assembly, ['experiment', 'passage_category'])
        # assembly['neuroid_id'] = 'neuroid', _build_id(assembly, ['subject', 'voxel_num'])
        print(counter)
        counter += 1

        assemblies.append(assembly)

    all_sessions_assembly = merge_data_arrays(assemblies)

    return all_sessions_assembly

def load_willett2023_stimuli(sentences):  # Stimuli
    sentence_lst, sentence_id = [], []
    sentence_words_all, word_nums_all = [], []

    sentence_counter = 0
    for sentence in sentences:
        sentence = sentence.split(' ')
        sentence_words, word_nums = [], []
        word_counter = 0
        for word in sentence:
            if (word == '\n') | (word == ''):
                continue
            word = word.rstrip('\n')
            sentence_words.append(word)
            word_nums.append(word_counter)
            word_counter += 1

        # print(sentence_words)
        # print(word_nums)

        sentence_lst.append(sentence_counter)
        sentence_id.append(np.repeat(sentence_counter, word_counter))
        sentence_counter += 1

        sentence_words_all.append(sentence_words)
        word_nums_all.append(word_nums)

    #_logger.debug(sentence_words)

    stim = {'sentence_words': sentence_words_all, 'word_nums': word_nums_all, 'sentence_lst': sentence_lst, 'sentence_id': sentence_id}

    return stim


def load_willett2023_spikePow(data, sentences, stim):

    word_neural_data = []
    time_bin_start = []
    time_bin_end = []
    for i in range(np.shape(sentences)[0]):
        sentence = sentences[i].rstrip()
        nChars_per_sentence = len(sentence) + sentence.count(' ') # assuming space take more time (double) than a single character
        # An estimate of the number of samples for each word in the sentence:
        nNeural_samples_per_sentence = np.shape(data['spikePow'][0][i])[0]
        nNeural_samples_per_char = int(np.round(nNeural_samples_per_sentence / nChars_per_sentence))

        nWords_in_sentence = len(stim['sentence_words'][i])

        t = 0
        for j in range(nWords_in_sentence):
            time_bin_start.append(t)
            nChars_in_word = len(stim['sentence_words'][i][j])
            time_bin_end.append(t + nNeural_samples_per_char * nChars_in_word)
            t = t + nNeural_samples_per_char * nChars_in_word + 1
            word_neural_data.append(data['spikePow'][0][i][time_bin_start[-1]:time_bin_end[-1], :])

    sum = 0
    for i in range(np.shape(sentences)[0]):
        sum += len(stim['word_nums'][i])
    assert sum == len(word_neural_data), ...
    'The number of spike power word data is different from the number of words. Check your code'

    word_data = {'word_neural_data': word_neural_data, 'time_bin_start': time_bin_start, 'time_bin_end': time_bin_end}

    # just for testing: setting a manuat duration per word  = 8
    min_word_data_len = 8

    # Making them all in the same length:
    min_word_data_len = min(word_neural_data, key=lambda x: x.shape[0]).shape[0]
    word_data_cropped = []
    for i in range(len(word_neural_data)):
        word_data_cropped.append(word_neural_data[i][0:min_word_data_len, :])
    word_data = {'word_neural_data': word_data_cropped, 'time_bin_start': time_bin_start, 'time_bin_end': [min_word_data_len-1]*len(time_bin_start)}

    return word_data



# def load_pereira2018(atlas):
#     assembly = load_pereira2018_full()
#     assembly = assembly.sel(atlas=atlas, atlas_selection_lower=90)
#     assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
#                                      for filter_strategy in assembly['filter_strategy'].values]}]
#     return assembly


def _build_id(assembly, coords):
    return [".".join([f"{value}" for value in values]) for values in zip(*[assembly[coord].values for coord in coords])]


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    upload_pereira2018(atlas='language')
    upload_pereira2018(atlas='auditory')
