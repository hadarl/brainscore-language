# [HADAR] NOTE: FOR TESTING, I REMOVED CEILING NORMALIZATION
import logging
import xarray as xr
import numpy as np
from numpy.random import RandomState
from tqdm import tqdm, trange
import itertools

from brainio.assemblies import array_is_element, walk_coords, DataAssembly, merge_data_arrays
from brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_core.metrics import Score, Metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.benchmarks.willett2023.ceiling_packaging import ExtrapolationCeiling, HoldoutSubjectCeilingWordAverage

# from brainscore_language.data.blank2014 import BIBTEX
# from brainscore_language.utils.ceiling import ceiling_normalize
#from brainscore_language.data.willett2023 import load_willett2023_reference

logger = logging.getLogger(__name__)

BIBTEX = """@proceedings{merity2017pointer,
  title={Pointer sentinel mixture models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  conference={International Conference on Learning Representations (ICLR)},
  url={https://openreview.net/forum?id=Byj72udxe},
  year={2016}
}"""


class Willett2023Linear(BenchmarkBase):

    def __init__(self):
        self.data = load_dataset('Willett2023-linear')
        #load_willett2023_reference()
        self.metric = load_metric('linear_pearsonr')
        cons_metric = load_metric('pearsonr')
        # ceiler = ExtrapolationCeiling()
        #ceiling = ceiler(assembly=self.data, metric=self.metric)
        # ceiler = SplitHalvesConsistency(num_splits=10, split_coordinate='subject', consistency_metric=cons_metric)  #self.metric)
        ceiler = InternalConsistency(num_splits=10, split_coordinate='subject', consistency_metric=cons_metric)  # self.metric)
        #subject_column = 'subject_id'
        #ceiler = HoldoutSubjectCeilingWordAverage(subject_column=subject_column)  # self.metric)
        ceiling = ceiler(assembly=self.data)

        super(Willett2023Linear, self).__init__(
            identifier='Willett2023-linear',
            version=1,
            parent='neural_language',
            ceiling=ceiling,
            bibtex='BIBTEX')

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)

        words = self.data['word']
        predictions = []
        for i in range(words.size):  #word in words.values:  # go over individual stories, sorting to keep consistency across runs
            # word_indexer = [stimulus_word == word for stimulus_word in words]
            #word_stimuli = self.data[word_indexer]
            word = str(words[i].data)   #self.data[]
            word_predictions = candidate.digest_text(word)['neural']
            #word_predictions['stimulus_id'] = 'presentation', words['stimulus_id'].values[i]  # word_stimuli['stimulus_index'].values
            predictions.append(word_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        predictions['stimulus_id'] = 'presentation', words['stimulus_id'].values
        raw_score = self.metric(predictions[:,1:257], self.data[:,:,1])
        print(raw_score)
        # score = ceiling_normalize(raw_score, self.ceiling)
        return raw_score

    # used implementation from here:
    # https: // cancerdatascience.org / blog / posts / pearson - correlation /



############# Adopted from brainscore_vision in metrics/internal_consistency/ceiling.py
    class SplitHalfWrapper:
        def __init__(self, split_coord, consistency_metric: Metric, correction):
            self._split_coord = split_coord
            self._consistency_metric = consistency_metric
            self._correction = correction

        def __call__(self, half1, half2):
            half1, half2 = self._average_repetitions(half1), self._average_repetitions(half2)
            consistency = self._consistency_metric(half1, half2)
            corrected_consistency = self._correction(consistency, n=2)
            return corrected_consistency

        def _average_repetitions(self, assembly):
            repetition_dims = assembly[self._split_coord].dims
            nonrepetition_coords = [coord for coord, dims, values in walk_coords(assembly)
                                    if dims == repetition_dims and coord != self._split_coord]
            average = assembly.multi_groupby(nonrepetition_coords).mean(dim=repetition_dims)
            return average

#############


class InternalConsistency:
    # following
    # https://github.com/brain-score/brain-score/blob/c51b8aa2c94212a9ac56c06c556afad0bb0a3521/brainscore/metrics/ceiling.py#L25-L96
    # and
    # https://github.com/brain-score/vision/blob/70bc577c1043f93cbab6b3f414180e6e14a0bcd8/brainscore_vision/metrics/internal_consistency/ceiling.py#L75

    def __init__(self, num_splits: int, split_coordinate: str, consistency_metric: Metric):
        """
        :param num_splits: how many times to create two halves
        :param split_coordinate: over which coordinate to split the assembly into halves
        :param consistency_metric: which metric to use to compute the consistency of two halves
        """
        self.num_splits = num_splits
        self.split_coordinate = split_coordinate
        self.consistency_metric = consistency_metric


    def __call__(self, assembly: DataAssembly) -> Score:
        split_dim = np.array(assembly[self.split_coordinate].dims).item()

        num_subjects = len(set(assembly['subject'].values))
        subject_subsamples = self.build_subject_subsamples(num_subjects) # This is the number of subjects (sessions) to include in the subsample.
        consistencies, uncorrected_consistencies = [], []
        for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
            selection_combinations = self.iterate_subsets(assembly, num_subjects=num_subjects)
            for selections, sub_assembly in tqdm(selection_combinations, desc='selections'):

                split_values = np.unique(sub_assembly['subject'].values)
                random_state = RandomState(0)
                #consistencies, uncorrected_consistencies = [], []
                splits = range(self.num_splits)
                # probably need to add here a loop over splits
                # cut the selections (subset of sessions) into two halves
                half1_values = random_state.choice(split_values, size=len(split_values) // 2, replace=False)
                half2_values = list(set(split_values) - set(half1_values))  # this only works because of `replace=False` above

                split_half1_indices = (sub_assembly['subject'] == half1_values).values
                split_half2_indices = (sub_assembly['subject'] == half2_values).values


                # half1 = sub_assembly[{split_dim: [value in half1_values for value in split_half1_indices]}]   #.mean(split_dim)
                # half2 = sub_assembly[{split_dim: [value in half2_values for value in split_half2_indices]}]   #.mean(split_dim)

                half1 = sub_assembly[split_half1_indices, :, :]
                half2 = sub_assembly[split_half2_indices, :, :]

                # pool_words_average = pool_assembly.multi_groupby(nonrepetition_coords).mean(dim=repetition_dims)

                # then: find all overlapping words
                stimuli_set_half1 = set(half1['word'].values)
                stimuli_set_half2 = set(half2['word'].values)
                overlapping_stimuli = stimuli_set_half1.intersection(stimuli_set_half2)
                overlapping_stimuli = list(overlapping_stimuli)
                stimuli_set_half1 = list(stimuli_set_half1)
                stimuli_set_half2 = list(stimuli_set_half2)

                # group half1, half2 by words, and average neural data over duplicate words.
                half1_grouped_by_word = half1.groupby('word').median(dim='presentation').sortby('word')
                half2_grouped_by_word = half2.groupby('word').median(dim='presentation').sortby('word')

                # keep only overlapping words from half1 and half2
                overlapping_stimuli = sorted(overlapping_stimuli)
                half1_grouped_by_word_overlap = half1_grouped_by_word.sel(word=overlapping_stimuli)
                half2_grouped_by_word_overlap = half2_grouped_by_word.sel(word=overlapping_stimuli)

                # Each assembly should have elements as the number of unique words
                assert(len(half1_grouped_by_word_overlap) == len(overlapping_stimuli))
                assert(len(half2_grouped_by_word_overlap) == len(overlapping_stimuli))

                # plot_activations_per_word(half2_grouped_by_word_overlap[0:10,:,:])
                # plot_histogram_neuroid(half2_grouped_by_word_overlap, 195)
                # max_indices = np.unravel_index(np.nanargmax(assembly.values), assembly.values.shape)
                # assembly_clipped = assembly.copy(data=np.clip(assembly.values, a_min=None, a_max=np.nanpercentile(assembly.values,99.9)))
                # plot_histogram_neuroid(clipped_assembly, 195)
                # assembly_averaged_over_timebins = assembly_clipped.mean(dim='time_bin')
                # neuron_means = assembly_averaged_over_timebins.mean(dim='presentation')
                # assembly_averaged_over_timebins_normalized = assembly_averaged_over_timebins/neuron_means
                # neuron_activation_per_word = assembly_averaged_over_timebins_normalized.groupby('word').mean(dim='presentation').sortby('word')

                # neuron_ind = 2
                # specific_neuroid_data = neuron_activation_per_word.sel(neuroid=neuron_ind)
                # Plot the values for the specific neuroid
                # specific_neuroid_data['word'] = specific_neuroid_data['word'].astype(str)
                # specific_neuroid_data.plot(x='word')

                # Comparing the two halves:
                # half1_grouped_by_word_overlap_tim_averaged = half1_grouped_by_word_overlap.mean(dim='time_bin')
                # half2_grouped_by_word_overlap_tim_averaged = half2_grouped_by_word_overlap.mean(dim='time_bin')

                # neuron_means = half1_grouped_by_word_overlap_tim_averaged.mean(dim='word')
                # half1_time_averaged_normalized = half1_grouped_by_word_overlap_tim_averaged/neuron_means
                # neuron_means = half2_grouped_by_word_overlap_tim_averaged.mean(dim='word')
                # half2_time_averaged_normalized = half2_grouped_by_word_overlap_tim_averaged/neuron_means

                # neuron_ind = 2
                # specific_neuroid_data_half1 = half1_time_averaged_normalized.sel(neuroid=neuron_ind)
                # specific_neuroid_data_half2 = half2_time_averaged_normalized.sel(neuroid=neuron_ind)

                # Plot the values for the specific neuroid
                # specific_neuroid_data_half1['word'] = specific_neuroid_data_half1['word'].astype(str)
                # specific_neuroid_data_half2['word'] = specific_neuroid_data_half2['word'].astype(str)
                # specific_neuroid_data_half1.plot(x='word')
                # plt.show()
                # specific_neuroid_data_half2.plot(x='word')
                # plt.show()

                # fig, axs = plt.subplots(nrows=2)
                # specific_neuroid_data_half1.plot(x='word', ax=axs[0])
                # specific_neuroid_data_half2.plot(x='word', ax=axs[1])
                # plt.tight_layout()
                # plt.show(figsize=(10, 6))
                #
                #
                # specific_neuroid_data_half1.plot(x='word')
                # specific_neuroid_data_half2.plot(x='word')
                # plt.tight_layout()
                # plt.show(figsize=(10, 3))
                #

                # plt.scatter(specific_neuroid_data_half1.values, specific_neuroid_data_half2.values, label=f'Neuroid {neuron_ind}', marker='o')
                #
                # # Add labels and title
                # plt.xlabel('Normalized Values - Half 1')
                # plt.ylabel('Normalized Values - Half 2')
                # plt.title(f'Scatter Plot for Neuroid {neuron_ind}')
                # plt.legend()
                #
                # # Show the modified figure size
                # plt.show()


                # Calculate correlation between the two averaged responses (one vector per half)
                consistency = self.consistency_metric(half1[:, 1], half2[:, 1])
                uncorrected_consistencies.append(consistency)
                # Spearman-Brown correction for sub-sampling
                # corrected_consistency = 2 * consistency / (1 + (2 - 1) * consistency)
                # consistencies.append(corrected_consistency)
                
            consistencies = Score(consistencies, coords={'split': splits}, dims=['split'])
            uncorrected_consistencies = Score(uncorrected_consistencies, coords={'split': splits}, dims=['split'])
            average_consistency = consistencies.median('split')
            average_consistency.attrs['raw'] = consistencies
            average_consistency.attrs['uncorrected_consistencies'] = uncorrected_consistencies
            return average_consistency


    def build_subject_subsamples(self, num_subjects):
        return tuple(range(2, num_subjects + 1))

    def iterate_subsets(self, assembly, num_subjects):
        subjects = set(assembly['subject'].values)
        subject_combinations = list(itertools.combinations(sorted(subjects), num_subjects))
        for sub_subjects in subject_combinations:
            # selected_indices = {'presentation': [subject in sub_subjects for subject in assembly[self.subject_column].values]}
            selected_indices = [subject in sub_subjects for subject in assembly['subject'].values]
            sub_assembly = assembly[selected_indices,:,:]
            yield {'subject': sub_subjects}, sub_assembly

    def average_collected(self, scores):
        return scores.median('neuroid')



class SplitHalvesConsistency:
    # following
    # https://github.com/brain-score/brain-score/blob/c51b8aa2c94212a9ac56c06c556afad0bb0a3521/brainscore/metrics/ceiling.py#L25-L96


    def __init__(self, num_splits: int, split_coordinate: str, consistency_metric: Metric):
        """
        :param num_splits: how many times to create two halves
        :param split_coordinate: over which coordinate to split the assembly into halves
        :param consistency_metric: which metric to use to compute the consistency of two halves
        """
        self.num_splits = num_splits
        self.split_coordinate = split_coordinate
        self.consistency_metric = consistency_metric

    def __call__(self, assembly: DataAssembly) -> Score:
        split_dim = np.array(assembly[self.split_coordinate].dims).item()
        split_values = assembly[self.split_coordinate].values
        random_state = RandomState(0)
        consistencies, uncorrected_consistencies = [], []
        splits = range(self.num_splits)
        for _ in splits:
            half1_values = random_state.choice(split_values, size=len(split_values) // 2, replace=False)
            half2_values = set(split_values) - set(half1_values)  # this only works because of `replace=False` above
            half1 = assembly[{split_dim: [value in half1_values for value in split_values]}].mean(split_dim)
            half2 = assembly[{split_dim: [value in half2_values for value in split_values]}].mean(split_dim)



            consistency = self.consistency_metric(half1[:,1], half1[:,1])
            uncorrected_consistencies.append(consistency)
            # Spearman-Brown correction for sub-sampling
            corrected_consistency = 2 * consistency / (1 + (2 - 1) * consistency)
            consistencies.append(corrected_consistency)
        consistencies = Score(consistencies, coords={'split': splits}, dims=['split'])
        uncorrected_consistencies = Score(uncorrected_consistencies, coords={'split': splits}, dims=['split'])
        average_consistency = consistencies.median('split')
        average_consistency.attrs['raw'] = consistencies
        average_consistency.attrs['uncorrected_consistencies'] = uncorrected_consistencies
        return average_consistency
