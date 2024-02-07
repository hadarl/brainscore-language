# [HADAR] NOTE: FOR TESTING, I REMOVED CEILING NORMALIZATION
import logging
import xarray as xr
import numpy as np
from numpy.random import RandomState

from brainio.assemblies import DataAssembly
from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_core.metrics import Score, Metric
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.benchmarks.willett2023.ceiling_packaging import ExtrapolationCeiling
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
        self.metric = load_metric ('linear_pearsonr')
        cons_metric = load_metric ('pearsonr')
        # ceiler = ExtrapolationCeiling()
        #ceiling = ceiler(assembly=self.data, metric=self.metric)
        ceiler = SplitHalvesConsistency(num_splits=10, split_coordinate='subject', consistency_metric=cons_metric)  #self.metric)
        ceiling = ceiler(self.data)
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