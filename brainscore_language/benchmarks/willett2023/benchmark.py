# [HADAR] NOTE: FOR TESTING, I REMOVED CEILING NORMALIZATION

import xarray as xr

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_core.metrics import Score
from brainscore_language import load_dataset, load_metric
from brainscore_language.artificial_subject import ArtificialSubject
# from brainscore_language.benchmarks.blank2014.ceiling import ExtrapolationCeiling
# from brainscore_language.data.blank2014 import BIBTEX
from brainscore_language.utils.ceiling import ceiling_normalize


class Willett2023Linear(BenchmarkBase):

    def __init__(self):
        self.data = load_dataset('Willett2023-linear')
        self.metric = load_metric('linear_pearsonr')
        ceiler = ExtrapolationCeiling()
        ceiling = ceiler(assembly=self.data, metric=self.metric)
        super(Willett2023Linear, self).__init__(
            identifier='Willett2023-linear',
            version=1,
            parent='neural_language',
            ceiling=ceiling,
            bibtex=BIBTEX)

    def __call__(self, candidate: ArtificialSubject) -> Score:
        candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                         recording_type=ArtificialSubject.RecordingType.fMRI)
        stimuli = self.data['stimulus']
        stories = self.data['story'].values
        predictions = []
        for story in sorted(set(stories)):  # go over individual stories, sorting to keep consistency across runs
            story_indexer = [stimulus_story == story for stimulus_story in stories]
            story_stimuli = stimuli[story_indexer]
            story_predictions = candidate.digest_text(story_stimuli.values)['neural']
            story_predictions['stimulus_id'] = 'presentation', story_stimuli['stimulus_id'].values
            predictions.append(story_predictions)
        predictions = xr.concat(predictions, dim='presentation')
        raw_score = self.metric(predictions, self.data)
        # score = ceiling_normalize(raw_score, self.ceiling)
        return raw_score
