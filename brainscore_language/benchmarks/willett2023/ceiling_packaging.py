import itertools
import logging
import numpy as np
from numpy.random import RandomState
from scipy.optimize import curve_fit
from tqdm import tqdm, trange

from brainio.assemblies import array_is_element, walk_coords, DataAssembly, merge_data_arrays
from brainscore_core.metrics import Score
from brainscore_language.benchmark_helpers import ci_error, manual_merge
from brainscore_language.utils import fullname
from brainscore_language.utils.transformations import apply_aggregate


def v(x, v0, tau0):
    return v0 * (1 - np.exp(-x / tau0))


class ExtrapolationCeiling:
    def __init__(self, subject_column='session_index', extrapolation_dimension='neuroid', num_bootstraps=100):
        self._logger = logging.getLogger(fullname(self))
        self.subject_column = subject_column
        self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
        self.extrapolation_dimension = extrapolation_dimension
        self.num_bootstraps = num_bootstraps

    def __call__(self, assembly, metric):
        scores = self.collect(assembly=assembly, metric=metric)
        return self.extrapolate(scores)

    def collect(self, assembly, metric):
        num_subjects = len(set(assembly[self.subject_column].values))
        subject_subsamples = self.build_subject_subsamples(num_subjects)
        scores = []
        for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
            selection_combinations = self.iterate_subsets(assembly, num_subjects=num_subjects)
            for selections, sub_assembly in tqdm(selection_combinations, desc='selections'):
                score = self.holdout_ceiling(assembly=sub_assembly, metric=metric)
                score = score.expand_dims('num_subjects')
                score['num_subjects'] = [num_subjects]
                for key, selection in selections.items():
                    expand_dim = f'sub_{key}'
                    score = score.expand_dims(expand_dim)
                    score[expand_dim] = [str(selection)]
                scores.append(score.raw)
        scores = Score.merge(*scores)
        assert hasattr(scores, 'neuroid_id')
        return scores

    def build_subject_subsamples(self, num_subjects):
        return tuple(range(2, num_subjects + 1))

    def iterate_subsets(self, assembly, num_subjects):
        subjects = set(assembly[self.subject_column].values)
        subject_combinations = list(itertools.combinations(sorted(subjects), num_subjects))
        for sub_subjects in subject_combinations:
            # selected_indices = {'presentation': [subject in sub_subjects for subject in assembly[self.subject_column].values]}
            selected_indices = [subject in sub_subjects for subject in assembly[self.subject_column].values]
            sub_assembly = assembly[selected_indices,:,:]
            yield {self.subject_column: sub_subjects}, sub_assembly

    def average_collected(self, scores):
        return scores.median('neuroid')

    def extrapolate(self, ceilings):
        neuroid_ceilings, bootstrap_params, endpoint_xs = [], [], []
        for i in trange(len(ceilings[self.extrapolation_dimension]),
                        desc=f'{self.extrapolation_dimension} extrapolations'):
            # extrapolate per-neuroid ceiling
            neuroid_ceiling = ceilings.isel(**{self.extrapolation_dimension: [i]})
            extrapolated_ceiling = self.extrapolate_neuroid(neuroid_ceiling.squeeze())
            extrapolated_ceiling = self.add_neuroid_meta(extrapolated_ceiling, neuroid_ceiling)
            neuroid_ceilings.append(extrapolated_ceiling)
            # also keep track of bootstrapped parameters
            neuroid_bootstrap_params = extrapolated_ceiling.bootstrapped_params
            neuroid_bootstrap_params = self.add_neuroid_meta(neuroid_bootstrap_params, neuroid_ceiling)
            bootstrap_params.append(neuroid_bootstrap_params)
            # and endpoints
            endpoint_x = self.add_neuroid_meta(extrapolated_ceiling.endpoint_x, neuroid_ceiling)
            endpoint_xs.append(endpoint_x)
        # merge and add meta
        self._logger.debug("Merging neuroid ceilings")
        neuroid_ceilings = manual_merge(*neuroid_ceilings, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['raw'] = ceilings
        self._logger.debug("Merging bootstrap params")
        bootstrap_params = manual_merge(*bootstrap_params, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['bootstrapped_params'] = bootstrap_params
        self._logger.debug("Merging endpoints")
        endpoint_xs = manual_merge(*endpoint_xs, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['endpoint_x'] = endpoint_xs
        # aggregate
        ceiling = self.aggregate_neuroid_ceilings(neuroid_ceilings)
        return ceiling

    def add_neuroid_meta(self, target, source):
        target = target.expand_dims(self.extrapolation_dimension)
        for coord, dims, values in walk_coords(source):
            if array_is_element(dims, self.extrapolation_dimension):
                target[coord] = dims, values
        return target

    def aggregate_neuroid_ceilings(self, neuroid_ceilings):
        ceiling = neuroid_ceilings.median(self.extrapolation_dimension)
        ceiling.attrs['bootstrapped_params'] = neuroid_ceilings.bootstrapped_params.median(self.extrapolation_dimension)
        ceiling.attrs['endpoint_x'] = neuroid_ceilings.endpoint_x.median(self.extrapolation_dimension)
        ceiling.attrs['raw'] = neuroid_ceilings
        return ceiling

    def extrapolate_neuroid(self, ceilings):
        # figure out how many extrapolation x points we have. E.g. for Pereira, not all combinations are possible
        subject_subsamples = list(sorted(set(ceilings['num_subjects'].values)))
        rng = RandomState(0)
        bootstrap_params = []
        for bootstrap in range(self.num_bootstraps):
            bootstrapped_scores = []
            for num_subjects in subject_subsamples:
                num_scores = ceilings.sel(num_subjects=num_subjects)
                # the sub_subjects dimension creates nans, get rid of those
                num_scores = num_scores.dropna(f'sub_{self.subject_column}')
                assert set(num_scores.dims) == {f'sub_{self.subject_column}', 'split'} or \
                       set(num_scores.dims) == {f'sub_{self.subject_column}'}
                # choose from subject subsets and the splits therein, with replacement for variance
                choices = num_scores.values.flatten()
                bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                bootstrapped_scores.append(np.mean(bootstrapped_score))

            params = self.fit(subject_subsamples, bootstrapped_scores)
            params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                  dims=['bootstrap', 'param'])
            bootstrap_params.append(params)
        bootstrap_params = merge_data_arrays(bootstrap_params)
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values
                       if not np.isnan(params).any()])
        median_ys = np.median(ys, axis=0)
        diffs = np.diff(median_ys)
        end_x = np.where(diffs < asymptote_threshold)[0].min()  # first x where increase smaller than threshold
        # put together
        center = np.median(np.array(bootstrap_params)[:, 0])
        error_low, error_high = ci_error(ys[:, end_x], center=center)
        score = Score(center)
        score.attrs['error_low'] = error_low
        score.attrs['error_high'] = error_high
        score.attrs['raw'] = ceilings
        score.attrs['bootstrapped_params'] = bootstrap_params
        score.attrs['endpoint_x'] = DataAssembly(end_x)
        return score

    def fit(self, subject_subsamples, bootstrapped_scores):
        params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores,
                                 # v (i.e. max ceiling) is between 0 and 1, tau0 unconstrained
                                 bounds=([0, -np.inf], [1, np.inf]))
        return params


class HoldoutSubjectCeiling:
    def __init__(self, subject_column):
        self.subject_column = subject_column
        self._logger = logging.getLogger(fullname(self))

    def get_first_occurrence(group):
        return group.isel(presentation=0)

    def __call__(self, assembly, metric):
        subjects = set(assembly[self.subject_column].values)
        scores = []
        iterate_subjects = self.get_subject_iterations(subjects)
        for subject in tqdm(iterate_subjects, desc='heldout subject'):
            try:
                #subject_assembly = assembly[{'neuroid': [subject_value == subject
                #                                         for subject_value in assembly[self.subject_column].values]}]

                # Extracting data from the specific session (="subject")
                selected_indices = [subject_value == subject for subject_value in assembly[self.subject_column].values]
                subject_assembly = assembly[selected_indices, :, :]
                # run subject pool as neural candidate

                # Extracting data of all other sessions (="subjects")
                subject_pool = subjects - {subject}
                selected_indices_pool = [subject in subject_pool for subject in assembly[self.subject_column].values]
                pool_assembly = assembly[selected_indices_pool, :, :]
                #pool_assembly = assembly[
                #    {'presentation': [subject in subject_pool for subject in assembly[self.subject_column].values]}]

                # Finding the set of words that exist in both datasets:
                stimuli_set_pool = set(pool_assembly['word'].values)
                stimuli_set_subject = set(subject_assembly['word'].values)
                overlapping_stimuli = stimuli_set_pool.intersection(stimuli_set_subject)
                overlapping_stimuli = list(overlapping_stimuli)
                stimuli_set_pool = list(stimuli_set_pool)
                stimuli_set_subject = list(stimuli_set_subject)


                # Finding the indices of the overlapping words in the pool set (finds only first occurrence)
                first_occurrence_indices_pool = np.full_like(overlapping_stimuli, -1, dtype=int)
                # Find the first occurrence of each string element in array2
                for i, element in enumerate(overlapping_stimuli):
                    first_occurrence_indices_pool[i]  = stimuli_set_pool.index(element)

                # Finding the indices of the overlapping words in the subject set (finds only first occurrence)
                first_occurrence_indices_subject = np.full_like(overlapping_stimuli, -1, dtype=int)
                # Find the first occurrence of each string element in array2
                for i, element in enumerate(overlapping_stimuli):
                    first_occurrence_indices_subject[i] = stimuli_set_subject.index(element)

                # Extracting the assemblies of the overlapping words
                pool_assembly = pool_assembly[first_occurrence_indices_pool, :, :]
                subject_assembly = subject_assembly[first_occurrence_indices_subject, :, :]

                # PS, the reason for doing it in this very long way and not using intersect is because intersect obliterates most of the assembly fields

                #Problem: WE ARE NOT COMPARING THE SAME SENTENCES, BUT THE SAME WORDS COMING FROM DIFFERENT SENTENCES.
                # THEREFORE THE STIMULUS_ID IS NOT IDENTICAL. CURRENTLY I WILL BE IGNORING THIS AND OVERWRITE THE STIMULUS_ID SO THAT THEY CAN BE SCORED TOGETHER

                score = self.score(pool_assembly, subject_assembly, metric=metric)
                # store scores
                apply_raw = 'raw' in score.attrs and \
                            not hasattr(score.raw, self.subject_column)  # only propagate if column not part of score
                score = score.expand_dims(self.subject_column, _apply_raw=apply_raw)
                score.__setitem__(self.subject_column, [subject], _apply_raw=apply_raw)
                scores.append(score)
            except NoOverlapException as e:
                self._logger.debug(f"Ignoring no overlap {e}")
                continue  # ignore
            except ValueError as e:
                if "Found array with" in str(e):
                    self._logger.debug(f"Ignoring empty array {e}")
                    continue
                else:
                    raise e

        scores = Score.merge(*scores)
        score = apply_aggregate(lambda scores: scores.mean(self.subject_column), scores)
        score.attrs['error'] = scores.std(self.subject_column)
        return scores

    def get_subject_iterations(self, subjects):
        return subjects  # iterate over all subjects

    def score(self, pool_assembly, subject_assembly, metric):
        return metric(pool_assembly, subject_assembly)


class NoOverlapException(Exception):
    pass
