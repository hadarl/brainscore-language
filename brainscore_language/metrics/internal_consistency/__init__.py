# Adopted from brainscore-vision

from brainscore_language import metric_registry
from .ceiling import InternalConsistency

metric_registry['internal_consistency'] = InternalConsistency
