"""Reusable PB core package for generic experiments."""

from .interfaces import (
    BatchData,
    ContextBuilder,
    DatasetProvider,
    LossFn,
    MetricsFn,
    NoiseModel,
    NominalPlant,
    TruePlant,
)
from .factories import FactorizedBuildSpec, build_factorized_controller, infer_dims_from_probe
from .noise import DecayingGaussianNoise, ZeroNoise
from .registry import Registry
from .rollout import RolloutResult, rollout_pb
from .runner import PBExperimentRunner, RunnerConfig
from .validation import validate_component_compatibility

__all__ = [
    "BatchData",
    "ContextBuilder",
    "DatasetProvider",
    "LossFn",
    "MetricsFn",
    "NoiseModel",
    "NominalPlant",
    "TruePlant",
    "DecayingGaussianNoise",
    "ZeroNoise",
    "FactorizedBuildSpec",
    "build_factorized_controller",
    "infer_dims_from_probe",
    "Registry",
    "RolloutResult",
    "rollout_pb",
    "PBExperimentRunner",
    "RunnerConfig",
    "validate_component_compatibility",
]
