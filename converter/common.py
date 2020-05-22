#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301
"""common module."""
from enum import Enum, unique

KaldiOps = [
    'Affine',
    'Append',
    'BatchNorm',
    'Dropout',
    'Linear',
    'LogSoftmax',
    'NoOp',
    'Offset',
    'Permute',
    'Relu',
    'ReplaceIndex',
    'Scale',
    'Sum',
    'Subsample',
    'Splice',
    'Tdnn',
    'Input',
    'Output',
]

KaldiOpType = Enum('KaldiOpType', [(op, op) for op in KaldiOps], type=str)

KaldiOpRawType = {
    "input-node": 'Input',
    "output-node": 'Output',
    "AffineComponent": 'Gemm',
    "BatchNormComponent": 'BatchNorm',
    "FixedAffineComponent": 'Gemm',
    "GeneralDropoutComponent": 'Dropout',
    "LinearComponent": 'Linear',
    "LogSoftmaxComponent": 'LogSoftmax',
    "NaturalGradientAffineComponent": 'Gemm',
    "NonlinearComponent": 'Nonlinear',
    "NoOpComponent": "NoOp",
    "PermuteComponent": 'Permute',
    "RectifiedLinearComponent": 'Relu',
    "RepeatedAffineComponent": 'Gemm',
    "ScaleComponent": 'Scale',
    "TdnnComponent": 'Tdnn',
}


@unique
class Descriptor(Enum):
  """Kaldi nnet3 descriptor."""

  Append = "Append"
  Offset = "Offset"
  ReplaceIndex = "ReplaceIndex"
  Scale = "Scale"
  Sum = "Sum"


Components = [
    'AffineComponent',
    'BatchNormComponent',
    'FixedAffineComponent',
    'GeneralDropoutComponent',
    'LinearComponent',
    'LogSoftmaxComponent',
    'NaturalGradientAffineComponent',
    'NonlinearComponent',
    'NoOpComponent',
    'PermuteComponent',
    'RectifiedLinearComponent',
    'TdnnComponent',
]
Component = Enum('Component', [(op, op) for op in Components], type=str)


ATTRIBUTE_NAMES = {
    # KaldiOpType.Gemm.name: ['num_repeats', 'num_blocks'],
    KaldiOpType.BatchNorm.name: ['dim',
                                 'block_dim',
                                 'epsilon',
                                 'target_rms',
                                 'count',
                                 'test_mode'],
    KaldiOpType.Dropout.name: ['dim'],
    KaldiOpType.ReplaceIndex.name: ['var_name',
                                    'value',
                                    'chunk_size',
                                    'left_context',
                                    'right_context'],
    KaldiOpType.Linear.name: ['rank_inout',
                              'updated_period',
                              'num_samples_history',
                              'alpha'],
    # KaldiOpType.Nonlinear.name: ['count', 'block_dim'],
    KaldiOpType.Offset.name: ['offset'],
    KaldiOpType.Scale.name: ['scale', 'dim'],
    KaldiOpType.Splice.name: ['dim',
                              'left_context',
                              'right_context',
                              'context',
                              'input_dim',
                              'output_dim',
                              'const_component_dim'],
}

CONSTS_NAMES = {
    # KaldiOpType.Gemm.name: ['params', 'bias'],
    KaldiOpType.BatchNorm.name: ['stats_mean', 'stats_var'],
    KaldiOpType.Linear.name: ['params'],
    # KaldiOpType.Nonlinear.name: ['value_avg', 'deriv_avg', 'value_sum', 'deriv_sum'],
    KaldiOpType.Permute.name: ['column_map', 'reorder'],
    KaldiOpType.Tdnn.name: ['time_offsets', 'params', 'bias'],
}
