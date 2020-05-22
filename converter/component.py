#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Nnet3 component."""
from abc import ABCMeta
from typing import Set, TextIO

from converter.utils import *


class Component(metaclass=ABCMeta):
  """Kaldi nnet3 component.

  Attributes:
    _params: params dict.
  """

  def __init__(self):
    """Initialize."""
    self._params = dict()

  @staticmethod
  def _actions():
    """Get actions for read different params.

    Returns:
      actions dict.
    """
    actions = {
        '<Dim>': (read_int, 'dim'),
        '<InputDim>': (read_int, 'input_dim'),
        '<OutputDim>': (read_int, 'output_dim')
    }
    return actions

  def read_params(self, line_buffer: TextIO, line: str, pos: int,
                  terminating_tokens: Set):
    """Read component params from line.

    Args:
      line_buffer: buffer of file.
      line: current line.
      pos: start position.
      terminating_tokens: set of terminating tokens.
    """
    actions = self._actions()

    while True:
      token, pos = read_next_token(line, pos)
      if token in terminating_tokens:
        break

      if token is None:
        line = next(line_buffer)
        if line is None:
          raise ValueError(f"Error parsing nnet3 file.")
        else:
          pos = 0
          continue

      if token in actions:
        func, name = actions[token]
        obj, pos = func(line, pos, line_buffer)
        self._params[name] = obj


class GeneralDropoutComponent(Component):
  """GeneralDropoutComponent."""


class NoOpComponent(Component):
  """NoOpComponent."""


class AffineComponent(Component):
  """AffineComponent."""

  def _actions(self):
    """See baseclass document."""
    actions = {
        '<LinearParams>': (read_matrix, 'params'),
        '<BiasParams>': (read_vector, 'bias'),
        '<MaxChange>': (read_float, 'max_change'),
        '<RankIn>': (read_int, 'rank_in'),
        '<RankOut>': (read_int, 'rank_out'),
        '<UpdatedPeriod>': (read_int, 'updated_period'),
        '<NumSamplesHistory>': (read_float, 'num_samples_history'),
        '<Alpha>': (read_float, 'alpha'),
        '<NumRepeats>': (read_int, 'num_repeats'),
        '<NumBlocks>': (read_int, 'num_blocks'),
    }
    return actions


class FixedAffineComponent(AffineComponent):
  """FixedAffineComponent."""


class NaturalGradientAffineComponent(AffineComponent):
  """FixedAffineComponent."""


class BatchNormComponent(Component):
  """BatchNormComponent."""

  def _actions(self):
    """See baseclass document."""
    actions = {
        '<Dim>': (read_int, 'dim'),
        '<BlockDim>': (read_int, 'block_dim'),
        '<Epsilon>': (read_float, 'epsilon'),
        '<TargetRms>': (read_float, 'target_rms'),
        '<Count>': (read_float, 'count'),
        '<StatsMean>': (read_vector, 'stats_mean'),
        '<StatsVar>': (read_vector, 'stats_var'),
        '<TestMode>': (read_bool, 'test_mode'),
    }
    return actions


class LinearComponent(Component):
  """LinearComponent."""

  def _actions(self):
    """See baseclass document."""
    actions = {
        '<Params>': (read_matrix, 'params'),
        '<RankInOut>': (read_int, 'rank_inout'),
        '<UpdatedPeriod>': (read_int, 'updated_period'),
        '<NumSamplesHistory>': (read_float, 'num_samples_history'),
        '<Alpha>': (read_float, 'alpha')
    }
    return actions


class NonlinearComponent(Component):
  """NonlinearComponent."""

  def _actions(self):
    """See baseclass document."""
    actions = {
        '<Dim>': (read_int, 'dim'),
        '<BlockDim>': (read_int, 'block_dim'),
        '<ValueAvg>': (read_vector, 'value_avg'),
        '<DerivAvg>': (read_vector, 'deriv_avg'),
        '<OderivRms>': (read_vector, 'oderiv_rms'),
        '<Count>': (read_float, 'count'),
        '<OderivCount>': (read_float, 'oderiv_count')
    }
    return actions


class LogSoftmaxComponent(NonlinearComponent):
  """LogSoftmaxComponent."""


class RectifiedLinearComponent(NonlinearComponent):
  """RectifiedLinearComponent."""


class PermuteComponent(Component):
  """PermuteComponent."""

  def _actions(self):
    """See baseclass document."""
    return {'<ColumnMap>': (read_vector, 'column_map')}


class TdnnComponent(Component):
  """TdnnComponent."""

  def _actions(self):
    """See baseclass document."""
    actions = {
        '<TimeOffsets>': (read_vector_int, 'time_offsets'),
        '<LinearParams>': (read_matrix, 'params'),
        '<BiasParams>': (read_vector, 'bias'),
        '<OrthonormalConstraint>': (read_float, 'orthonormal_constraint'),
        '<UseNaturalGradient>': (read_bool, 'use_natrual_gradient'),
        '<RankInOut>': (read_int, 'rank_inout'),
        '<NumSamplesHistory>': (read_float, 'num_samples_history'),
        '<Alpha>': (read_float, 'alpha'),
        '<AlphaInOut>': (read_float, 'alpha_inout'),
    }
    return actions
