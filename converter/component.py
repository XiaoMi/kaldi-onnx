#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Nnet3 component."""
from abc import ABCMeta
from enum import Enum, unique
from typing import Dict, Optional, Set, TextIO, Tuple

import numpy as np

from converter.common import KaldiOpRawType


class Component(metaclass=ABCMeta):
  """Kaldi nnet3 component.

  Attributes:
    __params: params dict.
  """

  def __init__(self):
    """Initialize."""
    self.__params = dict()

  @staticmethod
  def _actions() -> Dict:
    """Get actions for read different params.

    Returns:
      actions dict.
    """
    actions = {
        '<Dim>': (_read_int, 'dim'),
        '<InputDim>': (_read_int, 'input_dim'),
        '<OutputDim>': (_read_int, 'output_dim')
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
    self.__params['type'] = KaldiOpRawType[self.__class__.__name__]
    self.__params['raw-type'] = self.__class__.__name__[1:-10]
    actions = self._actions()

    while True:
      token, pos = read_next_token(line, pos)
      if token in terminating_tokens:
        break

      if token is None:
        line = next(line_buffer)
        if line is None:
          raise ValueError('Error parsing nnet3 file.')

        pos = 0
        continue

      if token in actions:
        func, name = actions[token]
        obj, pos = func(line, pos, line_buffer)
        self.__params[name] = obj

  def update_params(self, params_dict: Dict):
    """Update params.

    Args:
      params_dict: params dict.
    """
    self.__params.update(params_dict)


class GeneralDropoutComponent(Component):
  """GeneralDropoutComponent."""


class NoOpComponent(Component):
  """NoOpComponent."""


class AffineComponent(Component):
  """AffineComponent."""

  def _actions(self) -> Dict:
    """See baseclass document."""
    actions = {
        '<LinearParams>': (_read_matrix_trans, 'params'),
        '<BiasParams>': (_read_vector_float, 'bias'),
        '<MaxChange>': (_read_float, 'max_change'),
        '<RankIn>': (_read_int, 'rank_in'),
        '<RankOut>': (_read_int, 'rank_out'),
        '<UpdatedPeriod>': (_read_int, 'updated_period'),
        '<NumSamplesHistory>': (_read_float, 'num_samples_history'),
        '<Alpha>': (_read_float, 'alpha'),
        '<NumRepeats>': (_read_int, 'num_repeats'),
        '<NumBlocks>': (_read_int, 'num_blocks'),
    }
    return actions


class FixedAffineComponent(AffineComponent):
  """FixedAffineComponent."""


class NaturalGradientAffineComponent(AffineComponent):
  """FixedAffineComponent."""


class BatchNormComponent(Component):
  """BatchNormComponent."""

  def _actions(self) -> Dict:
    """See baseclass document."""
    actions = {
        '<Dim>': (_read_int, 'dim'),
        '<BlockDim>': (_read_int, 'block_dim'),
        '<Epsilon>': (_read_float, 'epsilon'),
        '<TargetRms>': (_read_float, 'target_rms'),
        '<Count>': (_read_float, 'count'),
        '<StatsMean>': (_read_vector_float, 'stats_mean'),
        '<StatsVar>': (_read_vector_float, 'stats_var'),
        '<TestMode>': (_read_bool, 'test_mode'),
    }
    return actions


class LinearComponent(Component):
  """LinearComponent."""

  def _actions(self) -> Dict:
    """See baseclass document."""
    actions = {
        '<Params>': (_read_matrix_trans, 'params'),
        '<RankInOut>': (_read_int, 'rank_inout'),
        '<UpdatedPeriod>': (_read_int, 'updated_period'),
        '<NumSamplesHistory>': (_read_float, 'num_samples_history'),
        '<Alpha>': (_read_float, 'alpha')
    }
    return actions


class NonlinearComponent(Component):
  """NonlinearComponent."""

  def _actions(self) -> Dict:
    """See baseclass document."""
    actions = {
        '<Dim>': (_read_int, 'dim'),
        '<BlockDim>': (_read_int, 'block_dim'),
        '<ValueAvg>': (_read_vector_float, 'value_avg'),
        '<DerivAvg>': (_read_vector_float, 'deriv_avg'),
        '<OderivRms>': (_read_vector_float, 'oderiv_rms'),
        '<Count>': (_read_float, 'count'),
        '<OderivCount>': (_read_float, 'oderiv_count')
    }
    return actions


class LogSoftmaxComponent(NonlinearComponent):
  """LogSoftmaxComponent."""


class RectifiedLinearComponent(NonlinearComponent):
  """RectifiedLinearComponent."""


class TdnnComponent(Component):
  """TdnnComponent."""

  def _actions(self) -> Dict:
    """See baseclass document."""
    actions = {
        '<TimeOffsets>': (_read_vector_int, 'time_offsets'),
        '<LinearParams>': (_read_matrix_trans, 'params'),
        '<BiasParams>': (_read_vector_float, 'bias'),
        '<OrthonormalConstraint>': (_read_float, 'orthonormal_constraint'),
        '<UseNaturalGradient>': (_read_bool, 'use_natrual_gradient'),
        '<RankInOut>': (_read_int, 'rank_inout'),
        '<NumSamplesHistory>': (_read_float, 'num_samples_history'),
        '<Alpha>': (_read_float, 'alpha'),
        '<AlphaInOut>': (_read_float, 'alpha_inout'),
    }
    return actions


@unique
class Components(Enum):
  """Kaldi nnet3 Components."""

  AffineComponent = AffineComponent
  BatchNormComponent = BatchNormComponent
  FixedAffineComponent = FixedAffineComponent
  GeneralDropoutComponent = GeneralDropoutComponent
  LinearComponent = LinearComponent
  LogSoftmaxComponent = LogSoftmaxComponent
  NaturalGradientAffineComponent = NaturalGradientAffineComponent
  NonlinearComponent = NonlinearComponent
  NoOpComponent = NoOpComponent
  RectifiedLinearComponent = RectifiedLinearComponent
  TdnnComponent = TdnnComponent


def read_next_token(line: str, pos: int) -> Tuple[Optional[str], int]:
  """Read next token from line.

  Args:
    line: line.
    pos: current position.

  Returns:
    Token (None if not found) and current position.
  """
  assert isinstance(line, str) and isinstance(pos, int)
  assert pos >= 0

  while pos < len(line) and line[pos].isspace():
    pos += 1

  if pos >= len(line):
    return None, pos

  initial_pos = pos
  while pos < len(line) and not line[pos].isspace():
    pos += 1
  return line[initial_pos:pos], pos


def read_component_type(line: str, pos: int) -> Tuple[str, int]:
  """Read component type from line.

  Args:
    line: line.
    pos: current position.

  Returns:
    component type and current position.
  """
  component_type, pos = read_next_token(line, pos)
  if (isinstance(component_type, str) and len(component_type) >= 13 and
      component_type[0] == '<' and component_type[-10:] == 'Component>'):
    return component_type, pos
  else:
    raise ValueError(f'Error reading Component at position {pos}, '
                     f'expected <xxxComponent>, got: {component_type}.')


# pylint: disable = unused-argument
def _read_bool(line: str, pos: int, line_buffer: TextIO) -> Tuple[bool, int]:
  """Read bool value from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    bool value and current position.
  """
  tok, pos = read_next_token(line, pos)
  if tok in ['F', 'False', 'false']:
    return False, pos
  elif tok in ['T', 'True', 'true']:
    return True, pos
  else:
    raise ValueError(f'Error at position {pos}, expected bool but got {tok}.')


# pylint: disable = unused-argument
def _read_int(line: str, pos: int, line_buffer: TextIO) -> Tuple[int, int]:
  """Read int value from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    int value and current position.
  """
  tok, pos = read_next_token(line, pos)
  return int(tok), pos


# pylint: disable = unused-argument
def _read_float(line: str, pos: int, line_buffer: TextIO) -> Tuple[float, int]:
  """Read float value from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    float value and current position.
  """
  tok, pos = read_next_token(line, pos)
  return float(tok), pos


def __read_vector(line: str, pos: int,
                  line_buffer: TextIO) -> Tuple[np.array, int]:
  """Read vector from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    vector and current position.
  """
  tok, pos = read_next_token(line, pos)
  if tok != '[':
    raise ValueError(f'Error at line position {pos}, expected [ but got {tok}.')

  vector = []
  while True:
    tok, pos = read_next_token(line, pos)
    if tok == ']':
      break
    if tok is None:
      line = next(line_buffer)
      if line is None:
        raise ValueError('Encountered EOF while reading vector.')

      pos = 0
      continue

    vector.append(tok)

  if tok is None:
    raise ValueError('Encountered EOF while reading vector.')
  return vector, pos


def _read_vector_int(line: str, pos: int,
                     line_buffer: TextIO) -> Tuple[np.array, int]:
  """Read int vector from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    float int and current position.
  """
  vector, pos = __read_vector(line, pos, line_buffer)
  return np.array([int(v) for v in vector], dtype=np.int), pos


def _read_vector_float(line: str, pos: int,
                       line_buffer: TextIO) -> Tuple[np.array, int]:
  """Read float vector from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    float vector and current position.
  """
  vector, pos = __read_vector(line, pos, line_buffer)
  return np.array([float(v) for v in vector], dtype=np.float32), pos


def __check_for_newline(line: str, pos: int) -> Tuple[bool, int]:
  """Check if line is newline.

  Args:
    line: line.
    pos: current position.

  Returns:
    bool and current position.
  """
  assert isinstance(line, str) and isinstance(pos, int)
  assert pos >= 0

  saw_newline = False
  while pos < len(line) and line[pos].isspace():
    if line[pos] == '\n':
      saw_newline = True
    pos += 1
  return saw_newline, pos


def _read_matrix_trans(line: str, pos: int,
                       line_buffer: TextIO) -> Tuple[np.array, int]:
  """Read matrix transpose from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    matrix transpose and current position.
  """
  tok, pos = read_next_token(line, pos)
  if tok != '[':
    raise ValueError(f'Error at line position {pos}, expected [ but got {tok}.')

  mat = []
  while True:
    one_row = []
    while True:
      tok, pos = read_next_token(line, pos)
      if tok == '[':
        tok, pos = read_next_token(line, pos)

      if tok == ']' or tok is None:
        break

      one_row.append(float(tok))

      saw_newline, pos = __check_for_newline(line, pos)
      if saw_newline:  # Newline terminates each row of the matrix.
        break

    if len(one_row) > 0:
      mat.append(one_row)
    if tok == ']':
      break
    if tok is None:
      line = next(line_buffer)
      if line is None:
        raise ValueError('Encountered EOF while reading matrix.')
      pos = 0

  return np.transpose(np.array(mat, dtype=np.float32)), pos
