#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/21
"""Parse nnet3 model."""
import re

from converter.common import *
from converter.utils import *


class Nnet3Parser:
  """Kaldi nnet3 model parser."""

  def __init__(self, line_buffer):
    """Initialize.

    Args:
      line_buffer: line buffer for kaldi's text mdl file.
    """
    self._affine_actions = {
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

    self._bachnorm_actions = {
        '<Dim>': (read_int, 'dim'),
        '<BlockDim>': (read_int, 'block_dim'),
        '<Epsilon>': (read_float, 'epsilon'),
        '<TargetRms>': (read_float, 'target_rms'),
        '<Count>': (read_float, 'count'),
        '<StatsMean>': (read_vector, 'stats_mean'),
        '<StatsVar>': (read_vector, 'stats_var'),
        '<TestMode>': (read_bool, 'test_mode'),
    }

    self._basic_actions = {
        '<Dim>': (read_int, 'dim'),
        '<InputDim>': (read_int, 'input_dim'),
        '<OutputDim>': (read_int, 'output_dim')
    }

    self._linear_actions = {
        '<Params>': (read_matrix, 'params'),
        '<RankInOut>': (read_int, 'rank_inout'),
        '<UpdatedPeriod>': (read_int, 'updated_period'),
        '<NumSamplesHistory>': (read_float, 'num_samples_history'),
        '<Alpha>': (read_float, 'alpha')
    }

    self._nonlinear_actions = {
        '<Dim>': (read_int, 'dim'),
        '<BlockDim>': (read_int, 'block_dim'),
        '<ValueAvg>': (read_vector, 'value_avg'),
        '<DerivAvg>': (read_vector, 'deriv_avg'),
        '<OderivRms>': (read_vector, 'oderiv_rms'),
        '<Count>': (read_float, 'count'),
        '<OderivCount>': (read_float, 'oderiv_count')
    }

    self._permute_actions = {
        '<ColumnMap>': (read_vector, 'column_map'),
    }

    self._tdnn_actions = {
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

    self._component_parsers = {
        Component.AffineComponent.name: self._affine_actions,
        Component.BatchNormComponent.name: self._bachnorm_actions,
        Component.BlockAffineComponent.name: self._affine_actions,
        Component.BlockAffineComponentPreconditioned.name:
            self._affine_actions,
        Component.ClipGradientComponent.name: self._basic_actions,
        Component.DistributeComponent.name: self._basic_actions,
        Component.DropoutComponent.name: self._basic_actions,
        Component.DropoutMaskComponent.name: self._basic_actions,
        Component.ElementwiseProductComponent.name:
            self._basic_actions,
        Component.FixedAffineComponent.name: self._affine_actions,
        Component.GeneralDropoutComponent.name: self._basic_actions,
        Component.LinearComponent.name: self._linear_actions,
        Component.LogSoftmaxComponent.name: self._nonlinear_actions,
        Component.NaturalGradientAffineComponent.name:
            self._affine_actions,
        Component.NaturalGradientRepeatedAffineComponent.name:
            self._affine_actions,
        Component.NonlinearComponent.name: self._nonlinear_actions,
        Component.NoOpComponent.name: self._basic_actions,
        Component.PermuteComponent.name: self._permute_actions,
        Component.PnormComponent.name: self._basic_actions,
        Component.RandomComponent.name: self._basic_actions,
        Component.RectifiedLinearComponent.name:
            self._nonlinear_actions,
        Component.RepeatedAffineComponent.name:
            self._affine_actions,
        Component.SigmoidComponent.name:
            self._nonlinear_actions,
        Component.SoftmaxComponent.name:
            self._nonlinear_actions,
        Component.TanhComponent.name: self._nonlinear_actions,
        Component.TdnnComponent.name: self._tdnn_actions,
    }

    # self._configs = []
    self._components_by_name = dict()
    self._component_names = []
    self._components = []
    self._num_components = 0
    self._line_buffer = line_buffer
    self._pos = 0
    self._current_id = 0
    self._transition_model = []

  def run(self):
    """Start parse."""
    self.check_header()
    self.__parse_nnet3_configs()
    self.parse_component_lines()
    self._components = []
    for component_name in self._components_by_name:
      self._components.append(
          self._components_by_name[component_name])
    return self._components, self._transition_model

  def check_header(self):
    """Check nnet3 file header."""
    line = next(self._line_buffer)
    if not line.startswith('<Nnet3>'):
      raise ValueError('Parse error: <Nnet3> header not found.')

  def __parse_nnet3_configs(self):
    """Parse all nnet3 config."""
    while True:
      line = next(self._line_buffer, 'Parser_EOF')
      if line == 'Parser_EOF':
        raise Exception('No <NumComponents> in File.')
      if line.startswith('<NumComponents>'):
        self._num_components = int(line.split()[1])
        break

      config_type, parsed_config = self.__parse_nnet3_config(line)
      if config_type is not None:
          parsed_config['node_type'] = config_type

          if 'input' in parsed_config:
              input = parsed_config['input']
              parsed_input = self.parse_input_descriptor(input)
              if isinstance(parsed_input, list):
                  parsed_config['input'] = parsed_input
              else:
                  parsed_config['input'] = [parsed_input]

          if config_type in ['output-node',
                             'input-node',
                             'dim-range-node']:
              parsed_config['type'] = KaldiOpRawType[config_type]
              kaldi_check('name' in parsed_config,
                          "Expect 'name' value in %s" % parsed_config)
              if 'input-node' in parsed_config:
                  parsed_config['input'] = [parsed_config['input-node']]
              if 'dim-offset' in parsed_config:
                  parsed_config['offset'] = parsed_config['dim-offset']
          self._current_id += 1
          parsed_config['id'] = self._current_id
          self.add_component(parsed_config)

  def add_component(self, component):
      kaldi_check('name' in component or 'component' in component,
                  "'name' or 'component' is required.")
      if 'component' in component:
          component_name = component['component']
      else:
          component_name = component['name']
      self._components.append(component)
      self._components_by_name[component_name] = component
      self._component_names.append(component['name'])

  def parse_input_descriptor(self, input_str):
      sub_components = []
      input_str = input_str.replace(' ', '')
      type = self.check_sub_inputs(input_str)
      if type is not None:
          input_name = self.parse_descriptor(type, input_str, sub_components)
          for item in sub_components:
              self.add_component(item)
      else:
          input_name = input_str
      return input_name

  @staticmethod
  def check_sub_inputs(input_str):
      try:
          type_end_index = input_str.index('(')
          if type_end_index > 0:
              type = input_str[0: type_end_index]
              return type
          else:
              return None
      except ValueError:
          return None

  def parse_descriptor(self, type, input_str, sub_components):
      input = input_str[len(type) + 1: -1]
      if type == Descriptor.Offset.name:
          return self.parse_offset_descp(input, sub_components)
      elif type == Descriptor.Sum.name:
          return self.parse_sum_descp(input, sub_components)
      elif type == Descriptor.Scale.name:
          return self.parse_scale_descp(input, sub_components)
      elif type == Descriptor.Const.name:
          return self.parse_const_descp(input, sub_components)
      elif type == Descriptor.ReplaceIndex.name:
          return self.parse_replace_index_descp(input, sub_components)
      elif type == Descriptor.Append.name:
          return self.parse_append_descp(input, sub_components)
      else:
          raise Exception(
              'Does not support this descriptor type: {0} in input: {1}'
              .format(type, input_str))

  def parse_append_descp(self, input_str, sub_components):
      input_str = input_str.replace(" ", "")
      items = parenthesis_split(input_str, ",")
      num_inputs = len(items)
      kaldi_check(num_inputs >= 2,
                  "Append should have at least two inputs.")
      append_inputs = []
      offset_components = []
      offset_inputs = []
      offset_indexes = []
      offsets = []
      for item in items:
          type = self.check_sub_inputs(item)
          if type in Descriptors:
              sub_comp_name = self.parse_descriptor(
                  type, item, sub_components)
              sub_comp = sub_components[-1]
              append_inputs.append(sub_comp_name)
              if type == Descriptor.Offset.name:
                  offset_components.append(sub_comp)
                  offset_in = sub_comp['input']
                  offsets.append(sub_comp['offset'])
                  offset_inputs.extend(offset_in)
                  offset_indexes.append(items.index(item))
          else:
              offsets.append(0)
              offset_inputs.append(item)
              offset_indexes.append(items.index(item))
              append_inputs.append(item)

      # check if fusing to splice needed
      pure_inputs = list(set(offset_inputs))

      if num_inputs == len(offset_inputs) and len(pure_inputs) == 1:
          self._current_id += 1
          comp_name = 'splice_' + str(self._current_id)
          component = {
              'id': self._current_id,
              'type': 'Splice',
              'name': comp_name,
              'input': pure_inputs,
              'context': offsets}
          for item in offset_components:
              sub_components.remove(item)
          sub_components.append(component)
      else:
          splice_indexes = splice_continous_numbers(offset_indexes)
          if len(pure_inputs) == 1 and len(splice_indexes) == 1 and\
                  len(offset_inputs) > 1:
              self._current_id += 1
              splice_comp_name = 'splice_' + str(self._current_id)
              splice_component = {
                  'id': self._current_id,
                  'type': 'Splice',
                  'name': splice_comp_name,
                  'context': offsets,
                  'input': pure_inputs}
              new_append_inputs = []
              for i in range(num_inputs):
                  if i not in offset_indexes:
                      new_append_inputs.append(append_inputs[i])
                  elif i == offset_indexes[0]:
                      new_append_inputs.append(splice_comp_name)
              append_inputs = new_append_inputs
              for item in offset_components:
                  sub_components.remove(item)
              sub_components.append(splice_component)
          self._current_id += 1
          comp_name = 'append_' + str(self._current_id)
          component = {
              'id': self._current_id,
              'type': 'Append',
              'name': comp_name,
              'input': append_inputs}
          sub_components.append(component)
      return comp_name

  def parse_offset_descp(self, input, sub_components):
      items = parenthesis_split(input, ",")
      kaldi_check(len(items) == 2, 'Offset descriptor should have 2 items.')
      sub_type = self.check_sub_inputs(items[0])
      if sub_type is not None:
          input_name = self.parse_descriptor(sub_type,
                                             items[0],
                                             sub_components)
      else:
          input_name = items[0]
      offset = int(items[1])
      self._current_id += 1
      comp_name = input_name + '.Offset.' + str(offset)
      component = {
          'id': self._current_id,
          'type': 'Offset',
          'name': comp_name,
          'input': [input_name],
          'offset': offset
      }
      sub_components.append(component)
      return comp_name

  def parse_scale_descp(self, input, sub_components):
      items = parenthesis_split(input, ",")
      kaldi_check(len(items) == 2, 'Scale descriptor should have 2 items.')
      scale = float(items[0])
      sub_type = self.check_sub_inputs(items[1])
      if sub_type is not None:
          input_name = self.parse_descriptor(sub_type,
                                             items[1],
                                             sub_components)
      else:
          input_name = items[1]
      comp_name = input_name + '.Scale.' + items[0]
      self._current_id += 1
      component = {
          'id': self._current_id,
          'type': 'Scale',
          'name': comp_name,
          'input': [input_name],
          'scale': scale
      }
      sub_components.append(component)
      return comp_name

  def parse_const_descp(self, input, sub_components):
      items = parenthesis_split(input, ",")
      kaldi_check(len(items) == 2, 'Const descriptor should have 2 items.')
      value = float(items[0])
      dimension = int(items[1])
      comp_name = 'Const' + '_' + items[0] + '_' + items[1]
      self._current_id += 1
      component = {
          'id': self._current_id,
          'type': 'Const',
          'name': comp_name,
          'input': [],
          'value': value,
          'dim': dimension,
      }
      sub_components.append(component)
      return comp_name

  def parse_sum_descp(self, input, sub_components):
      items = parenthesis_split(input, ",")
      kaldi_check(len(items) == 2, 'Sum descriptor should have 2 items.')
      sub_type = self.check_sub_inputs(items[0])
      if sub_type is not None:
          input_name = self.parse_descriptor(sub_type,
                                             items[0],
                                             sub_components)
      else:
          input_name = items[0]
      sub_type = self.check_sub_inputs(items[1])
      if sub_type is not None:
          other_name = self.parse_descriptor(sub_type,
                                             items[1],
                                             sub_components)
      else:
          other_name = items[1]

      comp_name = input_name + '.Sum.' + other_name
      self._current_id += 1
      component = {
          'id': self._current_id,
          'type': 'Sum',
          'name': comp_name,
          'input': [input_name, other_name],
      }
      sub_components.append(component)
      return comp_name

  def parse_replace_index_descp(self, input, sub_components):
      items = parenthesis_split(input, ",")
      kaldi_check(len(items) == 3,
                  'ReplaceIndex descriptor should have 3 items.')
      sub_type = self.check_sub_inputs(items[0])
      if sub_type is not None:
          input_name = self.parse_descriptor(sub_type,
                                             items[0],
                                             sub_components)
      else:
          input_name = items[0]
      var_name = items[1]
      value = int(items[2])
      inputs = [input_name]
      comp_name = input_name + '.ReplaceIndex.' + items[1] + items[2]
      self._current_id += 1
      component = {
          'id': self._current_id,
          'type': 'ReplaceIndex',
          'name': comp_name,
          'input': inputs,
          'var_name': var_name,
          'modulus': value}
      sub_components.append(component)
      return comp_name

  def parse_component_lines(self):
      """Parse all components lines before </Nnet3>"""
      num = 0
      while True:
          line = next(self._line_buffer)
          pos = 0
          tok, pos = read_next_token(line, pos)
          if tok is None:
              line = next(self._line_buffer)
              pos = 0
              if line is None:
                  logging.error("unexpected EOF on line:\n {}".format(line))
                  break
              else:
                  tok, pos = read_next_token(line, pos)
          if tok == '<ComponentName>':
              component_pos = pos
              component_name, pos = read_next_token(line, pos)

              component_type, pos = read_component_type(line, pos)
              assert is_component_type(component_type)
              component_dict, line, pos = self.read_component(line,
                                                              pos,
                                                              component_type)
              if component_dict is not None:
                  if component_name in self._components_by_name:
                      config_dict = self._components_by_name[component_name]
                      new_dict = merge_two_dicts(config_dict, component_dict)
                      self._components_by_name[component_name] = new_dict
                      num += 1
              else:
                  logging.error("{0}: error reading component with name {1}"
                             " at position {2}"
                             .format(sys.argv[0],
                                     component_name,
                                     component_pos))
          elif tok == '</Nnet3>':
              logging.info("finished parsing nnet3 (%s) components." % num)
              assert num == self._num_components
              break
          else:
              logging.error("{0}: error reading Component:"
                         " at position {1}, expected <ComponentName>,"
                         " got: {2}"
                         .format(sys.argv[0], pos, tok))
              break

  def read_component(self, line, pos, component_type):
      terminating_token = "</" + component_type[1:]
      terminating_tokens = {terminating_token, '<ComponentName>'}

      type = component_type[1:-1]
      if type in self._component_parsers:
          action_dict = self._component_parsers[type]
          d, pos = self.read_generic(line,
                                     pos,
                                     self._line_buffer,
                                     terminating_tokens,
                                     action_dict)
          if d is not None:
              d['type'] = KaldiOpRawType[type]
              d['raw-type'] = component_type[1:-10]  # e.g. 'Linear'
          return d, line, pos
      else:
          logging.info("Component: %s not supported yet." % type)
          return None, line, pos

  @staticmethod
  def read_generic(line, pos, line_buffer, terminating_token, action_dict):
      if isinstance(terminating_token, str):
          terminating_tokens = set([terminating_token])
      else:
          terminating_tokens = terminating_token
          assert isinstance(terminating_tokens, set)
      assert isinstance(action_dict, dict)

      # d will contain the fields of the object.
      d = dict()
      orig_pos = pos
      while True:
          tok, pos = read_next_token(line, pos)
          if tok in terminating_tokens:
              break
          if tok is None:
              line = next(line_buffer)
              if line is None:
                  logging.error(
                      "{0}: error reading object starting at position {1},"
                      " got EOF while expecting one of: {2}".format(
                          sys.argv[0], orig_pos, terminating_tokens))
                  break
              else:
                  pos = 0
                  continue
          if tok in action_dict:
              func, name = action_dict[tok]
              obj, pos = func(line, pos, line_buffer)
              d[name] = obj
      return d, pos

  @staticmethod
  def __parse_nnet3_config(line):
    """Parse config from one line content of nnet3 file.

    Args:
      line: one line content of nnet3 file.

    Returns:
      type and content.
    """
    if re.search('^input-node|^component|^output-node|^component-node|'
                 '^dim-range-node', line.strip()) is None:
      return [None, None]

    parts = line.split()
    config_type = parts[0]
    fields = []
    prev_field = ''
    for i in range(1, len(parts)):
        if re.search('=', parts[i]) is None:
            prev_field += ' ' + parts[i]
        else:
            if not (prev_field.strip() == ''):
                fields.append(prev_field)
            sub_parts = parts[i].split('=')
            if len(sub_parts) != 2:
                raise Exception('Malformed config line {0}'.format(str))
            fields.append(sub_parts[0])
            prev_field = sub_parts[1]
    fields.append(prev_field)

    parsed_string = {}
    try:
        while len(fields) > 0:
            value = re.sub(',$', '', fields.pop().strip())
            key = fields.pop()
            parsed_string[key.strip()] = value.strip()
    except IndexError:
        raise Exception('Malformed config line {0}'.format(str))
    return [config_type, parsed_string]
