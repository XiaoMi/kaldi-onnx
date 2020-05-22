#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/21
"""Parse nnet3 model."""
import re
from typing import Dict, List, Optional, TextIO, Tuple

from converter.common import *
from converter.utils import *


class Parser:
  """Kaldi nnet3 model parser."""

  def __init__(self, line_buffer: TextIO):
    """Initialize.

    Args:
      line_buffer: line buffer for kaldi's text mdl file.
    """
    self.__affine_actions = {
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

    self.__batchnorm_actions = {
        '<Dim>': (read_int, 'dim'),
        '<BlockDim>': (read_int, 'block_dim'),
        '<Epsilon>': (read_float, 'epsilon'),
        '<TargetRms>': (read_float, 'target_rms'),
        '<Count>': (read_float, 'count'),
        '<StatsMean>': (read_vector, 'stats_mean'),
        '<StatsVar>': (read_vector, 'stats_var'),
        '<TestMode>': (read_bool, 'test_mode'),
    }

    self.__basic_actions = {
        '<Dim>': (read_int, 'dim'),
        '<InputDim>': (read_int, 'input_dim'),
        '<OutputDim>': (read_int, 'output_dim')
    }

    self.__linear_actions = {
        '<Params>': (read_matrix, 'params'),
        '<RankInOut>': (read_int, 'rank_inout'),
        '<UpdatedPeriod>': (read_int, 'updated_period'),
        '<NumSamplesHistory>': (read_float, 'num_samples_history'),
        '<Alpha>': (read_float, 'alpha')
    }

    self.__nonlinear_actions = {
        '<Dim>': (read_int, 'dim'),
        '<BlockDim>': (read_int, 'block_dim'),
        '<ValueAvg>': (read_vector, 'value_avg'),
        '<DerivAvg>': (read_vector, 'deriv_avg'),
        '<OderivRms>': (read_vector, 'oderiv_rms'),
        '<Count>': (read_float, 'count'),
        '<OderivCount>': (read_float, 'oderiv_count')
    }

    self.__permute_actions = {
        '<ColumnMap>': (read_vector, 'column_map'),
    }

    self.__tdnn_actions = {
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

    self.__component_parsers = {
        Component.AffineComponent.name: self.__affine_actions,
        Component.BatchNormComponent.name: self.__batchnorm_actions,
        Component.FixedAffineComponent.name: self.__affine_actions,
        Component.GeneralDropoutComponent.name: self.__basic_actions,
        Component.LinearComponent.name: self.__linear_actions,
        Component.LogSoftmaxComponent.name: self.__nonlinear_actions,
        Component.NaturalGradientAffineComponent.name: self.__affine_actions,
        Component.NonlinearComponent.name: self.__nonlinear_actions,
        Component.NoOpComponent.name: self.__basic_actions,
        Component.PermuteComponent.name: self.__permute_actions,
        Component.RectifiedLinearComponent.name: self.__nonlinear_actions,
        Component.TdnnComponent.name: self.__tdnn_actions,
    }

    self.__name_to_component = dict()
    self.__num_components = 0
    self.__line_buffer = line_buffer
    self.__pos = 0
    self.__current_id = 0

  def run(self) -> List:
    """Start parse nnet3 model file."""
    self.__check_header()
    self.__parse_nnet3_configs()
    self.__parse_component_lines()
    return list(self.__name_to_component.values())

  def __check_header(self):
    """Check nnet3 file header."""
    line = next(self.__line_buffer)
    if not line.startswith('<Nnet3>'):
      raise ValueError('Parse error: <Nnet3> header not found.')

  def __parse_nnet3_configs(self):
    """Parse all nnet3 config."""
    while True:
      line = next(self.__line_buffer, 'Parser_EOF')
      if line == 'Parser_EOF':
        raise ValueError('No <NumComponents> in File.')

      if line.startswith('<NumComponents>'):
        self.__num_components = int(line.split()[1])
        break

      component = self.__parse_one_line(line)
      if component is not None:
        if 'input' in component:
          component['input'] = self.__parse_component_input(component['input'])

        if component['node_type'] in {'input-node', 'output-node'}:
          msg = f'"name" attribute is required: {component}'
          kaldi_check('name' in component, msg)

          component['type'] = KaldiOpRawType[component['node_type']]
          if 'input-node' in component:
            component['input'] = [component['input-node']]

        self.__current_id += 1
        component['id'] = self.__current_id
        self.__add_component(component)

  @staticmethod
  def __parse_one_line(line: str) -> Optional[Dict]:
    """Parse config from one line content of nnet3 file.

    Args:
      line: one line content of nnet3 file.

    Returns:
      Parsed component dict.
    """
    pattern = '^input-node|^output-node|^component|^component-node|'
    if re.search(pattern, line.strip()) is None:
      return None

    items = []
    split_contents = line.split()
    for content in split_contents[1:]:
      if '=' in content:
        items.append(content)
      else:
        items[-1] += f' {content}'

    component = {'node_type': split_contents[0]}
    for item in items:
      component_key, component_value = item.split('=')
      component[component_key] = component_value
    return component

  def __add_component(self, component: Dict):
    """Add one component.

    Args:
      component: component dict.
    """
    cond = 'name' in component or 'component' in component
    msg = f'"name" or "component" attribute is required: {component}.'
    kaldi_check(cond, msg)

    has_component = 'component' in component
    name = component['component'] if has_component else component['name']
    self.__name_to_component[name] = component

  def __parse_component_input(self, input_str) -> List[str]:
    """Parse input of one component.
    
    Args:
      input_str: input string.

    Returns:
      Input names list of one component.
    """
    input_str = input_str.replace(' ', '')
    sub_type = self.__parse_sub_type(input_str)

    if sub_type is not None:
      sub_components = []
      input_name = self.__parse_descriptor(sub_type, input_str, sub_components)
      for component in sub_components:
        self.__add_component(component)
    else:
      input_name = input_str
    return input_name if isinstance(input_name, list) else [input_name]

  @staticmethod
  def __parse_sub_type(input_str: str) -> Optional[str]:
    """Parse input string to get sub component type.

    For example, input Append(Offset(input, -1), input), sub type is Append.

    Args:
      input_str: input string.

    Returns:
      Sub component type, can be None if no sub component type is found.
    """
    if '(' in input_str:
      bracket_index = input_str.index('(')
      if bracket_index > 0:
        return input_str[0: bracket_index]
      else:
        return None

  def __parse_descriptor(self, sub_type, input_str, sub_components) -> str:
    """Parse kaldi descriptor.

    Args:
      sub_type: sub type name.
      input_str: input string.
      sub_components: sub component list,
                      may change if new sub component is parsed.
    
    Returns:
      Component name.
    """
    sub_str = input_str[len(sub_type) + 1: -1]
    if sub_type == Descriptor.Append.name:
      return self.__parse_append_descriptor(sub_str, sub_components)
    if sub_type == Descriptor.Offset.name:
      return self.__parse_offset_descriptor(sub_str, sub_components)
    elif sub_type == Descriptor.ReplaceIndex.name:
      return self.__parse_replace_index_descriptor(sub_str, sub_components)
    elif sub_type == Descriptor.Scale.name:
      return self.__parse_scale_descriptor(sub_str, sub_components)
    elif sub_type == Descriptor.Sum.name:
      return self.__parse_sum_descriptor(sub_str, sub_components)
    else:
      raise NotImplementedError(f'Does not support this descriptor type: '
                                f'{sub_type} in input: {input_str}')

  @staticmethod
  def __is_descriptor(node_type) -> bool:
    """If the node belongs to nnet3 descriptor.

    Args:
      node_type: type of node.

    Returns:
      Descriptor or not.
    """
    for descriptor_name in Descriptor.value:
      if node_type == descriptor_name:
        return True
    return False

  def __parse_append_descriptor(self, input_str: str,
                                components: List[Optional[Dict]]) -> str:
    """Parse kaldi Append descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = parenthesis_split(input_str, ",")
    num_inputs = len(items)
    kaldi_check(num_inputs >= 2, "Append should have at least two inputs.")

    append_inputs = []
    offset_components = []
    offset_inputs = []
    offset_indexes = []
    offsets = []
    for item in items:
      sub_type = self.__parse_sub_type(item)
      if self.__is_descriptor(sub_type):
        sub_comp_name = self.__parse_descriptor(sub_type, item, components)
        sub_comp = components[-1]
        append_inputs.append(sub_comp_name)

        if sub_type == Descriptor.Offset.name:
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
      self.__current_id += 1
      component_name = 'splice_' + str(self.__current_id)
      component = {
          'id': self.__current_id,
          'type': 'Splice',
          'name': component_name,
          'input': pure_inputs,
          'context': offsets}

      for item in offset_components:
        components.remove(item)
      components.append(component)
    else:
      splice_indexes = splice_continous_numbers(offset_indexes)
      if (len(pure_inputs) == 1 and len(splice_indexes) == 1 and
          len(offset_inputs) > 1):
        self.__current_id += 1
        splice_comp_name = 'splice_' + str(self.__current_id)
        splice_component = {
            'id': self.__current_id,
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
          components.remove(item)
        components.append(splice_component)

      self.__current_id += 1
      component_name = 'append_' + str(self.__current_id)
      component = {
          'id': self.__current_id,
          'type': 'Append',
          'name': component_name,
          'input': append_inputs}
      components.append(component)
    return component_name

  def __parse_offset_descriptor(self, input_str: str,
                                components: List[Optional[Dict]]) -> str:
    """Parse kaldi Offset descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = parenthesis_split(input_str, ",")
    kaldi_check(len(items) == 2, 'Offset descriptor should have 2 items.')

    sub_type = self.__parse_sub_type(items[0])
    if sub_type is not None:
      input_name = self.__parse_descriptor(sub_type, items[0], components)
    else:
      input_name = items[0]

    offset = int(items[1])
    self.__current_id += 1
    component_name = input_name + '.Offset.' + str(offset)
    component = {
        'id': self.__current_id,
        'type': 'Offset',
        'name': component_name,
        'input': [input_name],
        'offset': offset
    }
    components.append(component)
    return component_name

  def __parse_replace_index_descriptor(self, input_str: str,
                                       components: List[Optional[Dict]]) -> str:
    """Parse kaldi ReplaceIndex descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = parenthesis_split(input_str, ",")
    kaldi_check(len(items) == 3, 'ReplaceIndex descriptor should have 3 items.')

    sub_type = self.__parse_sub_type(items[0])
    if sub_type is not None:
      input_name = self.__parse_descriptor(sub_type, items[0], components)
    else:
      input_name = items[0]

    component_name = input_name + '.ReplaceIndex.' + items[1] + items[2]
    self.__current_id += 1
    component = {
        'id': self.__current_id,
        'type': 'ReplaceIndex',
        'name': component_name,
        'input': [input_name],
        'var_name': items[1],
        'modulus': int(items[2])}
    components.append(component)
    return component_name

  def __parse_scale_descriptor(self, input_str: str,
                               components: List[Optional[Dict]]) -> str:
    """Parse kaldi Scale descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = parenthesis_split(input_str, ",")
    kaldi_check(len(items) == 2, 'Scale descriptor should have 2 items.')

    scale = float(items[0])
    sub_type = self.__parse_sub_type(items[1])
    if sub_type is not None:
      input_name = self.__parse_descriptor(sub_type, items[1], components)
    else:
      input_name = items[1]

    component_name = input_name + '.Scale.' + items[0]
    self.__current_id += 1
    component = {
        'id': self.__current_id,
        'type': 'Scale',
        'name': component_name,
        'input': [input_name],
        'scale': scale
    }
    components.append(component)
    return component_name

  def __parse_sum_descriptor(self, input_str: str,
                             components: List[Optional[Dict]]) -> str:
    """Parse kaldi Sum descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      component name.
    """
    items = parenthesis_split(input_str, ",")
    kaldi_check(len(items) == 2, 'Sum descriptor should have 2 items.')

    input_names = list()
    for item in items:
      sub_type = self.__parse_sub_type(item)
      if sub_type is not None:
        input_name = self.__parse_descriptor(sub_type, item, components)
      else:
        input_name = item
      input_names.append(input_name)

    component_name = '.Sum.'.join(input_names)
    self.__current_id += 1
    component = {
        'id': self.__current_id,
        'type': 'Sum',
        'name': component_name,
        'input': input_names,
    }
    components.append(component)
    return component_name

  def __parse_component_lines(self):
    """Parse all components lines."""
    num = 0
    while True:
      line = next(self.__line_buffer)
      pos = 0
      tok, pos = read_next_token(line, pos)

      if tok is None:
        line = next(self.__line_buffer)
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
        component_dict, line, pos = self.read_component(line, pos,
                                                        component_type)
        if component_dict is not None:
          if component_name in self.__name_to_component:
            config_dict = self.__name_to_component[component_name]
            new_dict = merge_two_dicts(config_dict, component_dict)
            self.__name_to_component[component_name] = new_dict
            num += 1
        else:
          raise ValueError(f"Error reading component with name {component_name}"
                           f" at position {component_pos}.")
      elif tok == '</Nnet3>':
        assert num == self.__num_components
        logging.info(f"Finished parsing nnet3 {num} components.")
        break
      else:
        raise ValueError(f"Error reading component at position {pos}, "
                         f"expected <ComponentName>, got: {tok}.")

  def read_component(self, line, pos, component_type) -> Tuple[Dict, str, int]:
    """Read component.

    Args:
      line: line.
      pos: position.
      component_type: type of component.

    Returns:
      Component, line and position after reading component.
    """
    terminating_token = "</" + component_type[1:]
    terminating_tokens = {terminating_token, '<ComponentName>'}

    component_type = component_type[1:-1]
    if component_type in self.__component_parsers:
      action_dict = self.__component_parsers[component_type]
      component, pos = read_generic(line, pos, self.__line_buffer,
                                    terminating_tokens, action_dict)

      if component is not None:
        component['type'] = KaldiOpRawType[component_type]
        component['raw-type'] = component_type[1:-10]
      return component, line, pos
    else:
      raise NotImplementedError(f"Component: {component_type} not supported.")
