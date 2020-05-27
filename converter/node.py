#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/27
"""Nnet3 node."""
from typing import Dict, List

from converter.common import KaldiOpType
from converter.utils import kaldi_check


class Node:
  """Node.

  """

  def __init__(self, name, n_type, inputs, outputs, attrs=None, consts=None):
    self.name = name
    self.type = n_type
    self.nexts = []
    self.__inputs = inputs
    self.__outputs = outputs
    self.__input_dim = 0
    self.__output_dim = 0
    self.__dependencies = []
    self.__input_indexes = []
    self.__output_indexes = []
    self.output_shape = None

    self.input_range = [-100000, 100000]
    self.output_range = [-100000, 100000]

    if attrs is None:
      self.attrs = dict()
    else:
      self.attrs = attrs
      self.__update_attributes()

    if consts is None:
      self.consts = {}
    else:
      self.consts = consts

  def __update_attributes(self):
    """Update attributes and change type."""
    if self.type == KaldiOpType.Append.name:
      self.attrs['axis'] = -1
    elif self.type == KaldiOpType.Splice.name:
      if ('left_context' in self.attrs and 'right_context' in self.attrs and 
          'context' not in self.attrs): 
        left_context = self.attrs['left_context']
        right_context = self.attrs['right_context']
        self.attrs['context'] = list(range(-left_context, right_context + 1))

    for attr_name, attr_value in self.attrs.items():
      if attr_name in {'const_component_dim', 'mod', 'offset', 'dim', 'p'
                       'input_dim', 'output_dim', 
                       'left_context', 'right_context'}:
        self.attrs[attr_name] = int(attr_value)
      elif attr_name in ['count', 'epsilon', 'scale', 'target_rms', 
                         'variance_floor']:
        self.attrs[attr_name] = float(attr_value)
      elif attr_name in ['context'] and not isinstance(attr_value, list):
        self.attrs[attr_name] = attr_value.tolist()

  @property
  def inputs(self) -> List[str]:
    """Get input node names."""
    return self.__inputs

  @inputs.setter
  def inputs(self, inputs: List[str]):
    """Set input node names."""
    self.__inputs = [inputs]

  @property
  def outputs(self) -> List[str]:
    """Get output node names."""
    return self.__outputs

  @outputs.setter
  def outputs(self, outputs: List[str]):
    """Set output node names."""
    self.__outputs = outputs

  @property
  def dependencies(self) -> List[int]:
    """Get dependencies."""
    return self.__dependencies

  @dependencies.setter
  def dependencies(self, dependencies: List[int]):
    """Set dependencies。"""
    self.__dependencies = dependencies

  @property
  def input_indexes(self) -> List[int]:
    """Get input_indexes."""
    return self.__input_indexes

  @input_indexes.setter
  def input_indexes(self, input_indexes: List[int]):
    """Set input indexes."""
    self.__input_indexes = input_indexes

  @property
  def output_indexes(self) -> List[int]:
    """Get output indexes."""
    return self.__output_indexes

  @output_indexes.setter
  def output_indexes(self, output_indexes: List[int]):
    """Set output indexes."""
    self.__output_indexes = output_indexes

  @property
  def output_dim(self) -> int:
    """Get output dim."""
    return self.__output_dim

  @output_dim.setter
  def output_dim(self, output_dim: int):
    """Set output dim."""
    self.__output_dim = output_dim
    self.attrs['output_dim'] = output_dim

  @property
  def input_dim(self) -> int:
    """Get input dim."""
    return self.__input_dim

  @input_dim.setter
  def input_dim(self, input_dim: int):
    """Set input dim."""
    self.__input_dim = input_dim
    self.attrs['input_dim'] = input_dim

  def set_attribute(self, attr_name: str, attr_value):
    """Set attribute by name and value."""
    self.attrs[attr_name] = attr_value

  def read_attribute(self, attr_name: str):
    """Read attribute by name."""
    kaldi_check(attr_name in self.attrs, f'cannot find attribute {attr_name}.')
    return self.attrs[attr_name]

  def inference_dim(self, name_to_dim: Dict[str, int],
                    name_to_node: Dict[str, 'Node']):
    if self.name in name_to_dim:
      output_dim = name_to_dim[self.name]
      self.input_dim = output_dim
    elif 'output_dim' in self.attrs:
      output_dim = self.read_attribute('output_dim')
      self.input_dim = output_dim
    elif 'dim' in self.attrs:
      output_dim = self.read_attribute('dim')
      self.input_dim = output_dim
    elif 'input_dim' in self.attrs:
      output_dim = self.read_attribute('input_dim')
    else:
      if self.inputs[0] in name_to_dim:
        output_dim = name_to_dim[self.inputs[0]]
        self.input_dim = output_dim
      else:
        kaldi_check(self.inputs[0] in name_to_node,
                    f'Cannot find: {self.inputs[0]}.')
        input_node = name_to_node[self.inputs[0]]
        input_node.inference_dim(name_to_dim, name_to_node)
        self.input_dim = input_node.output_dim
        output_dim = self.input_dim
    self.output_dim = output_dim
    name_to_dim[self.name] = output_dim

  def is_simple(self) -> bool:
    """Simple node cannot be subsample node."""
    return True

  def inference_shape(self, batch, shapes, name_to_node):
    if self.name in shapes:
        return
    output_chunk = len(self.output_indexes)
    output_shape = [batch, output_chunk, self.output_dim]
    shapes[self.name] = output_shape
    self.output_shape = output_shape

  def pre_compute(self):
    pass

  def inference_index(self, name_to_indexes: Dict, name_to_node):
    input_name = self.inputs[0]
    if input_name in name_to_indexes:
      input_indexes = name_to_indexes[input_name]
      self.input_indexes = input_indexes
    else:
      kaldi_check(input_name in name_to_node, f'Cannot find: {input_name}')
      input_node = name_to_node[input_name]
      input_node.inference_index(name_to_indexes, name_to_node)
      input_indexes = name_to_indexes[input_name]
      self.input_indexes = input_indexes
    name_to_indexes[self.name] = self.output_indexes
    kaldi_check(set(self.dependencies) <= set(self.input_indexes),
                'input indexes is sufficient for computation')

  def inference_dependencies(self,
                             output_indexes,
                             name_to_dependencies,
                             name_to_node,
                             subsample_factor):
    kaldi_check(len(output_indexes) > 0, 'invalid output indexes values.')
    dependencies = list()
    [start, end] = self.input_range
    current_output_indexes = list()
    for index in output_indexes:
      if index in range(start, int(end + 1)):
        dependencies.append(index)
        current_output_indexes.append(index)
    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])
    dependencies = list(set(dependencies))
    dependencies.sort()
    self.dependencies = dependencies
    current_output_indexes.extend(self.output_indexes)
    current_output_indexes = list(set(current_output_indexes))
    current_output_indexes.sort()
    self.output_indexes = current_output_indexes
    name_to_dependencies[self.name] = dependencies

  def inference_range(self, name_to_range, name_to_node):
    if self.name not in name_to_range:
      input_name = self.inputs[0]
      if input_name in name_to_range:
        [start, end] = name_to_range[input_name]
      else:
        kaldi_check(input_name in name_to_node, f'Cannot find: {input_name}')
        input_node = name_to_node[input_name]
        input_node.inference_range(name_to_range, name_to_node)
        [start, end] = input_node.output_range
      name_to_range[self.name] = [start, end]
      self.input_range = [start, end]
      self.output_range = [start, end]


class GemmNode(Node):

  def inference_dim(self, name_to_dim, name_to_node):
    if 'num_repeats' in self.attrs:
      num_repeats = self.attrs['num_repeats']
    else:
      num_repeats = 1
    weights_name = self.inputs[1]
    kaldi_check(weights_name in self.consts,
                f'{weights_name} is not found in const.')
    weights_shape = self.consts[weights_name].shape
    output_dim = weights_shape[0] * num_repeats
    self.output_dim = output_dim
    name_to_dim[self.name] = output_dim


class AppendNode(Node):
  def is_simple(self):
    return False

  def inference_dim(self, name_to_dim, name_to_node):
    output_dim = 0
    for input_name in self.inputs:
      if input_name in name_to_dim:
        input_dim = name_to_dim[input_name]
      else:
        kaldi_check(input_name in name_to_node, f'Cannot find {input_name}')
        input_node = name_to_node[input_name]
        input_node.inference_dim(name_to_dim, name_to_node)
        input_dim = input_node.output_dim
      output_dim += input_dim
    self.output_dim = output_dim
    name_to_dim[self.name] = output_dim

  def inference_index(self, name_to_indexes, name_to_node):
    input_indexes = list()
    for input_name in self.inputs:
      if input_name in name_to_indexes:
        input_indexes.extend(name_to_indexes[input_name])
    input_indexes = list(set(input_indexes))
    input_indexes.sort()
    self.input_indexes = input_indexes
    name_to_indexes[self.name] = self.output_indexes
    kaldi_check(set(self.dependencies) <= set(self.input_indexes),
                'input indexes is sufficient for computation')

  def inference_range(self, name_to_range, name_to_node):
    if self.name not in name_to_range:
      [start, end] = self.input_range
      for input_name in self.inputs:
        if input_name in name_to_range:
          [input_start, input_end] = name_to_range[input_name]
        else:
          kaldi_check(input_name in name_to_node,
                      f'Cannot find: {input_name}.')
          input_node = name_to_node[input_name]
          input_node.inference_range(name_to_range, name_to_node)
          [input_start, input_end] = input_node.output_range
        start = max(start, input_start)
        end = min(end, input_end)
      name_to_range[self.name] = [start, end]
      self.input_range = [start, end]
      self.output_range = [start, end]


class IdentityNode(Node):
  def inference_index(self, name_to_indexes, name_to_node):
    input_name = self.inputs[0]
    if input_name in name_to_indexes:
      input_indexes = name_to_indexes[input_name]
    else:
      kaldi_check(input_name in name_to_node,
                  f'Cannot find: {input_name}.')
      input_node = name_to_node[input_name]
      input_node.inference_index(name_to_indexes, name_to_node)
      input_indexes = name_to_indexes[input_name]
    self.input_indexes = input_indexes
    self.output_indexes = input_indexes
    name_to_indexes[self.name] = self.output_indexes


class OffsetNode(Node):

  def is_simple(self):
    return False

  def inference_range(self, name_to_range, name_to_node):
    if self.name not in name_to_range:
      offset = self.read_attribute('offset')
      input_name = self.inputs[0]
      if input_name in name_to_range:
        [input_start, input_end] = name_to_range[input_name]
      else:
        kaldi_check(input_name in name_to_node,
                    f'Cannot find: {input_name}.')
        input_node = name_to_node[input_name]
        input_node.inference_range(name_to_range, name_to_node)
        [input_start, input_end] = input_node.output_range
      self.input_range = [input_start, input_end]
      self.output_range = [input_start - offset, input_end - offset]
      name_to_range[self.name] = self.output_range

  def pre_compute(self):
    forward_indexes = list()
    offset = self.read_attribute('offset')
    for idx in self.output_indexes:
      dep = idx + offset
      kaldi_check(dep in self.input_indexes,
                  f'Input index {dep} is required.')
      pos = self.input_indexes.index(dep)
      forward_indexes.append(pos)
    self.attrs['forward_indexes'] = forward_indexes

  def inference_dependencies(self, output_indexes, name_to_dependencies,
                             name_to_node, subsample_factor):
    kaldi_check(len(output_indexes) > 0,
                'number of output indexes should be greater than zero.')
    offset = self.read_attribute('offset')
    current_output_indexes = list()
    for i in output_indexes:
      current_output_indexes.append(i)
    dependencies = [i + offset for i in current_output_indexes]
    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])
    dependencies = list(set(dependencies))
    dependencies.sort()
    name_to_dependencies[self.name] = dependencies
    self.dependencies = dependencies
    self.output_indexes = current_output_indexes


class ReplaceIndexNode(Node):

  def is_simple(self):
    return False

  def inference_range(self, name_to_range, name_to_node):
    if self.name not in name_to_range:
      left_context = self.read_attribute('left_context')
      right_context = self.read_attribute('right_context')
      chunk_size = self.read_attribute('chunk_size')
      mod = left_context % chunk_size
      input_start = (-left_context // chunk_size) * chunk_size
      if mod > 0:
        input_start -= chunk_size
      input_end = chunk_size + right_context - 1
      input_end = (input_end // chunk_size) * chunk_size
      start = input_start
      end = input_end + chunk_size - 1
      self.input_range = [input_start, input_end]
      self.output_range = [start, end]
      name_to_range[self.name] = [start, end]

  def inference_dependencies(self, output_indexes, name_to_dependencies,
                             name_to_node, subsample_factor):
    kaldi_check(len(output_indexes) > 0,
                'number of output indexes should be greater than zero.')
    dependencies = list()
    chunk_size = self.read_attribute('chunk_size')
    for i in output_indexes:
      depend = chunk_size * (i // chunk_size)
      dependencies.append(depend)
    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])
    dependencies = list(set(dependencies))
    dependencies.sort()
    name_to_dependencies[self.name] = dependencies
    self.dependencies = dependencies
    output_indexes = list(set(output_indexes))
    output_indexes.sort()
    self.output_indexes = output_indexes

  def pre_compute(self):
    forward_indexes = list()
    modulus = self.read_attribute('chunk_size')
    for idx in self.output_indexes:
      dep = int(idx // modulus) * modulus
      kaldi_check(dep in self.input_indexes,
                  f'{self.name} cannot compute index: {dep}.')
      pos = self.input_indexes.index(dep)
      forward_indexes.append(pos)
    self.attrs['forward_indexes'] = forward_indexes


class SpliceNode(Node):

  def is_simple(self):
    return False

  def inference_dim(self, name_to_dim, name_to_node):
    if 'output_dim' in self.attrs:
      output_dim = self.attrs['output_dim']
    else:
      input_name = self.inputs[0]
      if input_name in name_to_dim:
        input_dim = name_to_dim[input_name]
      else:
        kaldi_check(input_name in name_to_node,
                    f'Cannot find : {input_name}')
        input_node = name_to_node[input_name]
        input_node.inference_dim(name_to_dim, name_to_node)
        input_dim = input_node.output_dim
      if 'const_component_dim' in self.attrs:
        const_component_dim = self.attrs['const_component_dim']
      else:
        const_component_dim = 0
      context = self.read_attribute('context')
      output_dim =\
          (input_dim - const_component_dim) * len(context) +\
          const_component_dim
    self.output_dim = output_dim
    name_to_dim[self.name] = output_dim

  def inference_range(self, name_to_range, name_to_node):
    if self.name not in name_to_range:
      context = self.read_attribute('context')
      left_context = context[0]
      right_context = context[-1]
      input_name = self.inputs[0]
      if input_name in name_to_range:
        [input_start, input_end] = name_to_range[input_name]
      else:
        kaldi_check(input_name in name_to_node,
                    f'Cannot find: {input_name}.')
        input_node = name_to_node[input_name]
        input_node.inference_range(name_to_range, name_to_node)
        [input_start, input_end] = input_node.output_range
        self.input_range = [input_start, input_end]
      output_start = input_start - left_context
      output_end = input_end - right_context
      self.input_range = [input_start, input_end]
      self.output_range = [output_start, output_end]
      name_to_range[self.name] = self.output_range

  def inference_dependencies(self,
                             output_indexes,
                             name_to_dependencies,
                             name_to_node,
                             subsample_factor):
    kaldi_check(len(output_indexes) > 0,
                'number of output indexes should be greater than zero.')
    dependencies = list()
    context = self.read_attribute('context')
    for i in output_indexes:
      dependencies.extend([i + c for c in context])

    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])
    dependencies = list(set(dependencies))
    dependencies.sort()
    name_to_dependencies[self.name] = dependencies
    input_indexes = list(dependencies)
    self.dependencies = input_indexes
    new_output_indexes = output_indexes
    new_output_indexes.extend(self.output_indexes)
    new_output_indexes = list(set(new_output_indexes))
    new_output_indexes.sort()
    self.output_indexes = new_output_indexes

  def pre_compute(self):
    forward_indexes = list()
    forward_const_indexes = list()
    context = self.read_attribute('context')
    const_dim = 0
    if 'const_component_dim' in self.attrs:
      const_dim = self.read_attribute('const_component_dim')
    for idx in self.output_indexes:
      computed_indexes = [idx + c for c in context]
      kaldi_check(set(computed_indexes) <= set(self.input_indexes),
                  'Splice is not computable.')
      forward_index = [self.input_indexes.index(i) for i in computed_indexes]
      forward_indexes.extend(forward_index)
      if const_dim > 0:
          pos = forward_index[0]
          forward_const_indexes.append(pos)
    self.attrs['forward_indexes'] = forward_indexes
    if const_dim > 0:
      self.attrs['forward_const_indexes'] = forward_const_indexes


class SubsampleNode(Node):

  def pre_compute(self):
    forward_indexes = list()
    for idx in self.output_indexes:
      kaldi_check(idx in self.input_indexes,
                  f'{self.name} cannot compute index: {idx}')
      pos = self.input_indexes.index(idx)
      forward_indexes.append(pos)
    self.set_attribute('forward_indexes', forward_indexes)


def make_node(name: str, node_type: str, inputs: List[str], outputs: List[str],
              attrs=None, consts=None) -> Node:
  """Make node.

  Args:
    name: node name.
    node_type: node type.
    inputs: input node names.
    outputs: output node names.
    attrs: attributes, default is None.
    consts: consts, default is None.

  Returns:
    Node.
  """
  if node_type == KaldiOpType.Gemm.name:
    return GemmNode(name, node_type, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Append.name:
    return AppendNode(name, node_type, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Identity.name:
    return IdentityNode(name, node_type, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Offset.name:
    return OffsetNode(name, node_type, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.ReplaceIndex.name:
    return ReplaceIndexNode(name, node_type, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Splice.name:
    return SpliceNode(name, node_type, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Subsample.name:
    return SubsampleNode(name, node_type, inputs, outputs, attrs, consts)
  else:
    return Node(name, node_type, inputs, outputs, attrs, consts)
