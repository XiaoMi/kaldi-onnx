#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Convert kaldi model to tensorflow pb."""
import argparse
import logging
import os
from pathlib import Path

import six

from converter.common import *
from converter.graph import Graph
from converter.node import make_node
from converter.parser import Parser
from converter.utils import kaldi_check


class Converter:
  """Kaldi model to tensorflow pb converter.

  Attributes:
    __chunk_size: kaldi nnet3 default chunk size is 21,
                  converted tf model support dynamic chunk size.
    __subsample_factor: kaldi nnet3 default subsample factor is 3,
                        so 21 frames input (210ms) will output 7 frames for decode.
    __nnet3_file: kaldi's nnet3 model file.
    __left_context: left context of model.
    __right_context: right context of model.
    __out_pb_file: output tensorflow pb file path.
  """

  def __init__(self, nnet3_file, left_context, right_context, out_pb_file):
    """Initialize.

    Args:
      nnet3_file: kaldi's nnet3 model file.
      left_context: left context of model.
      right_context: right context of model.
      out_pb_file: output tensorflow pb file path.
    """
    self.__chunk_size = 21
    self.__subsample_factor = 3
    self.__components = []
    self.__nodes = []
    self.__inputs = []
    self.__outputs = []
    self.__input_dims = {}
    self.__nnet3_file = nnet3_file
    self.__left_context = left_context
    self.__right_context = right_context
    self.__out_pb_file = out_pb_file
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  def run(self):
    """Start convert."""
    self.__parse_nnet3_file()
    self.__components_to_nodes()
    g = Graph(self.__nodes,
              self.__inputs,
              self.__outputs,
              self._batch,
              self._chunk_size,
              self.__left_context,
              self.__right_context,
              self._modulus,
              self.__input_dims,
              self._subsample_factor,
              self._nnet_type,
              self._fuse_lstm,
              self._fuse_stats).run()

  def __parse_nnet3_file(self):
    """Parse kaldi's nnet3 model file to get components."""
    logging.info(f'Start parse nnet3 model file: {self.__nnet3_file}.')
    with self.__nnet3_file.open(encoding='utf-8') as nnet3_line_buffer:
      self.__components = Parser(nnet3_line_buffer).run()

  def __components_to_nodes(self):
    """Convert all kaldi's nnet3 components to nodes."""
    logging.info('Convert nnet3 components to nodes.')
    nodes = []
    for component in self.__components:
        kaldi_check('type' in component,
                    "'type' is required in component: %s" % component)
        type = component['type']
        if type in KaldiOps:
            node = self.__component_to_node(component)
            nodes.append(node)
        elif type == 'Input':
            self.__input_component_to_node(component)
        elif type == 'Output':
            self.__output_component_to_node(component)
        else:
            raise Exception(
                "Unrecognised component type: {0}.".format(type))
    self._nodes = nodes

  def __component_to_node(self, component):
    """Convert one kaldi's nnet3 component to node."""
    cond = 'input' in component and 'name' in component and 'type' in component
    msg = f'"input", "name" and "type" are required: {component}'
    kaldi_check(cond, msg)

    inputs = component['input']
    name = component['name']
    type = component['type']

    if not isinstance(inputs, list):
      inputs = [inputs]

    inputs = [input if isinstance(input, six.string_types)
              else str(input)
              for input in inputs]

    attrs = {}
    if type in ATTRIBUTE_NAMES:
      attrs_names = ATTRIBUTE_NAMES[type]
      for key, value in component.items():
        if key in attrs_names:
          attrs[key] = value

    if type == KaldiOpType.ReplaceIndex.name:
      attrs['chunk_size'] = self.__chunk_size
      attrs['left_context'] = self.__left_context
      attrs['right_context'] = self.__right_context

    consts = {}
    if type in CONSTS_NAMES:
      param_names = CONSTS_NAMES[type]
      for p_name in param_names:
        if p_name in component:
          p_values = component[p_name]
          p_tensor_name = name + '_' + p_name
          consts[p_tensor_name] = p_values
          inputs.append(p_tensor_name)
    return make_node(name, type, inputs, [name], attrs, consts)

  def __input_component_to_node(self, component):
    """Convert kaldi's nnet3 input component to node."""
    cond = 'dim' in component or 'input_dim' in component
    msg = f'"dim" or "input_dim" attribute is required: {component}'
    kaldi_check(cond, msg)

    has_input_dim = 'input_dim' in component
    dim = component['input_dim'] if has_input_dim else component['dim']
    self.__input_dims[component['name']] = int(dim)
    self.__inputs.append(component['name'])

  def __output_component_to_node(self, component):
    """Convert kaldi's nnet3 output component to node."""
    self.__outputs.extend(component['input'])


def __main():
  """Main function."""
  desc = "convert kaldi nnet3 model to tensorflow pb."
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument("nnet3_file", type=Path, help="kaldi's nnet3 model file.")
  parser.add_argument("left_context", type=int, help="left context.")
  parser.add_argument("right_context", type=int, help="right context.")
  parser.add_argument("out_pb_path", type=Path, help="out tensorflow pb path.")
  args = parser.parse_args()
  Converter(args.nnet3_file, args.left_context, args.right_context,
            args.out_pb_path).run()


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                      level=logging.INFO)
  __main()
