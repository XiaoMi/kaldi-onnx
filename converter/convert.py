# Copyright 2019 Xiaomi, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
converter.Convert - class to manage top level of converting progress
python convert.py --input=path/to/kaldi_model.mdl \
                  --output=path/to/save/onnx_model.onnx \
                  --batch=b --chunk-size=cs --nnet-type=(2 or 3) \
                  --fuse-lstm=(true or false, default is true) \
                  --fuse-stats=(true or false, default is true)
"""

import argparse
import logging
import os
import six

from common import *
from graph import Graph
from node import make_node
from parser import Nnet2Parser, Nnet3Parser
from utils import kaldi_check


class Converter(object):

    def __init__(self,
                 config_file,
                 batch,
                 chunk_size,
                 nnet_type,
                 fuse_lstm=True,
                 fuse_stats=True):

        self._components = []
        self._nodes = []
        self._inputs = []
        self._outputs = []
        self._input_dims = {}
        self._nnet_type = nnet_type
        self._batch = batch
        self._chunk_size = chunk_size
        self._config_file = config_file
        self._fuse_lstm = fuse_lstm
        self._fuse_stats = fuse_stats

    def run(self):
        # parse config file to get components
        self.parse_configs()
        # convert components to nodes, inputs and outputs
        self.convert_components()
        # to build graph, graph will take over the converting work
        g = Graph(self._nodes,
                  self._inputs,
                  self._outputs,
                  self._batch,
                  self._chunk_size,
                  self._input_dims,
                  self._nnet_type,
                  self._fuse_lstm,
                  self._fuse_stats)
        onnx_model = g.run()
        return onnx_model

    def parse_configs(self):
        if self._nnet_type == NNet3:
            parser = Nnet3Parser(self._config_file)
        elif self._nnet_type == NNet2:
            parser = Nnet2Parser(self._config_file)
        else:
            raise Exception("nnet-type should be 2 or 3.")
        self._components = parser.run()

    def convert_components(self):
        nodes = []
        for component in self._components:
            kaldi_check('type' in component,
                        "'type' is required in component: %s" % component)
            type = component['type']
            if type in KaldiOps:
                node = self.node_from_component(component)
                nodes.append(node)
            elif type == 'Input':
                self.convert_input(component)
            elif type == 'Output':
                self.convert_output(component)
            else:
                raise Exception(
                    "Unrecognised component type: {0}.".format(type))
        self._nodes = nodes

    @staticmethod
    def node_from_component(component):
        kaldi_check('name' in component
                    and 'input' in component
                    and 'type' in component,
                    "'name', 'type' and 'input'"
                    " are required in component: %s" % component)
        type = component['type']
        name = component['name']
        inputs = component['input']
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

        consts = {}
        if type in CONSTS_NAMES:
            param_names = CONSTS_NAMES[type]
            for p_name in param_names:
                if p_name in component:
                    p_values = component[p_name]
                    p_tensor_name = name + '_' + p_name
                    consts[p_tensor_name] = p_values
                    inputs.append(p_tensor_name)
        return make_node(name, type, inputs, attrs, consts)

    def convert_input(self, component):
        kaldi_check('input_dim' in component or 'dim' in component,
                    "input_dim or dim attribute is required in input"
                    " component: %s" % component)
        if 'input_dim' in component:
            dim = component['input_dim']
        else:
            dim = component['dim']
        self._input_dims[component['name']] = int(dim)

    def convert_output(self, component):
        outputs = component['input']
        self._outputs.extend(outputs)


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model file")
    parser.add_argument("--output", help="path to save output onnx model file")
    parser.add_argument('--chunk-size', type=int, dest='chunk_size',
                        help='chunk size, default is 20',
                        default=DefaultChunkSize)
    parser.add_argument('--batch', type=int, dest='batch',
                        help='batch size, default is 1', default=DefaultBatch)
    parser.add_argument('--nnet-type', type=int,
                        dest='nnet_type', help='nnet type: 2 or 3',
                        default=NNet3)
    parser.add_argument('--fuse-lstm', type=str2bool,
                        dest='fuse_lstm',
                        help='fuse lstm four parts to dynamic lstm or not,'
                             ' default is true',
                        default=True)
    parser.add_argument('--fuse-stats', type=str2bool,
                        dest='fuse_stats',
                        help='fuse StatisticsExtraction/StatisticsPooling'
                             ' or not, default is true',
                        default=True)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.input:
        with open(args.input, 'r') as config_file:
            converter = Converter(config_file,
                                  args.batch,
                                  args.chunk_size,
                                  args.nnet_type,
                                  args.fuse_lstm,
                                  args.fuse_stats)
            onnx_model = converter.run()
            if args.output:
                output_path = args.output
            else:
                output_path = os.path.splitext(args.input)
                output_path = output_path[0] + '.onnx'
            with open(output_path, "wb") as of:
                of.write(onnx_model.SerializeToString())
                logging.info("Kaldi to ONNX converting finished!")
                logging.info("The new onnx model file is: %s" % output_path)
    else:
        raise Exception("invalid input file path: {0}.".format(args.input))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO)

    main()
