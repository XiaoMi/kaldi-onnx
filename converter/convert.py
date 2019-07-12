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
                  --conf=path/to/save/configure.conf \
                  --trans-model=path/to/save/transition.mdl \
                  --batch=b --chunk-size=cs --nnet-type=(2 or 3) \
                  --left-context=lc(required) --right-context=rc(required) \
                  --modulus=m(default is 1) \
                  --subsample-factor=sf(default is 1) \
                  --fuse-lstm=(true or false, default is true) \
                  --fuse-stats=(true or false, default is true)
Using -h or --help for more details.
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
                 nnet_file,
                 batch,
                 chunk_size,
                 left_context,
                 right_context,
                 modulus,
                 nnet_type,
                 subsample_factor=1,
                 fuse_lstm=True,
                 fuse_stats=True):
        self._components = []
        self._nodes = []
        self._inputs = []
        self._outputs = []
        self._input_dims = {}
        self._nnet_type = nnet_type
        self._batch = batch
        self._chunk_size = self.get_chunk_size(chunk_size,
                                               subsample_factor,
                                               modulus)
        self._nnet_file = nnet_file
        self._fuse_lstm = fuse_lstm
        self._fuse_stats = fuse_stats
        self._left_context = left_context
        self._right_context = right_context
        self._subsample_factor = subsample_factor
        self._modulus = modulus
        self._transition_model = []
        logging.info(
            "frames per chunk: %s, left-context: %s, right-context: %s,"
            " modulus: %s"
            % (self._chunk_size, self._left_context,
               self._right_context, self._modulus))

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
                  self._left_context,
                  self._right_context,
                  self._modulus,
                  self._input_dims,
                  self._subsample_factor,
                  self._nnet_type,
                  self._fuse_lstm,
                  self._fuse_stats)

        onnx_model = g.run()
        input_info, output_info, cache_info = g.model_interface_info()
        input_nodes_str, input_shapes_str = self.nodes_info_to_str(input_info)

        output_nodes_str, output_shapes_str = \
            self.nodes_info_to_str(output_info)

        left_context_conf = "--left-context=" + str(self._left_context) + "\n"
        right_context_conf = \
            "--right-context=" + str(self._right_context) + "\n"
        modulus_conf = "--modulus=" + str(self._modulus) + "\n"
        frames_per_chunk_conf = \
            "--frames-per-chunk=" + str(self._chunk_size) + "\n"
        subsample_factor_conf = \
            "--frame-subsampling-factor=" + str(self._subsample_factor) + "\n"
        conf_lines = [left_context_conf, right_context_conf, modulus_conf,
                      frames_per_chunk_conf, subsample_factor_conf]

        input_node_conf = "--input-nodes=" + input_nodes_str + "\n"
        input_shapes_conf = "--input-shapes=" + input_shapes_str + "\n"
        conf_lines.append(input_node_conf)
        conf_lines.append(input_shapes_conf)
        output_node_conf = "--output-nodes=" + output_nodes_str + "\n"
        output_shapes_conf = "--output-shapes=" + output_shapes_str + "\n"
        conf_lines.append(output_node_conf)
        conf_lines.append(output_shapes_conf)
        if len(cache_info) > 0:
            cache_nodes_str, cache_shapes_str =\
                self.nodes_info_to_str(cache_info)
            cache_node_conf = "--cache-nodes=" + cache_nodes_str + "\n"
            cache_shapes_conf = "--cache-shapes=" + cache_shapes_str + "\n"
            conf_lines.append(cache_node_conf)
            conf_lines.append(cache_shapes_conf)
        return onnx_model, conf_lines, self._transition_model

    @staticmethod
    def get_chunk_size(chunk, subsample_factor, modulus):
        frames_per_chunk = chunk
        while frames_per_chunk % subsample_factor > 0 or\
                frames_per_chunk % modulus > 0:
            frames_per_chunk += 1
            if frames_per_chunk >= MaxChunkSize:
                raise Exception(
                    "The chunk size(%s) is over-ranged, maximum value is %s)."
                    % (frames_per_chunk, MaxChunkSize))
        return frames_per_chunk

    @staticmethod
    def shape_to_str(shape):
        shape_str = ''
        for i in shape:
            shape_str += str(i)
            shape_str += ','
        if shape_str.endswith(','):
            return shape_str[:-1]
        return shape_str

    def nodes_info_to_str(self, node_info):
        names_str = ''
        shapes_str = ''
        for name, shape in node_info.items():
            names_str += name
            names_str += ' '
            shapes_str += self.shape_to_str(shape)
            shapes_str += ' '
        return names_str, shapes_str

    def parse_configs(self):
        if self._nnet_type == NNet3:
            parser = Nnet3Parser(self._nnet_file)
        elif self._nnet_type == NNet2:
            parser = Nnet2Parser(self._nnet_file)
        else:
            raise Exception("nnet-type should be 2 or 3.")
        self._components, self._transition_model = parser.run()

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

    def node_from_component(self, component):
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

        if type == KaldiOpType.ReplaceIndex.name:
            attrs['chunk_size'] = self._chunk_size
            attrs['left_context'] = self._left_context
            attrs['right_context'] = self._right_context
        if type == KaldiOpType.IfDefined.name:
            attrs['chunk_size'] = self._chunk_size

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

    def convert_input(self, component):
        kaldi_check('input_dim' in component or 'dim' in component,
                    "input_dim or dim attribute is required in input"
                    " component: %s" % component)
        if 'input_dim' in component:
            dim = component['input_dim']
        else:
            dim = component['dim']
        self._input_dims[component['name']] = int(dim)
        self._inputs.append(component['name'])

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
    parser.add_argument("--input", required=True,
                        help="Input model file(*.mdl, should be text format).")
    parser.add_argument("--output",
                        help="Output onnx model file(*.onnx)."
                             "Using input model file's name as default.")
    parser.add_argument("--configure", dest='conf',
                        help="Path to save configure file(*.conf)."
                             "Using output model file's name as default.")
    parser.add_argument("--trans-model", dest='trans_path',
                        help="Output transition model file(*.trans)."
                             "Using output model file's name as default.")
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
    parser.add_argument('--left-context', required=True, type=int,
                        dest='left_context',
                        help='Add Left Context')

    parser.add_argument('--right-context', required=True, type=int,
                        dest='right_context',
                        help='Add RightContext')

    parser.add_argument('--modulus', type=int,
                        dest='modulus',
                        help='Modulus of the model.', default=1)

    parser.add_argument('--subsample-factor', type=int,
                        dest='subsample_factor',
                        help='Add Subsample factor, default is 1.', default=1)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.input:
        with open(args.input, 'r') as model_file:
            converter = Converter(model_file,
                                  args.batch,
                                  args.chunk_size,
                                  args.left_context,
                                  args.right_context,
                                  args.modulus,
                                  args.nnet_type,
                                  args.subsample_factor,
                                  args.fuse_lstm,
                                  args.fuse_stats)
            onnx_model, configs, trans_model = converter.run()
            logging.info("Kaldi to ONNX converting finished!")
            if args.output:
                output_path = args.output
            else:
                output_path = os.path.splitext(args.input)
                output_path = output_path[0] + '.onnx'
            with open(output_path, "wb") as of:
                of.write(onnx_model.SerializeToString())
                logging.info("The new onnx model file is: %s" % output_path)
            if len(trans_model) > 0:
                if args.trans_path:
                    trans_path = args.trans_path
                else:
                    trans_path = os.path.splitext(output_path)
                    trans_path = trans_path[0] + '.trans'
                with open(trans_path, "w") as trans_file:
                    trans_file.writelines(trans_model)
                    logging.info(
                        "The transition model file is: %s" % trans_path)
                    trans_model_conf = "--trans-model=" + trans_path + "\n"
                    configs.append(trans_model_conf)
            if len(configs) > 0:
                if args.conf:
                    conf_path = args.conf
                else:
                    conf_path = os.path.splitext(output_path)
                    conf_path = conf_path[0] + '.conf'
                with open(conf_path, "w") as conf_file:
                    conf_file.writelines(configs)
                    logging.info("The configure file is: %s" % conf_path)
    else:
        raise Exception("invalid input file path: {0}.".format(args.input))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO)

    main()
