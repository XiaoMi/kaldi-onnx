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
converter.graph - class to manage graph manipulation on top of ONNX
"""

from __future__ import division
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import defs, helper, checker, numpy_helper, optimizer, OperatorSetIdProto, onnx_pb
import six
from node import *

_LOG = logging.getLogger(__name__)

INTERNAL_NAME = 1
PREFERRED_OPSET = 7
ONNX_UNKNOWN_DIMENSION = -1

DOMAIN = "ai.kaldi2onnx"
PRODUCER = "kaldi2onnx"

class Graph(object):

    def __init__(self,
                 nodes,
                 inputs,
                 outputs,
                 batch,
                 chunk_size,
                 input_dims,
                 nnet_type,
                 fuse_lstm,
                 fuse_stats,
                 target=None,
                 opset=None,
                 extra_opst=None):
        if target is None:
            target = []
        self._target = set(target)
        self._nodes = nodes
        self._nodes_by_name = {}

        self._fuse_lstm = fuse_lstm
        self._fuse_stats = fuse_stats

        self.update_nodes_by_name()
        self._initializers = {}
        self._shapes = dict()
        self._consts = dict()
        self._model_inputs = {}
        self._inputs = inputs

        self._outputs = outputs

        self._opset = self.find_opset(opset)
        self._extra_opset = extra_opst

        self._batch = batch
        self._chunk_size = chunk_size
        self._input_dims = input_dims

        self._operatorsetid = DOMAIN
        self._producer_name = PRODUCER + "-nnet" + str(nnet_type)
        self._producer_version = "1.0"

        self._replace_input_tensors = {}
        self._replace_output_tensors = {}
        self._remove_nodes = []
        self._remove_inputs = []
        self._remove_outputs = []

        self._left_context = 0
        self._right_context = 0

        self._need_check_output_indexes = False

    def run(self):
        # copy all nodes' consts into _const and save their shapes in _shapes
        self.get_consts()
        # double check inputs and outputs
        self.update_input_outputs()
        # make nodes in order
        self.reorder_nodes()
        # compute left context and right context
        self.compute_context()
        # add PadContext for 'input' if the model has left or right context
        self.add_padding_node()
        # fuse statistics extraction and pooling, lstm cell
        self.fuse_nodes()
        # init input shapes
        self.init_input_shape()
        # some node needs to get some params before inference running,
        # like StatisticsExtraction, StatisticsPooling,
        #  or their's fused ExtractPooling
        self.precompute()
        # infer shapes in order
        self.infer_shapes()
        # add placeholders
        self.init_inputs()
        # make onnx model
        _LOG.info("model has %s nodes." % len(self._nodes))
        onnx_model = self.make_model()
        return onnx_model

    def add_padding_node(self):
        if self._left_context > 0 or self._right_context > 0:
            for input in self._inputs:
                if input in[INPUT_NAME, '0']:
                    node_name = input + '_pad_context'
                    inputs = [input]
                    attrs = {
                        'left_context': self._left_context,
                        'right_context': self._right_context,
                    }
                    padding_node = make_node(node_name,
                                             KaldiOpType.PadContext.name,
                                             inputs,
                                             attrs)
                    for node in self._nodes:
                        if input in node.inputs:
                            node.inputs = [node_name if item == input else item
                                           for item in node.inputs]
                    self._nodes.insert(0, padding_node)
                    self._nodes_by_name[node_name] = padding_node

    def print_graph_info(self):
        _LOG.info("graph shapes:")
        for s in self._shapes:
            _LOG.info(s, ":", self._shapes[s])
        for node in self._nodes:
            node.info()

    def get_consts(self):
        for node in self._nodes:
            for const_name, const in node.consts.items():
                self._consts[const_name] = const
                self._shapes[const_name] = const.shape

    def update_input_outputs(self):
        all_inputs  = []
        all_outputs = []
        for node in self._nodes:
            all_inputs.extend(node.inputs)
            all_outputs.extend(node.outputs)
        self._inputs = [i for i in all_inputs if i not in all_outputs]
        outputs = [i for i in all_outputs if i not in all_inputs]
        self._inputs = list(set(self._inputs))
        self._outputs.extend(outputs)
        self._outputs = list(set(self._outputs))

    def reorder_nodes(self, ifdefine=True):
        updated_nodes = []
        checked_names = []
        checked_names.extend(self._inputs)

        while len(updated_nodes) < len(self._nodes):
            for node in self._nodes:
                if node not in updated_nodes:
                    required_inputs = [input for input in node.inputs
                                       if input not in node.consts]
                    if set(required_inputs) <= set(checked_names) or \
                            (node.type == KaldiOpType.IfDefined.name and ifdefine):
                        updated_nodes.append(node)
                        checked_names.append(node.name)
        self._nodes = updated_nodes
        for node in self._nodes:
            for input in node.inputs:
                if input in self._nodes_by_name:
                    input_node = self._nodes_by_name[input]
                    input_node.nexts.append(node.name)

    def init_input_shape(self):
        for name in self._inputs:
            chunk = 1 if name == IVECTOR_NAME else self._chunk_size
            if name in self._input_dims:
                self._shapes[name] = [self._batch, chunk, self._input_dims[name]]

    def init_inputs(self):
        for name in self._inputs:
            if name not in self._consts:
                self.add_placeholder_op(name)

    def infer_shapes(self):
        for node in self._nodes:
            node.infer_shape(self._shapes)
            if node.output_shape is not None:
                self._shapes[node.name] = node.output_shape
        # need to fetch 'IfDefined's input shape from another inference
        self.check_ifdef_shape()

    def check_ifdef_shape(self):
        for node in self._nodes:
            if node.type == KaldiOpType.IfDefined.name and\
                    node.name not in self._shapes:
                if node.inputs[0] not in self._shapes:
                    origin_index = node.inputs[0].find('.IfDefined')
                    origin_input = node.inputs[0][0: origin_index]
                    if origin_input in self._shapes:
                        self._shapes[node.inputs[0]] = self._shapes[origin_input]
                node.infer_shape(self._shapes)
                if node.output_shape is not None:
                    self._shapes[node.name] = node.output_shape
                else:
                    node.info()
                for next in node.nexts:
                    next_node = self._nodes_by_name[next]
                    next_node.infer_shape(self._shapes)
                    if next_node.output_shape is not None:
                        self._shapes[next_node.name] = next_node.output_shape

    def fuse_dynamic_lstm(self,
                          node,
                          lstm_input,
                          nodes_before,
                          nodes_after):
        _LOG.info("Fuse LstmNonlinear.")
        first_ifdef = nodes_before[0]
        offset_a = first_ifdef.read_attribute('offset')
        first_affine = nodes_before[2]

        second_ifdef = nodes_before[3]
        offset_b = second_ifdef.read_attribute('offset')

        second_affine = nodes_after[2]
        scale_node = nodes_after[-3]
        scale = scale_node.read_attribute('scale')

        inputs = list()
        inputs.append(lstm_input)
        inputs.append(first_affine.inputs[1])
        inputs.append(node.inputs[1])
        inputs.append(second_affine.inputs[1])

        prev_a = nodes_after[-2]
        prev_b = nodes_after[-1]
        prev_a_dim = prev_a.read_attribute('dim')
        prev_b_dim = prev_b.read_attribute('dim')

        prev_out_offset = prev_b.read_attribute('offset')

        lstm_attrs = dict()
        lstm_attrs['prev_out_delay'] = offset_a
        lstm_attrs['prev_cell_delay'] = offset_b
        lstm_attrs['scale'] = scale
        lstm_attrs['prev_out_dim'] = prev_a_dim
        lstm_attrs['prev_cell_dim'] = prev_b_dim
        lstm_attrs['prev_out_offset'] = prev_out_offset

        if len(first_affine.inputs) == 3:
            inputs.append(first_affine.inputs[2])
            lstm_attrs['bias_a'] = 1
        if len(second_affine.inputs) == 3:
            inputs.append(second_affine.inputs[2])
            lstm_attrs['bias_b'] = 1

        node_name = node.name + '.fused'

        consts = {}
        for input in inputs:
            if input in self._consts:
                consts[input] = self._consts[input]

        dynamic_lstm_node = make_node(node_name,
                                      KaldiOpType.DynamicLSTM.name,
                                      inputs,
                                      lstm_attrs,
                                      consts)
        return dynamic_lstm_node

    def remove_inputs(self):
        new_inputs = [ipt for ipt in self._inputs
                      if ipt not in self._remove_inputs]
        self._inputs = new_inputs

    def remove_outputs(self):
        new_outputs = [opt for opt in self._outputs
                       if opt not in self._remove_outputs]
        self._outputs = new_outputs

    def safe_remove_node(self, node):
        for output in node.nexts:
            output_node = self._nodes_by_name[output]
            previous_inputs = [input for input in node.inputs
                               if input not in self._consts]
            new_inputs = []
            for input in output_node.inputs:
                if input == node.name:
                    new_inputs.extend(previous_inputs)
                else:
                    new_inputs.append(input)
            output_node.inputs = new_inputs
        del self._nodes_by_name[node.name]
        self._nodes.remove(node)

    def remove_nodes(self):
        self._nodes_by_name.clear()
        new_nodes = []
        for node in self._nodes:
            if node not in self._remove_nodes:
                new_nodes.append(node)
                self._nodes_by_name[node.name] = node
            else:
                for input in node.inputs:
                    if input in self._inputs:
                        self._remove_inputs.append(input)
                    if node.name in self._outputs:
                        self._remove_outputs.append(node.name)
        self._nodes = new_nodes

    def add_new_nodes(self, new_nodes):
        for node in new_nodes:
            kaldi_check(node.name not in self._nodes,
                        "Node(%s) is already in graph." % node.name)
            self._nodes.append(node)
            self._nodes_by_name[node.name] = node

    def update_input_output_tensors(self):
        for node in self._nodes:
            for i in range(len(node.inputs)):
                if node.inputs[i] in self._replace_input_tensors:
                    node.inputs[i] = self._replace_input_tensors[node.inputs[i]]
            for i in range(len(node.nexts)):
                if node.nexts[i] in self._replace_output_tensors:
                    node.nexts[i] = self._replace_output_tensors[node.nexts[i]]
        for i in range(len(self._outputs)):
            if self._outputs[i] in self._replace_input_tensors:
                self._outputs[i] = self._replace_input_tensors[self._outputs[i]]

    def update_with_fused_nodes(self, fused_nodes):
        self.remove_nodes()
        self.remove_inputs()
        self.remove_outputs()
        self.add_new_nodes(fused_nodes)
        self.update_input_output_tensors()
        self.reorder_nodes(False)

    def update_nodes_by_name(self):
        for node in self._nodes:
            self._nodes_by_name[node.name] = node

    def check_before_lstm(self, lstm_node):
        ifdef_inputs = []
        input = lstm_node.inputs[0]
        if input in self._nodes_by_name:
            append_a = self._nodes_by_name[input]
            if append_a.type == KaldiOpType.Concat.name and \
                    len(append_a.inputs) == 2:
                sup_affine_name = append_a.inputs[0]
                sup_ifdef_a_name = append_a.inputs[1]
                if sup_affine_name in self._nodes_by_name and \
                        sup_ifdef_a_name in self._nodes_by_name:
                    affine = self._nodes_by_name[sup_affine_name]
                    ifdef_a = self._nodes_by_name[sup_ifdef_a_name]
                    if (affine.type == KaldiOpType.Gemm.name and
                            ifdef_a.type == KaldiOpType.IfDefined.name):
                        ifdef_inputs.append(ifdef_a.inputs[0])
                        if affine.inputs[0] in self._nodes_by_name:
                            append_b = self._nodes_by_name[affine.inputs[0]]
                            if append_b.type == KaldiOpType.Concat.name and \
                                    len(append_b.inputs) == 2:
                                input_name = append_b.inputs[0]
                                ifdef_b_name = append_b.inputs[1]
                                if ifdef_b_name in self._nodes_by_name:
                                    ifdef_b = self._nodes_by_name[ifdef_b_name]
                                    if ifdef_b.type == KaldiOpType.IfDefined.name:
                                        ifdef_inputs.append(ifdef_b.inputs[0])
                                        nodes_before_lstm = [ifdef_b,
                                                             append_b,
                                                             affine,
                                                             ifdef_a,
                                                             append_a]
                                        return True, input_name, ifdef_inputs,  nodes_before_lstm
        return False, None, None, None

    def check_after_lstm(self, lstm_node, if_def_inputs):
        nodes_after_lstm = []
        if len(lstm_node.nexts) == 2:
            slice_a_name = lstm_node.nexts[0]
            slice_b_name = lstm_node.nexts[1]
            if slice_a_name in self._nodes_by_name and slice_b_name in self._nodes_by_name:
                slice_a = self._nodes_by_name[slice_a_name]
                slice_b = self._nodes_by_name[slice_b_name]
                if slice_a.type == KaldiOpType.DimRange.name and \
                        slice_b.type == KaldiOpType.DimRange.name:
                    if slice_a.nexts[0] in self._nodes_by_name and \
                            slice_b.nexts[0] in self._nodes_by_name:
                        left_node = self._nodes_by_name[slice_a.nexts[0]]
                        right_node = self._nodes_by_name[slice_b.nexts[0]]
                        if (left_node.type == KaldiOpType.Gemm.name and
                            right_node.type == KaldiOpType.Concat.name) or \
                                (left_node.type == KaldiOpType.Concat.name and
                                 right_node.type == KaldiOpType.Gemm.name):
                            if left_node.type == KaldiOpType.Gemm.name and \
                                    right_node.type == KaldiOpType.Concat.name:
                                append_node = right_node
                                affine_node = left_node
                                nodes_after_lstm.append(slice_a)
                                nodes_after_lstm.append(slice_b)
                            else:
                                append_node = left_node
                                affine_node = right_node
                                nodes_after_lstm.append(slice_b)
                                nodes_after_lstm.append(slice_a)
                            nodes_after_lstm.append(affine_node)
                            if len(append_node.inputs) == 2:
                                if slice_b_name == append_node.inputs[1]:
                                    dim_range_b_name = append_node.inputs[0]
                                else:
                                    dim_range_b_name = append_node.inputs[1]
                                if dim_range_b_name in affine_node.nexts:
                                    nodes_after_lstm.append(self._nodes_by_name[dim_range_b_name])
                                    nodes_after_lstm.append(append_node)
                                    if append_node.nexts[0] in self._nodes_by_name:
                                        scale_node = self._nodes_by_name[append_node.nexts[0]]
                                        if scale_node.type == KaldiOpType.Scale.name:
                                            if len(scale_node.nexts) == 2:
                                                nodes_after_lstm.append(scale_node)
                                                if scale_node.nexts[0] in self._nodes_by_name and \
                                                        scale_node.nexts[1] in self._nodes_by_name:
                                                    last_dim_range_0 = self._nodes_by_name[scale_node.nexts[0]]
                                                    last_dim_range_1 = self._nodes_by_name[scale_node.nexts[1]]
                                                    if last_dim_range_0.type == KaldiOpType.DimRange.name and \
                                                            last_dim_range_1.type == KaldiOpType.DimRange.name:
                                                        nodes_after_lstm.append(last_dim_range_0)
                                                        nodes_after_lstm.append(last_dim_range_1)
                                                        if (last_dim_range_0.name in if_def_inputs or
                                                            last_dim_range_0.name + '.IfDefined' in if_def_inputs) and \
                                                                (last_dim_range_1.name in if_def_inputs or
                                                                 last_dim_range_1.name + '.IfDefined' in if_def_inputs):
                                                            return True, affine_node.name, nodes_after_lstm
        return False, None, None

    def check_extraction_pooling_round(self, node):
        if len(node.inputs) == 1:
            input_name = node.inputs[0]
            if input_name in self._nodes_by_name:
                extraction_node = self._nodes_by_name[input_name]
                if extraction_node.type == KaldiOpType.StatisticsExtraction.name:
                    if len(node.nexts) == 1:
                        next_name = node.nexts[0]
                        if next_name in self._nodes_by_name:
                            round_node = self._nodes_by_name[next_name]
                            if round_node.type == KaldiOpType.Round.name:
                                return [extraction_node, node, round_node]
        return None

    def check_fuse_extraction_pooling_round(self, node):
        extract_pooling_pack = self.check_extraction_pooling_round(node)
        if extract_pooling_pack is not None:
            _LOG.info("Fuse Extraction/Pooling/Round to ExtractPooling.")
            extraction_node = extract_pooling_pack[0]
            pooling_node = extract_pooling_pack[1]
            round_node = extract_pooling_pack[2]

            extract_input_dim = extraction_node.read_attribute('input_dim')
            extract_input_period = extraction_node.read_attribute('input_period')
            extract_output_period = extraction_node.read_attribute('output_period')
            include_variance = extraction_node.read_attribute('include_variance')

            num_log_count = pooling_node.read_attribute('num_log_count_features')
            left_context = pooling_node.read_attribute('left_context')
            right_context = pooling_node.read_attribute('right_context')
            variance_floor = pooling_node.read_attribute('variance_floor')
            output_stddevs = pooling_node.read_attribute('output_stddevs')
            pooling_input_period = pooling_node.read_attribute('input_period')
            kaldi_check(pooling_input_period == extract_output_period,
                        "StatisticsExtraction's output period should"
                        " be equal to StatisticsPooling's input period.")

            round_modulus = round_node.read_attribute('modulus')

            extract_pooling_attrs = {
                'modulus': round_modulus,
                'input_period': extract_input_period,
                'output_period': extract_output_period,
                'include_variance': include_variance,
                'input_dim': extract_input_dim,
                'left_context': left_context,
                'right_context': right_context,
                'num_log_count': num_log_count,
                'variance_floor': variance_floor,
                'output_stddevs': output_stddevs,
            }

            node_name = extraction_node.name + '.fused'
            inputs = extraction_node.inputs
            extract_pooling_node = make_node(node_name,
                                             KaldiOpType.ExtractPooling.name,
                                             inputs,
                                             extract_pooling_attrs)

            self._replace_output_tensors[extraction_node.name] = node_name
            self._replace_input_tensors[round_node.name] = node_name
            self._remove_nodes.extend(extract_pooling_pack)
            return extract_pooling_node
        return None

    def fuse_nodes(self):
        fused_nodes = []
        for node in self._nodes:
            if node.type == KaldiOpType.StatisticsPooling.name and self._fuse_stats:
                extract_pooling_node = self.check_fuse_extraction_pooling_round(node)
                if extract_pooling_node is not None:
                    self._need_check_output_indexes = True
                    fused_nodes.append(extract_pooling_node)
            elif node.type == KaldiOpType.LstmNonlinear.name and self._fuse_lstm:
                check_before, lstm_input, ifdef_inputs, nodes_before = self.check_before_lstm(node)
                if check_before:
                    check_after, lstm_output, nodes_after = self.check_after_lstm(node, ifdef_inputs)
                    if check_after:
                        dynamic_lstm_node = self.fuse_dynamic_lstm(node,
                                                                   lstm_input,
                                                                   nodes_before,
                                                                   nodes_after)
                        self._replace_output_tensors[lstm_input] = dynamic_lstm_node.name
                        self._replace_input_tensors[lstm_output] = dynamic_lstm_node.name
                        self._remove_nodes.extend(nodes_before)
                        self._remove_nodes.append(node)
                        self._remove_nodes.extend(nodes_after)
                        fused_nodes.append(dynamic_lstm_node)
        if len(fused_nodes) > 0:
            self.update_with_fused_nodes(fused_nodes)

    def remove_node(self, node):
        remove_node_name = node.mame
        replace_node_name = node.inputs[0]
        for nd in self._nodes:
            inputs = nd.inputs
            inputs[:] = [x if x != remove_node_name else replace_node_name
                         for x in inputs]
            nd.inputs = inputs

    def get_input_nodes(self, node):
        input_nodes = []
        for input in node.inputs:
            if input in self._nodes_by_name:
                input_nodes.append(self._nodes_by_name[input])
        return input_nodes

    @staticmethod
    def map_to_input(node, start, end):
        node.map_to_input(start, end)
        node.start_index = start
        node.end_index = end

    def check_output_indexes(self):
        if self._need_check_output_indexes is False:
            return
        for output in self._outputs:
            if output in self._nodes_by_name:
                start = 0
                end = self._chunk_size - 1
                node = self._nodes_by_name[output]
                if node.start_index != start or node.end_index != end:
                    self.map_to_input(node, start, end)
                    [start, end] = node.read_attribute('input_time_range')
                    input_nodes = self.get_input_nodes(node)
                    is_continue = True
                    while len(input_nodes) == 1 and is_continue:
                        n = input_nodes[0]
                        self.map_to_input(n, start, end)
                        [start, end] = n.read_attribute('input_time_range')
                        input_nodes = self.get_input_nodes(n)
                        if n.type == KaldiOpType.ExtractPooling.name:
                            is_continue = False

    def precompute(self):
        input_indexes = self.get_input_indexes(0, self._chunk_size)
        self.infer_time_index(input_indexes, save_index=True)
        self.reset_append_input_index()
        self.check_output_indexes()
        for node in self._nodes:
            node.precompute()

    def compute_context(self):
        left_context = 0
        right_context = 0
        window_size = 100
        while True:
            check_left, check_right = self.evaluate_time_index(left_context,
                                                               right_context,
                                                               window_size)
            if check_left is False:
                left_context += 1
            else:
                if check_right is False:
                    right_context += 1
                else:
                    break
        self._left_context = left_context
        self._right_context = right_context
        _LOG.info("left_context: %s, right context: %s"
              % (left_context, right_context))

    def reset_append_input_index(self):
        for node in self._nodes:
            if node.type == KaldiOpType.Concat.name:
                input_nodes = self.get_input_nodes(node)
                for in_node in input_nodes:
                    if in_node.type in [KaldiOpType.ReplaceIndex.name,
                                        KaldiOpType.Round.name,
                                        KaldiOpType.ExtractPooling.name,
                                        KaldiOpType.IfDefined.name]:
                        in_node.start_index = node.start_index
                        in_node.end_index = node.end_index

    def get_input_indexes(self, start, end):
        input_indexes = [t for t in range(start, end)]
        indexes_by_name = {}
        for input in self._inputs:
            if input in[INPUT_NAME, '0']:
                indexes_by_name[input] = input_indexes
            else:
                indexes_by_name[input] = [0]
        return indexes_by_name

    def infer_time_index(self, input_indexes, save_index=False):
        for node in self._nodes:
            output_indexes = node.infer_index(input_indexes, save_index)
            input_indexes[node.name] = output_indexes

            if len(output_indexes) == 0 or output_indexes[0] > output_indexes[-1]:
                return False
        return True

    def evaluate_time_index(self, left_context, right_context, window_size=100):
        indexes_by_name = self.get_input_indexes(-left_context,
                                                 window_size + right_context)
        if self.infer_time_index(indexes_by_name) is False:
            return False, False

        check_left = True
        check_right = True
        for output in self._outputs:
            if output not in indexes_by_name:
                return False, False
            else:
                output_indexes = indexes_by_name[output]
                start = output_indexes[0]
                end = output_indexes[-1]
                check_left = check_left and (start <= 0)
                check_right = check_right and (end >= (window_size - 1))
        return check_left, check_right

    def convert_initializers(self):
        for const_name in self._consts:
            const = self._consts[const_name]
            tensor = self.kaldi_to_onnx_tensor(const, const_name)
            self._initializers[tensor.name] = tensor

    @staticmethod
    def make_name(name):
        """Make op name for inserted ops."""
        global INTERNAL_NAME
        INTERNAL_NAME += 1
        return "{}__{}".format(name, INTERNAL_NAME)

    @staticmethod
    def make_onnx_shape(shape):
        return [self.make_name("unk") if i == -1 else i for i in shape]

    @staticmethod
    def find_opset(opset):
        if opset is None or opset == 0:
            opset = defs.onnx_opset_version()
            if opset > PREFERRED_OPSET:
                opset = PREFERRED_OPSET
        return opset

    @staticmethod
    def kaldi_to_onnx_tensor(tensor, name=""):
        onnx_tensor = numpy_helper.from_array(tensor, name=name)
        return onnx_tensor

    def make_model(self):
        _LOG.info("start making ONNX model.")
        output_tensor_values = []
        for name in self._outputs:
            v = helper.make_tensor_value_info(
                name,
                onnx_pb.TensorProto.FLOAT,
                self.make_onnx_shape(self._shapes[name]))
            output_tensor_values.append(v)

        onnx_nodes = []
        for node in self._nodes:
            if node.type not in['Input', 'Output']:
                try:
                    input_names = node.inputs
                    output_names = node.outputs
                    onnx_node = helper.make_node(node.type,
                                                 input_names,
                                                 output_names,
                                                 name=node.name,
                                                 domain=self._operatorsetid,
                                                 **node.attrs)
                    onnx_nodes.append(onnx_node)
                except Exception as ex:
                    node.info()
                    raise Exception('convert failed for node: {0} err: {1}'
                                    .format(node.type, ex))

        self.convert_initializers()

        all_inputs = []
        for node in self._nodes:
            all_inputs.extend(node.inputs)

        initializers = [i for i in list(self._initializers.values())
                        if i.name in all_inputs]

        input_with_initializers = []
        initializers_names = []
        for initializer in initializers:
            val = helper.make_tensor_value_info(initializer.name,
                                                initializer.data_type,
                                                self.make_onnx_shape(
                                                    initializer.dims))
            input_with_initializers.append(val)
            initializers_names.append(initializer.name)
        input_with_initializers.extend(list(self._model_inputs.values()))
        input_tensors_names = [i for i in all_inputs
                               if i not in initializers_names or
                               i not in self._inputs]
        internal_inputs = []
        for name in input_tensors_names:
            val = helper.make_tensor_value_info(name,
                                                onnx_pb.TensorProto.FLOAT,
                                                self.make_onnx_shape(
                                                    self._shapes[name]))
            internal_inputs.append(val)

        graph = helper.make_graph(onnx_nodes,
                                  "kaldi2onnx",
                                  input_with_initializers,
                                  output_tensor_values,
                                  initializer=initializers,
                                  value_info=internal_inputs)

        kwargs = {"producer_name": self._producer_name,
                  "producer_version": self._producer_version}
        opsets = []

        imp = helper.make_operatorsetid(self._operatorsetid, 1)
        imp.version = self._opset
        opsets.append(imp)
        if self._extra_opset is not None:
            opsets.extend(self._extra_opset)
        kwargs["opset_imports"] = opsets
        model_proto = helper.make_model(graph, **kwargs)
        checker.check_model(model_proto)
        return model_proto

    @property
    def opset(self):
        return self._opset

    @property
    def initializers(self):
        return self._initializers

    def set_shape(self, name, val):
        if isinstance(val, np.ndarray):
            val = val.tolist()
        self._shapes[name] = val

    def get_shape(self, name):
        assert isinstance(name, six.text_type)
        shape = self._shapes.get(name)
        if shape:
            for i, v in enumerate(shape):
                if v is None:
                    shape[i] = -1
            if shape[0] == -1:
                shape[0] = ONNX_UNKNOWN_DIMENSION
        return shape

    def copy_shape(self, input_name, output_name):
        shape = self.get_shape(input_name)
        if shape:
            self.set_shape(output_name, shape)

    def add_initializer(self, tensor):
        self._initializers[tensor.name] = tensor

    def make_const(self, name, np_val):
        onnx_tensor = numpy_helper.from_array(np_val, name)
        self.add_initializer(onnx_tensor)

    def set_node_by_name(self, node):
        self._nodes_by_name[node.name] = node

    def add_placeholder_op(self, name):
        input_node = helper.make_tensor_value_info(name,
                                                   onnx_pb.TensorProto.FLOAT,
                                                   self.make_onnx_shape(
                                                       self._shapes[name]))
        self.add_model_input(name, input_node)

    def add_model_input(self, name, tensor_value_info):
        if name not in self._model_inputs:
            self._model_inputs[name] = tensor_value_info
        else:
            raise ValueError("model input already exist.")
