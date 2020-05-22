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
from __future__ import print_function

from onnx import checker, defs, helper, numpy_helper, onnx_pb

from converter.node import *

_LOG = logging.getLogger(__name__)

INTERNAL_NAME = 1
PREFERRED_OPSET = 7
ONNX_UNKNOWN_DIMENSION = -1

DOMAIN = "ai.kaldi.dnn"
PRODUCER = "kaldi2onnx"


class Graph(object):

    def __init__(self,
                 nodes,
                 inputs,
                 outputs,
                 batch,
                 chunk_size,
                 left_context,
                 right_context,
                 modulus,
                 input_dims,
                 subsample_factor,
                 nnet_type,
                 fuse_lstm=True,
                 fuse_stats=True,
                 target=None,
                 opset=None,
                 extra_opst=None):
        self._nodes = nodes
        self._nodes_by_name = {}

        self._fuse_lstm = fuse_lstm
        self._fuse_stats = fuse_stats

        self._shapes = dict()
        self._consts = dict()
        self._model_inputs = inputs
        self._model_outputs = outputs
        self._inputs = inputs
        self._outputs = outputs
        self._model_input_tensors = dict()

        self._batch = batch
        self._chunk_size = chunk_size
        self._left_context = left_context
        self._right_context = right_context
        self._modulus = modulus
        self._subsample_factor = subsample_factor
        self._input_dims = input_dims

        self._dims_by_name = dict()
        self._indexes_by_name = dict()
        self._dependencies_by_name = dict()

        # for onnx model
        if target is None:
            target = []
        self._target = set(target)
        self._initializers = {}
        self._opset = self.find_opset(opset)
        self._extra_opset = extra_opst
        self._operatorsetid = DOMAIN
        self._producer_name = PRODUCER + "-nnet" + str(nnet_type)
        self._producer_version = "1.1"

    def model_interface_info(self):
        self.fetch_model_inputs_outputs()
        input_info = dict()
        cache_info = dict()
        output_info = dict()
        for input in self._model_inputs:
            if input.endswith('.IfDefined'):
                orig_input = input.replace('.IfDefined', '')
                if orig_input in self._model_outputs:
                    cache_info[orig_input] = self._shapes[orig_input]
                else:
                    input_info[orig_input] = self._shapes[input]
            else:
                input_info[input] = self._shapes[input]

        for output in self._model_outputs:
            if output not in cache_info:
                output_info[output] = self._shapes[output]
        return input_info, output_info, cache_info

    def run(self):
        self.prepare_graph()
        # do some inference
        self.inference_input_ranges()
        self.inference_dependencies()
        self.inference_indexes()
        self.add_subsample_nodes()
        self.precompute()
        self.inference_dims()
        self.inference_shapes()
        # make onnx model
        _LOG.info("model has %s nodes." % len(self._nodes))
        onnx_model = self.make_model()
        return onnx_model

    def prepare_graph(self):
        _LOG.info("Prepare Graph.")
        # copy all nodes' consts into self._consts and
        # save their shapes in self._shapes
        self.get_consts()
        # check inputs and outputs
        self.fetch_inputs_outputs()
        # make nodes in order and update nodes_by_name
        self.reorder_nodes()
        self.update_nodes_by_name()
        # add PadContext for 'input' if the model has left or right context
        # fuse statistics extraction and pooling, lstm cell
        self.fuse_nodes()
        self.add_cache_nodes()
        self.fetch_model_inputs_outputs()

    def add_subsample_nodes(self):
        subsample_nodes = dict()
        if self._subsample_factor > 1:
            for node in self._nodes:
                if node.is_simple():
                    input_indexes = node.input_indexes
                    output_indexes = node.output_indexes
                    if len(output_indexes) < len(input_indexes):
                        input_name = node.inputs[0]
                        subsample_name = input + '.subsample.' + node.name
                        if subsample_name not in subsample_nodes:
                            subsample_inputs = [input_name]
                            attrs = {}
                            subsample_node = \
                                make_node(subsample_name,
                                          KaldiOpType.Subsample.name,
                                          subsample_inputs,
                                          [subsample_name], attrs)
                            subsample_node.input_indexes = input_indexes
                            subsample_node.output_indexes = output_indexes
                            subsample_nodes[subsample_name] = subsample_node
                        else:
                            subsample_node = subsample_nodes[subsample_name]
                            if set(output_indexes) != \
                                    set(subsample_node.output_indexes):
                                subsample_inputs = [input]
                                attrs = {}
                                subsample_name = node.name + subsample_name
                                subsample_node = \
                                    make_node(subsample_name,
                                              KaldiOpType.Subsample.name,
                                              subsample_inputs,
                                              [subsample_name], attrs)
                                subsample_node.input_indexes = input_indexes
                                subsample_node.output_indexes = output_indexes
                                subsample_nodes[subsample_name] = \
                                    subsample_node
                        node.input_indexes = output_indexes
                        node.inputs[0] = subsample_name
                elif node.type == KaldiOpType.Append.name:
                    dependencies = node.dependencies
                    for i in range(len(node.inputs)):
                        input = node.inputs[i]
                        if input in self._nodes_by_name or \
                                input in self._indexes_by_name:
                            if input in self._nodes_by_name:
                                input_node = self._nodes_by_name[input]
                                output_indexes = input_node.output_indexes
                            else:
                                output_indexes = self._indexes_by_name[input]
                            if set(dependencies) < set(output_indexes):
                                subsample_name = \
                                        input + '.subsample.' + node.name
                                if subsample_name not in subsample_nodes:
                                    subsample_inputs = [input]
                                    attrs = {}
                                    subsample_node = \
                                        make_node(subsample_name,
                                                  KaldiOpType.Subsample.name,
                                                  subsample_inputs,
                                                  [subsample_name], attrs)
                                    subsample_node.input_indexes = \
                                        output_indexes
                                    subsample_node.output_indexes = \
                                        dependencies
                                    subsample_node.dependencies = \
                                        output_indexes
                                    node.input_indexes = dependencies
                                    node.inputs[i] = subsample_name
                                    subsample_nodes[subsample_name] = \
                                        subsample_node
        if len(subsample_nodes) > 0:
            for name, node in subsample_nodes.items():
                self._nodes.append(node)
                self._nodes_by_name[name] = node
            self.fetch_inputs_outputs()
            self.reorder_nodes()

    def add_cache_nodes(self):
        cache_nodes = list()
        for node in self._nodes:
            input = node.inputs[0]
            if node.type == KaldiOpType.IfDefined.name and \
                    not input.endswith('.IfDefined'):
                if input in self._nodes_by_name:
                    input_node = self._nodes_by_name[input]
                    cache_node_name = input_node.name + '.Cache'
                    cache_inputs = [input_node.name]
                    cache_node = make_node(cache_node_name,
                                           KaldiOpType.Identity.name,
                                           cache_inputs,
                                           [cache_node_name])
                    cache_nodes.append(cache_node)
                    node.inputs.append(cache_node_name + '.IfDefined')
                else:
                    cache_node_name = input + '.Cache'
                    cache_inputs = [input]
                    cache_node = make_node(cache_node_name,
                                           KaldiOpType.Identity.name,
                                           cache_inputs,
                                           [cache_node_name])
                    node.inputs.append(cache_node_name + '.IfDefined')
                    cache_nodes.append(cache_node)
        if len(cache_nodes) > 0:
            self._nodes.extend(cache_nodes)
            self.fetch_inputs_outputs()
            self.reorder_nodes()

    def get_consts(self):
        for node in self._nodes:
            for const_name, const in node.consts.items():
                self._consts[const_name] = const
                self._shapes[const_name] = const.shape

    def init_indexes_by_name(self):
        input_start_idx = -self._left_context
        input_end_idx = self._chunk_size + self._right_context - 1
        for name in self._inputs:
            if name in [INPUT_NAME, '0']:
                indexes = list()
                for i in range(input_start_idx,
                               input_end_idx + 1):
                    indexes.append(i)
                self._indexes_by_name[name] = indexes
            elif name == IVECTOR_NAME:
                start = self._chunk_size * (
                        input_start_idx // self._chunk_size)
                end = self._chunk_size * (input_end_idx // self._chunk_size)
                indexes = [i for i in range(start, end + 1, self._chunk_size)]
                self._indexes_by_name[name] = indexes

    def inference_dims(self):
        _LOG.info("Inference dims")
        kaldi_check(INPUT_NAME or '0' in self._input_dims,
                    "cannot find input dim.")
        self._dims_by_name.update(self._input_dims)
        for node in self._nodes:
            node.inference_dim(self._dims_by_name, self._nodes_by_name)

    def inference_input_ranges(self):
        input_start_idx = -self._left_context
        input_end_idx = self._chunk_size + self._right_context - 1
        ranges_by_name = dict()
        input_range = [input_start_idx, input_end_idx]
        ranges_by_name[INPUT_NAME] = input_range
        ranges_by_name['0'] = input_range
        if IVECTOR_NAME in self._inputs:
            start = self._chunk_size * (input_start_idx // self._chunk_size)
            end = self._chunk_size * (input_end_idx // self._chunk_size)
            end += self._chunk_size - 1
            ranges_by_name[IVECTOR_NAME] = [start, end]
        for node in self._nodes:
            node.inference_range(ranges_by_name, self._nodes_by_name)

    def infer_node_dependencies(self, node, output_indexes):
        node.inference_dependencies(output_indexes,
                                    self._dependencies_by_name,
                                    self._nodes_by_name,
                                    self._subsample_factor)
        if node.type == KaldiOpType.IfDefined.name:
            current_dependencies = node.output_indexes
        else:
            current_dependencies = node.dependencies
        for input in node.inputs:
            if input.endswith('.IfDefined'):
                input_name = input.replace('.IfDefined', '')
            else:
                input_name = input
            if input_name in self._nodes_by_name:
                input_node = self._nodes_by_name[input_name]
                checked = \
                    set(current_dependencies) <= set(input_node.output_indexes)
                checked = checked and len(input_node.dependencies) > 0
                if not checked or input_name not in self._dependencies_by_name:
                    self.infer_node_dependencies(input_node,
                                                 current_dependencies)

    def inference_dependencies(self):
        _LOG.info("Inference dependencies.")
        final_output_indexes = self.init_output_indexes()
        for name in self._model_outputs:
            output_indexes = final_output_indexes
            if name in self._nodes_by_name and \
                    name + '.IfDefined' not in self._inputs:
                node = self._nodes_by_name[name]
                self.infer_node_dependencies(node, output_indexes)

    def inference_indexes(self):
        _LOG.info("Inference indexes")
        self.init_indexes_by_name()
        for node in self._nodes:
            node.inference_index(self._indexes_by_name,
                                 self._nodes_by_name)

    def init_output_indexes(self):
        output_indexes = list()
        i = 0
        while i < self._chunk_size:
            output_indexes.append(i)
            i += self._subsample_factor
        return output_indexes

    def print_graph_info(self):
        _LOG.info("graph shapes:")
        for s in self._shapes:
            _LOG.info(s, ":", self._shapes[s])
        for node in self._nodes:
            node.info()

    def fetch_inputs_outputs(self):
        self._inputs = list()
        self._outputs = list()
        all_inputs = []
        all_outputs = []
        for node in self._nodes:
            all_inputs.extend(node.inputs)
            all_outputs.extend(node.outputs)
        self._inputs = [i for i in all_inputs if i not in all_outputs]
        outputs = [i for i in all_outputs if i not in all_inputs]
        self._inputs = list(set(self._inputs))
        self._outputs.extend(outputs)
        self._outputs = list(set(self._outputs))

    def fetch_model_inputs_outputs(self):
        self._model_inputs = list()
        self._model_outputs = list()
        all_inputs = []
        all_outputs = []
        for node in self._nodes:
            for input in node.inputs:
                if input not in self._consts:
                    all_inputs.append(input)
            all_outputs.extend(node.outputs)
        self._model_inputs = [i for i in all_inputs
                              if i not in all_outputs]
        outputs = [i for i in all_outputs if i not in all_inputs]
        self._model_inputs = list(set(self._model_inputs))
        self._model_outputs.extend(outputs)
        self._model_outputs = list(set(self._model_outputs))

    def reorder_nodes(self, ifdefine=True):
        updated_nodes = []
        checked_names = []
        checked_names.extend(self._inputs)
        nodes_need_check = self._nodes[:]

        while len(nodes_need_check) > 0:
            for node in nodes_need_check:
                depend_inputs = [input for input in node.inputs
                                 if input not in node.consts]
                if set(depend_inputs) <= set(checked_names)\
                        or (node.type == KaldiOpType.IfDefined.name
                            and ifdefine and 'IfDefined' in node.inputs[0]):
                    updated_nodes.append(node)
                    checked_names.append(node.name)
                    nodes_need_check.remove(node)
        self._nodes = updated_nodes
        for node in self._nodes:
            del node.nexts[:]
        for node in self._nodes:
            self._nodes_by_name[node.name] = node
            for input in node.inputs:
                if input in self._nodes_by_name:
                    input_node = self._nodes_by_name[input]
                    input_node.nexts.append(node.name)

    def init_input_shape(self):
        for name in self._inputs:
            if name in [IVECTOR_NAME, INPUT_NAME, '0']:
                self._shapes[name] = [self._batch,
                                      len(self._indexes_by_name[name]),
                                      self._input_dims[name]]

    def init_inputs(self):
        for name in self._inputs:
            if name not in self._consts:
                self.add_placeholder_op(name)

    def inference_shapes(self):
        _LOG.info("Inference shapes")
        self.init_input_shape()
        for node in self._nodes:
            node.inference_shape(self._batch,
                                 self._shapes,
                                 self._nodes_by_name)

    def fuse_dynamic_lstm(self,
                          node,
                          lstm_input,
                          nodes_before,
                          nodes_after):
        _LOG.info("Fuse LstmNonlinear.")
        out_ifdef = nodes_before[0]
        offset_out = out_ifdef.read_attribute('offset')
        prev_out_name = out_ifdef.inputs[0]

        first_affine = nodes_before[2]

        cell_ifdef = nodes_before[3]
        offset_cell = cell_ifdef.read_attribute('offset')
        prev_cell_name = cell_ifdef.inputs[0]

        second_affine = nodes_after[2]
        scale_node = nodes_after[-3]
        scale = scale_node.read_attribute('scale')

        inputs = list()
        inputs.append(lstm_input)
        inputs.append(out_ifdef.inputs[0])
        inputs.append(cell_ifdef.inputs[0])
        inputs.append(first_affine.inputs[1])
        inputs.append(node.inputs[1])
        inputs.append(second_affine.inputs[1])

        prev_cell = nodes_after[-2]
        prev_out = nodes_after[-1]

        prev_cell_dim = prev_cell.read_attribute('dim')
        prev_out_dim = prev_out.read_attribute('dim')

        out_dimrange = nodes_after[3]
        prev_out_offset = out_dimrange.read_attribute('offset')

        lstm_attrs = dict()
        lstm_attrs['prev_out_delay'] = offset_out
        lstm_attrs['prev_cell_delay'] = offset_cell
        lstm_attrs['scale'] = scale
        lstm_attrs['prev_out_dim'] = prev_out_dim
        lstm_attrs['prev_cell_dim'] = prev_cell_dim
        lstm_attrs['prev_out_offset'] = prev_out_offset
        lstm_attrs['chunk_size'] = self._chunk_size

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

        outputs = list()
        outputs.append(node_name)
        outputs.append(prev_out_name.replace(".IfDefined", ""))
        outputs.append(prev_cell_name.replace(".IfDefined", ""))

        prev_cell_shape = [self._batch, abs(offset_cell), prev_cell_dim]
        prev_out_shape = [self._batch, abs(offset_out), prev_out_dim]

        self._shapes[inputs[1]] = prev_out_shape
        self._shapes[inputs[2]] = prev_cell_shape
        self._shapes[outputs[1]] = prev_out_shape
        self._shapes[outputs[2]] = prev_cell_shape

        dynamic_lstm_node = make_node(node_name,
                                      KaldiOpType.DynamicLSTM.name,
                                      inputs,
                                      outputs,
                                      lstm_attrs,
                                      consts)
        return dynamic_lstm_node

    def remove_nodes(self, nodes_to_remove):
        self._nodes_by_name.clear()
        new_nodes = [node for node in self._nodes
                     if node not in nodes_to_remove]
        self._nodes = new_nodes

    def add_new_nodes(self, new_nodes):
        for node in new_nodes:
            kaldi_check(node.name not in self._nodes,
                        "Node(%s) is already in graph." % node.name)
            self._nodes.append(node)

    def replace_inputs_outputs(self, inputs_to_replace, outputs_to_replace):
        for node in self._nodes:
            for i in range(len(node.inputs)):
                if node.inputs[i] in inputs_to_replace:
                    node.inputs[i] = inputs_to_replace[node.inputs[i]]
            for i in range(len(node.nexts)):
                if node.nexts[i] in outputs_to_replace:
                    node.nexts[i] = outputs_to_replace[node.nexts[i]]
        for i in range(len(self._outputs)):
            if self._outputs[i] in inputs_to_replace:
                self._outputs[i] = inputs_to_replace[self._outputs[i]]

    def update_with_fused_nodes(self,
                                fused_nodes,
                                nodes_to_remove,
                                outputs_to_replace,
                                inputs_to_replace):
        self.remove_nodes(nodes_to_remove)
        self.add_new_nodes(fused_nodes)
        self.replace_inputs_outputs(inputs_to_replace,
                                    outputs_to_replace)
        self.update_nodes_by_name()
        self.reorder_nodes(False)

    def update_nodes_by_name(self):
        self._nodes_by_name.clear()
        for node in self._nodes:
            self._nodes_by_name[node.name] = node

    def check_before_lstm(self, lstm_node):
        ifdef_inputs = []
        input = lstm_node.inputs[0]
        if input not in self._nodes_by_name:
            return False, None, None, None
        append_a = self._nodes_by_name[input]
        if append_a.type != KaldiOpType.Append.name or \
                len(append_a.inputs) != 2:
            return False, None, None, None
        sup_affine_name = append_a.inputs[0]
        sup_ifdef_a_name = append_a.inputs[1]
        if sup_affine_name not in self._nodes_by_name or \
                sup_ifdef_a_name not in self._nodes_by_name:
            return False, None, None, None
        affine = self._nodes_by_name[sup_affine_name]
        ifdef_a = self._nodes_by_name[sup_ifdef_a_name]
        if affine.type != KaldiOpType.Gemm.name or \
                ifdef_a.type != KaldiOpType.IfDefined.name:
            return False, None, None, None
        ifdef_inputs.append(ifdef_a.inputs[0])
        if affine.inputs[0] not in self._nodes_by_name:
            return False, None, None, None
        append_b = self._nodes_by_name[affine.inputs[0]]
        if append_b.type != KaldiOpType.Append.name or \
                len(append_b.inputs) != 2:
            return False, None, None, None
        input_name = append_b.inputs[0]
        ifdef_b_name = append_b.inputs[1]
        if ifdef_b_name in self._nodes_by_name:
            ifdef_b = self._nodes_by_name[ifdef_b_name]
            if ifdef_b.type == KaldiOpType.IfDefined.name:
                ifdef_inputs.append(ifdef_b.inputs[0])
                nodes_before = [ifdef_b,
                                append_b,
                                affine,
                                ifdef_a,
                                append_a]
                return (True,
                        input_name,
                        ifdef_inputs,
                        nodes_before)
        return False, None, None, None

    def check_after_lstm(self, lstm_node, ifdef_inputs):
        nodes_after = []
        if len(lstm_node.nexts) != 2:
            return False, None, None
        slice_a_name = lstm_node.nexts[0]
        slice_b_name = lstm_node.nexts[1]
        if slice_a_name not in self._nodes_by_name or \
                slice_b_name not in self._nodes_by_name:
            return False, None, None
        slice_a = self._nodes_by_name[slice_a_name]
        slice_b = self._nodes_by_name[slice_b_name]
        if slice_a.type != KaldiOpType.DimRange.name or \
                slice_b.type != KaldiOpType.DimRange.name:
            return False, None, None
        if slice_a.nexts[0] not in self._nodes_by_name or \
                slice_b.nexts[0] not in self._nodes_by_name:
            return False, None, None
        left_node = self._nodes_by_name[slice_a.nexts[0]]
        right_node = self._nodes_by_name[slice_b.nexts[0]]
        check_left_right = (left_node.type == KaldiOpType.Gemm.name and
                            right_node.type == KaldiOpType.Append.name) or \
                           (left_node.type == KaldiOpType.Append.name and
                            right_node.type == KaldiOpType.Gemm.name)
        if check_left_right is False:
            return False, None, None
        if left_node.type == KaldiOpType.Gemm.name and \
                right_node.type == KaldiOpType.Append.name:
            append_node = right_node
            affine_node = left_node
            nodes_after.append(slice_a)
            nodes_after.append(slice_b)
        else:
            append_node = left_node
            affine_node = right_node
            nodes_after.append(slice_b)
            nodes_after.append(slice_a)
        nodes_after.append(affine_node)
        if len(append_node.inputs) != 2:
            return False, None, None
        if slice_b_name == append_node.inputs[1]:
            dim_range_b_name = append_node.inputs[0]
        else:
            dim_range_b_name = append_node.inputs[1]
        if dim_range_b_name not in affine_node.nexts or \
                append_node.nexts[0] not in self._nodes_by_name:
            return False, None, None
        nodes_after.append(self._nodes_by_name[dim_range_b_name])
        nodes_after.append(append_node)
        scale_node = self._nodes_by_name[append_node.nexts[0]]
        if scale_node.type != KaldiOpType.Scale.name or \
                len(scale_node.nexts) != 2:
            return False, None, None
        nodes_after.append(scale_node)
        if scale_node.nexts[0] not in self._nodes_by_name or \
                scale_node.nexts[1] not in self._nodes_by_name:
            return False, None, None
        last_dim_range_0 = self._nodes_by_name[scale_node.nexts[0]]
        last_dim_range_1 = self._nodes_by_name[scale_node.nexts[1]]
        if last_dim_range_0.type != KaldiOpType.DimRange.name or \
                last_dim_range_1.type != KaldiOpType.DimRange.name:
            return False, None, None
        if (last_dim_range_0.name in ifdef_inputs or
            last_dim_range_0.name + '.IfDefined' in ifdef_inputs) and \
                (last_dim_range_1.name in ifdef_inputs or
                 last_dim_range_1.name + '.IfDefined' in ifdef_inputs):
            if last_dim_range_0.name == ifdef_inputs[0] or \
                    last_dim_range_0.name + '.IfDefined' == ifdef_inputs[0]:
                nodes_after.append(last_dim_range_0)
                nodes_after.append(last_dim_range_1)
            else:
                nodes_after.append(last_dim_range_1)
                nodes_after.append(last_dim_range_0)
            return True, affine_node.name, nodes_after
        return False, None, None

    def check_extraction_pooling(self, node):
        if len(node.inputs) != 1:
            _LOG.info(node.name, "Inputs > 1 ")
            return None
        input_name = node.inputs[0]
        if input_name not in self._nodes_by_name:
            _LOG.info(input_name, "not in nodes by name ")
            return None
        extraction_node = self._nodes_by_name[input_name]
        if extraction_node.type != \
                KaldiOpType.StatisticsExtraction.name:
            _LOG.info(input_name, "is not StatisticsExtraction.")
            return None
        if len(node.nexts) == 1:
            return [extraction_node, node]
        else:
            _LOG.info("nexts > 1.")
            return None

    def check_fuse_extraction_pooling(self, node):
        extract_pooling_pack = self.check_extraction_pooling(node)
        if extract_pooling_pack is not None:
            _LOG.info("Fuse StatisticsExtraction/StatisticsPooling "
                      "to ExtractPooling.")
            extraction_node = extract_pooling_pack[0]
            pooling_node = extract_pooling_pack[1]

            extract_input_dim = extraction_node.read_attribute('input_dim')
            extract_input_period = extraction_node.read_attribute(
                'input_period')
            extract_output_period = extraction_node.read_attribute(
                'output_period')
            include_variance = extraction_node.read_attribute(
                'include_variance')

            num_log_count = pooling_node.read_attribute(
                'num_log_count_features')
            left_context = pooling_node.read_attribute('left_context')
            right_context = pooling_node.read_attribute('right_context')
            variance_floor = pooling_node.read_attribute('variance_floor')
            output_stddevs = pooling_node.read_attribute('output_stddevs')
            pooling_input_period = pooling_node.read_attribute('input_period')
            kaldi_check(pooling_input_period == extract_output_period,
                        "StatisticsExtraction's output period should"
                        " be equal to StatisticsPooling's input period.")

            extract_pooling_attrs = {
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
                                             inputs, [node_name],
                                             extract_pooling_attrs)
            return extract_pooling_node, extract_pooling_pack
        return None, None

    def fuse_nodes(self):
        fused_nodes = []
        nodes_to_remove = list()
        outputs_to_replace = dict()
        inputs_to_replace = dict()
        for node in self._nodes:
            if node.type == KaldiOpType.StatisticsPooling.name and\
                    self._fuse_stats:
                extract_pooling_node, extract_pooling_pack = \
                    self.check_fuse_extraction_pooling(node)
                if extract_pooling_node is not None:
                    fused_nodes.append(extract_pooling_node)
                    extraction_node = extract_pooling_pack[0]
                    pooling_node = extract_pooling_pack[1]
                    outputs_to_replace[extraction_node.name] = \
                        extract_pooling_node.name
                    inputs_to_replace[pooling_node.name] = \
                        extract_pooling_node.name
                    nodes_to_remove.extend(extract_pooling_pack)
            elif node.type == KaldiOpType.LstmNonlinear.name and \
                    self._fuse_lstm:
                check_before, lstm_input, ifdef_inputs, nodes_before = \
                    self.check_before_lstm(node)
                if check_before:
                    check_after, lstm_output, nodes_after = \
                        self.check_after_lstm(node, ifdef_inputs)
                    if check_after:
                        dynamic_lstm_node = \
                            self.fuse_dynamic_lstm(node,
                                                   lstm_input,
                                                   nodes_before,
                                                   nodes_after)
                        outputs_to_replace[lstm_input] = \
                            dynamic_lstm_node.name
                        inputs_to_replace[lstm_output] = \
                            dynamic_lstm_node.name
                        nodes_to_remove.extend(nodes_before)
                        nodes_to_remove.append(node)
                        nodes_to_remove.extend(nodes_after)
                        fused_nodes.append(dynamic_lstm_node)
        if len(fused_nodes) > 0:
            self.update_with_fused_nodes(fused_nodes,
                                         nodes_to_remove,
                                         outputs_to_replace,
                                         inputs_to_replace)

    def precompute(self):
        _LOG.info("Precompute")
        for node in self._nodes:
            node.precompute()

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
        # add placeholders
        self.init_inputs()
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
        input_with_initializers.extend(
            list(self._model_input_tensors.values()))
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
                                  self._producer_name,
                                  input_with_initializers,
                                  output_tensor_values,
                                  initializer=initializers,
                                  value_info=internal_inputs)
        metadata_props = {"left_context": str(self._left_context),
                          "right_context": str(self._right_context),
                          "chunk_size": str(self._chunk_size),
                          "modulus": str(self._modulus),
                          "subsample_factor": str(self._subsample_factor)}
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
        helper.set_model_props(model_proto, metadata_props)
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
        if name not in self._model_input_tensors:
            self._model_input_tensors[name] = tensor_value_info
        else:
            raise ValueError("model input tensor already exists.")
