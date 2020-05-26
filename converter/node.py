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
converter.node - class to manage Node
"""

from __future__ import division
from __future__ import print_function

import logging

import six

from converter.common import *
from converter.utils import *

_LOG = logging.getLogger(__name__)


def make_node(name, type, inputs, outputs, attrs=None, consts=None):
    if type == KaldiOpType.Gemm.name:
        return GemmNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Append.name:
        return AppendNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Bias.name:
        return BiasNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Constant.name:
        return ConstantNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Conv1d.name:
        return Conv1dNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Conv.name:
        return ConvNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Dct.name:
        return DCTNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.DynamicLSTM.name:
        return DynamicLSTMNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.ExtractPooling.name:
        return ExtractPoolingNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Identity.name:
        return IdentityNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.IfDefined.name:
        return IfDefinedNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.LstmNonlinear.name:
        return LSTMNonLinearNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Offset.name:
        return OffsetNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.ReplaceIndex.name:
        return ReplaceIndexNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.RestrictedAttention.name:
        return RestrictedAttentionNode(name, type, inputs, outputs,
                                       attrs, consts)
    elif type == KaldiOpType.Round.name:
        return RoundNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Scales.name:
        return ScalesNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Splice.name:
        return SpliceNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.StatisticsExtraction.name:
        return StatisticsExtractionNode(name, type, inputs,
                                        outputs, attrs, consts)
    elif type == KaldiOpType.StatisticsPooling.name:
        return StatisticsPoolingNode(name, type, inputs, outputs,
                                     attrs, consts)
    elif type == KaldiOpType.SumGroup.name:
        return SumGroupNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.Subsample.name:
        return SubsampleNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.TargetRMSNorm.name:
        return TargetRMSNormNode(name, type, inputs, outputs, attrs, consts)
    elif type == KaldiOpType.PerEltScale.name:
        return PerEltScaleNode(name, type, inputs, outputs, attrs, consts)
    else:
        return Node(name, type, inputs, outputs, attrs, consts)


class Node(object):

    def __init__(self, name, type, inputs, outputs, attrs={}, consts=None):
        self.name = name
        self.type = type
        self.nexts = []
        self.output_shape = None
        self._inputs = inputs
        self._outputs = outputs
        self._input_dim = 0
        self._output_dim = 0
        self._dependencies = []
        self._input_indexes = []
        self._output_indexes = []

        self.input_range = [-100000, 100000]
        self.output_range = [-100000, 100000]

        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = attrs
            self.update_attrs()
        if consts is None:
            self.consts = {}
        else:
            self.consts = consts

    def update_attrs(self):
        if self.attrs is None:
            return
        if self.type == KaldiOpType.Gemm.name:
            self.attrs['transB'] = 1
        elif self.type == KaldiOpType.Append.name:
            self.attrs['axis'] = -1
        for key, value in self.attrs.items():
            if key in ['input_dim',
                       'dim',
                       'output_dim',
                       'const_component_dim',
                       'offset',
                       'mod',
                       'left_context', 'right_context',
                       'p'] and not isinstance(value, int):
                self.attrs[key] = int(value)
            elif key in ['target_rms', 'epsilon',
                         'count',
                         'scale',
                         'variance_floor'] and not isinstance(value, float):
                self.attrs[key] = float(value)
            elif key in ['context'] and not isinstance(value, list):
                self.attrs[key] = value.tolist()
            elif self.type == KaldiOpType.Splice.name:
                if 'context' not in self.attrs and\
                        'left_context' in self._attrs and\
                        'right_context' in self._attrs:
                    left_context = self.read_attribute('left_context')
                    right_context = self.read_attribute('right_context')
                    context = [
                        t for t in range(-left_context, right_context + 1)]
                    self.attrs['context'] = context

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if isinstance(inputs, six.string_types):
            self._inputs = [inputs]
        elif isinstance(inputs, list):
            self._inputs = inputs
        else:
            kaldi_check(False,
                        "inputs(%s) should be a list or a string!" % inputs)

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        if isinstance(outputs, six.string_types):
            self._outputs = [outputs]
        elif isinstance(outputs, list):
            self._outputs = outputs
        else:
            kaldi_check(False,
                        "outputs(%s) should be a list or a string!" % outputs)

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, dependencies):
        self._dependencies = dependencies
        # self.attrs['dependencies'] = dependencies

    @property
    def input_indexes(self):
        return self._input_indexes

    @input_indexes.setter
    def input_indexes(self, input_indexes):
        self._input_indexes = input_indexes
        # self.attrs['input_indexes'] = input_indexes

    @property
    def output_indexes(self):
        return self._output_indexes

    @output_indexes.setter
    def output_indexes(self, output_indexes):
        self._output_indexes = output_indexes
        # self.attrs['output_indexes'] = output_indexes

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, output_dim):
        self._output_dim = output_dim
        self.attrs['output_dim'] = output_dim

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, input_dim):
        self.attrs['input_dim'] = input_dim
        self._input_dim = input_dim

    def set_attribute(self, attr_name, attr_value):
        self.attrs[attr_name] = attr_value

    def read_attribute(self, attr_name):
        kaldi_check(attr_name in self.attrs, "cannot find")
        return self.attrs[attr_name]

    def info(self):
        _LOG.info("name:%s type: %s"
                  " inputs: %s,"
                  " outputs: %s,"
                  " attrs: %s,"
                  " shape: %s," %
                  (self.name,
                   self.type,
                   self.inputs,
                   self.outputs,
                   self.attrs,
                   self.output_shape))

    def inference_dim(self, dims_by_name, nodes_by_name):
        if self.name in dims_by_name:
            output_dim = dims_by_name[self.name]
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
            if self.inputs[0] in dims_by_name:
                output_dim = dims_by_name[self.inputs[0]]
                self.input_dim = output_dim
            else:
                kaldi_check(self.inputs[0] in nodes_by_name,
                            "Cannot find node: %s" % self.inputs[0])
                input_node = nodes_by_name[self.inputs[0]]
                input_node.inference_dim(dims_by_name, nodes_by_name)
                self.input_dim = input_node.output_dim
                output_dim = self.input_dim
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim

    def is_simple(self):
        return True

    def inference_shape(self, batch, shapes, nodes_by_name):
        if self.name in shapes:
            return
        output_chunk = len(self.output_indexes)
        output_shape = [batch, output_chunk, self.output_dim]
        shapes[self.name] = output_shape
        self.output_shape = output_shape

    def precompute(self):
        pass

    def inference_index(self, indexes_by_name, nodes_by_name):
        input_name = self.inputs[0]
        if input_name in indexes_by_name:
            input_indexes = indexes_by_name[input_name]
            self.input_indexes = input_indexes
        else:
            kaldi_check(input_name in nodes_by_name,
                        "Cannot find node: %s" % input_name)
            input_node = nodes_by_name[input_name]
            input_node.inference_index(indexes_by_name, nodes_by_name)
            input_indexes = indexes_by_name[input_name]
            self.input_indexes = input_indexes
        indexes_by_name[self.name] = self.output_indexes
        kaldi_check(set(self.dependencies) <= set(self.input_indexes),
                    "input indexes is sufficient for computation")

    def inference_dependencies(self,
                               output_indexes,
                               dependencies_by_name,
                               nodes_by_name,
                               subsample_factor):
        kaldi_check(len(output_indexes) > 0, "invalid output indexes values.")
        dependencies = list()
        [start, end] = self.input_range
        current_output_indexes = list()
        for index in output_indexes:
            if index in range(start, int(end + 1)):
                dependencies.append(index)
                current_output_indexes.append(index)
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        self.dependencies = dependencies
        current_output_indexes.extend(self.output_indexes)
        current_output_indexes = list(set(current_output_indexes))
        current_output_indexes.sort()
        self.output_indexes = current_output_indexes
        dependencies_by_name[self.name] = dependencies

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            input_name = self.inputs[0]
            if input_name in ranges_by_name:
                [start, end] = ranges_by_name[input_name]
            else:
                kaldi_check(input_name in nodes_by_name,
                            "Cannot find node: %s" % input_name)
                input_node = nodes_by_name[input_name]
                input_node.inference_range(ranges_by_name, nodes_by_name)
                [start, end] = input_node.output_range
            ranges_by_name[self.name] = [start, end]
            self.input_range = [start, end]
            self.output_range = [start, end]


class GemmNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'num_repeats' in self.attrs:
            num_repeats = self.attrs['num_repeats']
        else:
            num_repeats = 1
        weights_name = self.inputs[1]
        kaldi_check(weights_name in self.consts,
                    "%s is not found in const." % weights_name)
        weights_shape = self.consts[weights_name].shape
        output_dim = weights_shape[0] * num_repeats
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class AppendNode(Node):
    def is_simple(self):
        return False

    def inference_dim(self, dims_by_name, nodes_by_name):
        output_dim = 0
        for input_name in self.inputs:
            if input_name in dims_by_name:
                input_dim = dims_by_name[input_name]
            else:
                kaldi_check(input_name in nodes_by_name,
                            "Cannot find %s'." % input_name)
                input_node = nodes_by_name[input_name]
                input_node.inference_dim(dims_by_name, nodes_by_name)
                input_dim = input_node.output_dim
            output_dim += input_dim
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim

    def inference_index(self, indexes_by_name, nodes_by_name):
        input_indexes = list()
        for input_name in self.inputs:
            if input_name in indexes_by_name:
                input_indexes.extend(indexes_by_name[input_name])
        input_indexes = list(set(input_indexes))
        input_indexes.sort()
        self.input_indexes = input_indexes
        indexes_by_name[self.name] = self.output_indexes
        kaldi_check(set(self.dependencies) <= set(self.input_indexes),
                    "input indexes is sufficient for computation")

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            [start, end] = self.input_range
            for input in self.inputs:
                if input in ranges_by_name:
                    [input_start, input_end] = ranges_by_name[input]
                else:
                    kaldi_check(input in nodes_by_name,
                                "Cannot find node: %s" % input)
                    input_node = nodes_by_name[input]
                    input_node.inference_range(ranges_by_name, nodes_by_name)
                    [input_start, input_end] = input_node.output_range
                start = max(start, input_start)
                end = min(end, input_end)
            ranges_by_name[self.name] = [start, end]
            self.input_range = [start, end]
            self.output_range = [start, end]


class BiasNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            weight_name = self.inputs[-1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[-1]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class ConstantNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            weight_name = self.inputs[-1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[-1]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class ConstantFunctionNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            weight_name = self.inputs[-1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[-1]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class Conv1dNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            weight_name = self.inputs[1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            num_filters = weights_shape[-1]
            patch_stride = self.read_attribute('patch_stride')
            patch_step = self.read_attribute('patch_step')
            patch_dim = self.read_attribute('patch_dim')
            num_patches = 1 + (patch_stride - patch_dim) // patch_step
            output_dim = num_filters * num_patches
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class ConvNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            kaldi_check(len(self.inputs) >= 2,
                        "ConvNode(%s) should have at least TWO inputs." %
                        self.name)
            weight_name = self.inputs[1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            filter_shape = self.consts[weight_name].shape
            input_x_dim = self.read_attribute('input_x_dim')
            input_y_dim = self.read_attribute('input_y_dim')
            filt_x_dim = self.read_attribute('filt_x_dim')
            filt_y_dim = self.read_attribute('filt_y_dim')
            filt_x_step = self.read_attribute('filt_x_step')
            filt_y_step = self.read_attribute('filt_y_step')
            num_x_steps = 1 + (input_x_dim - filt_x_dim) // filt_x_step
            num_y_steps = 1 + (input_y_dim - filt_y_dim) // filt_y_step
            num_filters = filter_shape[1]
            output_dim = num_x_steps * num_y_steps * num_filters
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class DCTNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            dct_dim = self.read_attribute('dct_dim')
            if 'keep_dct_dim' not in self.attrs:
                dct_keep_dim = dct_dim
            else:
                dct_keep_dim = self.attrs['keep_dct_dim']
            dim = self.read_attribute('dim')
            assert(dct_dim > 0 and dim > 0 and dim % dct_dim == 0)
            output_dim = dct_keep_dim * (dim // dct_dim)
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class DynamicLSTMNode(Node):

    def is_simple(self):
        return False

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            kaldi_check(len(self.inputs) >= 4,
                        "ConvNode(%s) should have at least TWO inputs." %
                        self.name)
            weight_name = self.inputs[3]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[0]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim

    def inference_dependencies(self,
                               output_indexes, dependencies_by_name,
                               nodes_by_name, subsample_factor):
        kaldi_check(len(output_indexes) > 0, "invalid output indexes values.")
        [start, end] = self.input_range
        prev_cell_delay = self.read_attribute('prev_cell_delay')
        prev_out_delay = self.read_attribute('prev_out_delay')
        chunk_size = self.read_attribute('chunk_size')
        dependencies = list()
        out_start, out_end = output_indexes[0], output_indexes[-1]

        if prev_cell_delay % subsample_factor == 0 and\
                prev_out_delay % subsample_factor == 0:
            subsample = int(subsample_factor)
        else:
            subsample = 1
        self.attrs['subsample_factor'] = subsample
        out_end = min(out_end + subsample_factor, end + 1)
        current_output_indexes = [i for i in range(out_start,
                                                   out_end,
                                                   subsample)]
        prev_end = out_start + chunk_size
        cache_cell_start = prev_end + prev_cell_delay
        cache_out_start = prev_end + prev_out_delay
        for index in range(cache_cell_start, prev_end, subsample):
            if index not in current_output_indexes:
                current_output_indexes.append(index)
        for index in range(cache_out_start, prev_end, subsample):
            if index in current_output_indexes:
                current_output_indexes.append(index)
        for i in current_output_indexes:
            if i in range(start, int(end + 1)):
                dependencies.append(i)
            prev_cell_depend = i + prev_cell_delay
            while prev_cell_depend in range(start, int(end + 1)):
                dependencies.append(prev_cell_depend)
                prev_cell_depend += prev_cell_delay
            prev_out_depend = i + prev_out_delay
            while prev_out_depend in range(start, int(end + 1)):
                dependencies.append(prev_out_depend)
                prev_out_depend += prev_out_delay
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        dependencies_by_name[self.name] = dependencies
        self.dependencies = dependencies
        current_output_indexes.extend(self.output_indexes)
        current_output_indexes = list(set(current_output_indexes))
        current_output_indexes.sort()
        self.output_indexes = current_output_indexes

        cell_cache_indexes = list()
        out_cache_indexes = list()
        end = self.output_indexes[0] + chunk_size
        cache_cell_start = end + prev_cell_delay
        cache_out_start = end + prev_out_delay
        for index in range(cache_cell_start, end):
            if index in self.output_indexes:
                cell_cache_indexes.append(index)
        for index in range(cache_out_start, end):
            if index in self.output_indexes:
                out_cache_indexes.append(index)
        self.attrs['cell_cache_indexes'] = cell_cache_indexes
        self.attrs['out_cache_indexes'] = out_cache_indexes

    def inference_shape(self, batch, shapes, nodes_by_name):
        if self.name in shapes:
            return
        output_chunk = len(self.output_indexes)
        output_shape = [batch, output_chunk, self.output_dim]
        shapes[self.name] = output_shape
        self.output_shape = output_shape
        prev_out_dim = self.read_attribute('prev_out_dim')
        prev_cell_dim = self.read_attribute('prev_cell_dim')
        out_cache_indexes = self.read_attribute('out_cache_indexes')
        cell_cache_indexes = self.read_attribute('cell_cache_indexes')
        cache_out_shape = [batch, len(out_cache_indexes), prev_out_dim]
        cache_cell_shape = [batch, len(cell_cache_indexes), prev_cell_dim]
        shapes[self.inputs[1]] = cache_out_shape
        shapes[self.inputs[2]] = cache_cell_shape
        shapes[self.outputs[1]] = cache_out_shape
        shapes[self.outputs[2]] = cache_cell_shape

    def precompute(self):
        forward_indexes = list()
        for idx in self.output_indexes:
            dep = idx
            kaldi_check(dep in self.input_indexes,
                        "%s cannot compute index: %s" % (self.name, dep))
            pos = self.input_indexes.index(dep)
            forward_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes
        out_cache_indexes = self.read_attribute('out_cache_indexes')
        out_cache_forward_indexes = list()
        for idx in out_cache_indexes:
            dep = idx
            kaldi_check(dep in self.input_indexes,
                        "%s cannot compute index: %s" % (self.name, dep))
            pos = self.input_indexes.index(dep)
            out_cache_forward_indexes.append(pos)
        self.set_attribute('out_cache_indexes', out_cache_forward_indexes)

        cell_cache_indexes = self.read_attribute('cell_cache_indexes')
        cell_cache_forward_indexes = list()
        for idx in cell_cache_indexes:
            dep = idx
            kaldi_check(dep in self.input_indexes,
                        "%s cannot compute index: %s" % (self.name, dep))
            pos = self.input_indexes.index(dep)
            cell_cache_forward_indexes.append(pos)
        self.set_attribute('cell_cache_indexes', cell_cache_forward_indexes)


class ExtractPoolingNode(Node):

    def is_simple(self):
        return False

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            input_dim = self.read_attribute('input_dim')
            include_variance = self.read_attribute('include_variance')
            num_log_count = self.read_attribute('num_log_count')
            output_dim = input_dim + num_log_count
            if include_variance:
                output_dim += input_dim
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            if self.inputs[0] in ranges_by_name:
                self.input_range = ranges_by_name[self.inputs[0]]
            else:
                kaldi_check(self.inputs[0] in nodes_by_name,
                            "Cannot find node: %s" % self.inputs[0])
                input_node = nodes_by_name[self.inputs[0]]
                input_node.inference_range(ranges_by_name, nodes_by_name)
                [input_start, input_end] = input_node.output_range
                self.input_range = [input_start, input_end]
            ranges_by_name[self.name] = self.output_range

    def inference_dependencies(self, output_indexes, dependencies_by_name,
                               nodes_by_name, subsample_factor):
        kaldi_check(len(output_indexes) > 0, "invalid output indexes values.")
        output_period = int(self.read_attribute('output_period'))
        input_period = int(self.read_attribute('input_period'))
        left_context = self.read_attribute('left_context')
        right_context = self.read_attribute('right_context')

        current_output_indexes = list(output_indexes)
        [input_start, input_end] = self.input_range
        pooling_input_indexes = list()
        for middle_t in current_output_indexes:
            t_start = middle_t - left_context
            t_end = middle_t + right_context
            for t in range(t_start, t_end + 1, output_period):
                if t in range(input_start, input_end + 1):
                    pooling_input_indexes.append(t)
        # get StatisticExtraction input indexes
        dependencies = list()
        for index in pooling_input_indexes:
            t_start = int(output_period * (index // output_period))
            if t_start > index:
                t_start -= output_period
            t_end = t_start + output_period
            for t in range(t_start, t_end, input_period):
                if t in range(input_start, int(input_end + 1)):
                    dependencies.append(t)
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        self.dependencies = dependencies
        current_output_indexes.extend(self.output_indexes)
        current_output_indexes = list(set(current_output_indexes))
        current_output_indexes.sort()
        self.output_indexes = current_output_indexes
        dependencies_by_name[self.name] = dependencies

    def precompute(self):

        [start, end] = self.input_range
        input_period = int(self.read_attribute('input_period'))
        in_start_mod = start % input_period

        if in_start_mod > 0:
            in_start_mod -= input_period
        in_start = start - in_start_mod
        input_indexes = [t for t in range(in_start, end + 1, input_period)]
        kaldi_check(len(input_indexes) > 0,
                    " start time: %s, end time: %s" % (start, end))

        num_output_indexes = len(self.output_indexes)
        num_input_indexes = len(input_indexes)
        kaldi_check(num_output_indexes > 0, "output indexes is empty.")
        kaldi_check(num_input_indexes > 0, "input indexes is empty.")
        forward_indexes = []
        counts = []
        left_context = self.read_attribute('left_context')
        right_context = self.read_attribute('right_context')
        input_period = int(self.read_attribute('input_period'))
        for index in self.output_indexes:
            t_start = index - left_context
            t_end = index + right_context
            forward_index = [-1, -1]
            count = 0.0
            for input_t in range(t_start, t_end, input_period):
                if input_t in input_indexes:
                    input_pos = input_indexes.index(input_t)
                    assert input_pos >= 0
                    if forward_index[0] == -1:
                        forward_index[0] = input_pos
                        forward_index[1] = input_pos + 1
                        count = 1.0
                    else:
                        kaldi_check(forward_index[1] == input_pos,
                                    "input pos: %s != forward.second: %s"
                                    % (input_pos, forward_index[1]))
                        forward_index[1] = input_pos + 1
                        count += 1.0
                    kaldi_check(forward_index[0] != -1,
                                "invalid precomputed forward_index: %s"
                                % forward_index)
            forward_indexes.extend(forward_index)
            counts.append(count)

        self.attrs["forward_indexes"] = forward_indexes
        self.attrs["counts"] = counts
        self.attrs['compute_input_indexes'] = input_indexes


class IdentityNode(Node):
    def inference_index(self, indexes_by_name, nodes_by_name):
        input_name = self.inputs[0]
        if input_name in indexes_by_name:
            input_indexes = indexes_by_name[input_name]
        else:
            kaldi_check(input_name in nodes_by_name,
                        "Cannot find node: %s" % input_name)
            input_node = nodes_by_name[input_name]
            input_node.inference_index(indexes_by_name, nodes_by_name)
            input_indexes = indexes_by_name[input_name]
        self.input_indexes = input_indexes
        self.output_indexes = input_indexes
        indexes_by_name[self.name] = self.output_indexes


class IfDefinedNode(Node):

    def is_simple(self):
        return False

    def inference_dim(self, dims_by_name, nodes_by_name):
        if self.name in dims_by_name:
            output_dim = dims_by_name[self.name]
        elif 'output_dim' in self.attrs:
            output_dim = self.read_attribute('output_dim')
        else:
            input = self.inputs[0]
            if input.endswith('.IfDefined'):
                input_name = input.replace(".IfDefined", "")
            else:
                input_name = input
            if input_name in dims_by_name:
                output_dim = dims_by_name[input_name]
                self.input_dim = output_dim
            else:
                kaldi_check(input_name in nodes_by_name,
                            "Cannot find node: %s" % input_name)
                input_node = nodes_by_name[input_name]
                input_node.inference_dim(dims_by_name, nodes_by_name)
                self.input_dim = input_node.output_dim
                output_dim = self.input_dim
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            input = self.inputs[0]
            if input in ranges_by_name:
                [start, end] = ranges_by_name[input]
            else:
                if input.endswith('.IfDefined'):
                    input_name = input.replace(".IfDefined", "")
                else:
                    input_name = input
                kaldi_check(input_name in nodes_by_name,
                            "Cannot find node: %s" % input_name)
                input_node = nodes_by_name[input_name]
                [start, end] = input_node.output_range
            ranges_by_name[self.name] = [start, end]
            self.input_range = [start, end]
            self.output_range = [start, end]

    def inference_index(self, indexes_by_name, nodes_by_name):
        input_name = self.inputs[0]
        if input_name in indexes_by_name:
            input_indexes = indexes_by_name[input_name]
            self.input_indexes = input_indexes
        else:
            if input_name.endswith('.IfDefined'):
                input = input_name.replace(".IfDefined", "")
                kaldi_check(input in nodes_by_name,
                            "Cannot find node: %s" % input)
                input_node = nodes_by_name[input]
                input_indexes = input_node.output_indexes
                self.input_indexes = input_indexes
            else:
                kaldi_check(input_name in nodes_by_name,
                            "Cannot find node: %s" % input_name)
                input_node = nodes_by_name[input_name]
                input_node.inference_index(indexes_by_name, nodes_by_name)
                input_indexes = indexes_by_name[input_name]
                self.input_indexes = input_indexes
        self.output_indexes = self.input_indexes
        indexes_by_name[self.name] = self.output_indexes

    def inference_dependencies(self,
                               output_indexes,
                               dependencies_by_name,
                               nodes_by_name,
                               subsample_factor):
        offset = self.read_attribute('offset')
        chunk_size = self.read_attribute('chunk_size')
        [start, end] = self.input_range
        kaldi_check(len(output_indexes) > 0,
                    "number of output indexes should be greater than zero.")
        out_start = output_indexes[0]
        out_end = output_indexes[-1]
        subsample = subsample_factor if offset % subsample_factor == 0 else 1
        out_end = min(end + 1, out_end + subsample_factor)
        current_output_indexes = [i for i in range(out_start,
                                                   out_end,
                                                   subsample)]
        prev_end = start + chunk_size
        cache_start = prev_end + offset
        for index in range(cache_start, prev_end, subsample):
            if index not in current_output_indexes:
                current_output_indexes.append(index)
        current_output_indexes.sort()
        self.dependencies = current_output_indexes
        self.output_indexes = current_output_indexes
        dependencies_by_name[self.name] = current_output_indexes

    def inference_shape(self, batch, shapes, nodes_by_name):
        if self.name in shapes:
            return
        output_chunk = len(self.output_indexes)
        output_shape = [batch, output_chunk, self.output_dim]
        self.output_shape = output_shape
        shapes[self.name] = output_shape
        for input in self.inputs:
            if input not in shapes:
                if input.endswith('.IfDefined'):
                    input_name = input.replace('.IfDefined', '')
                    if input_name in shapes:
                        input_shape = shapes[input_name]
                    else:
                        kaldi_check(input_name in nodes_by_name,
                                    "cannot find node with name: %s" %
                                    input_name)
                        input_node = nodes_by_name[input_name]
                        input_node.inference_shape(batch,
                                                   shapes,
                                                   nodes_by_name)
                        input_shape = shapes[input_name]
                else:
                    input_shape = [batch,
                                   len(self.input_indexes),
                                   self.output_dim]
                shapes[input] = input_shape

    def precompute(self):
        forward_indexes = list()
        offset = self.read_attribute('offset')
        prev_indexes = list()
        for idx in self.output_indexes:
            dep = idx + offset
            if dep not in self.input_indexes:
                prev_indexes.append(dep)
                pos = -1
            else:
                pos = self.input_indexes.index(dep)
            forward_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes
        if len(self.inputs) >= 2:
            chunk_size = self.read_attribute('chunk_size')
            cache_forward_indexes = list()
            for idx in prev_indexes:
                dep = idx + chunk_size
                kaldi_check(dep in self.input_indexes,
                            "cannot find index:%s in: %s" %
                            (dep, self.input_indexes))
                pos = self.input_indexes.index(dep)
                cache_forward_indexes.append(pos)
            self.attrs['cache_forward_indexes'] = cache_forward_indexes


class LSTMNonLinearNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            kaldi_check(len(self.inputs) >= 2,
                        "LSTMNonLinear(%s) should have at least 2 inputs." %
                        self.name)
            params_shape = self.consts[self.inputs[1]].shape
            cell_dim = params_shape[1]
            output_dim = cell_dim * 2
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class OffsetNode(Node):

    def is_simple(self):
        return False

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            offset = self.read_attribute('offset')
            input = self.inputs[0]
            if input in ranges_by_name:
                [input_start, input_end] = ranges_by_name[input]
            else:
                kaldi_check(input in nodes_by_name,
                            "Cannot find node: %s" % input)
                input_node = nodes_by_name[input]
                input_node.inference_range(ranges_by_name, nodes_by_name)
                [input_start, input_end] = input_node.output_range
            self.input_range = [input_start, input_end]
            self.output_range = [input_start - offset, input_end - offset]
            ranges_by_name[self.name] = self.output_range

    def precompute(self):
        forward_indexes = list()
        offset = self.read_attribute('offset')
        for idx in self.output_indexes:
            dep = idx + offset
            kaldi_check(dep in self.input_indexes,
                        "input index %s is required." % dep)
            pos = self.input_indexes.index(dep)
            forward_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes

    def inference_dependencies(self, output_indexes, dependencies_by_name,
                               nodes_by_name, subsample_factor):
        kaldi_check(len(output_indexes) > 0,
                    "number of output indexes should be greater than zero.")
        offset = self.read_attribute('offset')
        current_output_indexes = list()
        for i in output_indexes:
            current_output_indexes.append(i)
        dependencies = [i + offset for i in current_output_indexes]
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        dependencies_by_name[self.name] = dependencies
        self.dependencies = dependencies
        self.output_indexes = current_output_indexes


class PerEltScaleNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            kaldi_check(len(self.inputs) >= 2,
                        "PerEltScale(%s) should have at least 2 inputs." %
                        self.name)
            weight_name = self.inputs[1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[1]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class PermuteNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            kaldi_check(len(self.inputs) >= 2,
                        "Permute(%s) should have at least 2 inputs." %
                        self.name)
            weight_name = self.inputs[1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[0]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class ReplaceIndexNode(Node):

    def is_simple(self):
        return False

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
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
            ranges_by_name[self.name] = [start, end]

    def inference_dependencies(self, output_indexes, dependencies_by_name,
                               nodes_by_name, subsample_factor):
        kaldi_check(len(output_indexes) > 0,
                    "number of output indexes should be greater than zero.")
        dependencies = list()
        chunk_size = self.read_attribute('chunk_size')
        for i in output_indexes:
            depend = chunk_size * (i // chunk_size)
            dependencies.append(depend)
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        dependencies_by_name[self.name] = dependencies
        self.dependencies = dependencies
        output_indexes = list(set(output_indexes))
        output_indexes.sort()
        self.output_indexes = output_indexes

    def precompute(self):
        forward_indexes = list()
        modulus = self.read_attribute('chunk_size')
        for idx in self.output_indexes:
            dep = int(idx // modulus) * modulus
            kaldi_check(dep in self.input_indexes,
                        "%s cannot compute index: %s" % (self.name, dep))
            pos = self.input_indexes.index(dep)
            forward_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes


class RestrictedAttentionNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            num_heads = self.read_attribute('num_heads')
            value_dim = self.read_attribute('value_dim')
            num_left_inputs = self.read_attribute('num_left_inputs')
            num_right_inputs = self.read_attribute('num_right_inputs')
            output_context = self.read_attribute('output_context')
            context_dim = num_left_inputs + 1 + num_right_inputs
            self.set_attribute('context_dim', context_dim)
            output_dim = num_heads * (value_dim + context_dim) \
                if output_context else num_heads * value_dim
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class RoundNode(Node):

    def is_simple(self):
        return False

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            modulus = self.read_attribute('modulus')
            input = self.inputs[0]
            if input in ranges_by_name:
                [input_start, input_end] = ranges_by_name[input]
            else:
                kaldi_check(input in nodes_by_name,
                            "Cannot find node: %s" % input)
                input_node = nodes_by_name[input]
                input_node.inference_range(ranges_by_name, nodes_by_name)
                [input_start, input_end] = input_node.output_range
                self.input_range = [input_start, input_end]
            output_start = int(modulus * (input_start // modulus))
            output_end = int(modulus * (input_end // modulus) + (modulus - 1))
            self.input_range = [input_start, input_end]
            self.output_range = [output_start, output_end]
            ranges_by_name[self.name] = self.output_range

    def inference_dependencies(self, output_indexes, dependencies_by_name,
                               nodes_by_name, subsample_factor):
        kaldi_check(len(output_indexes) > 0,
                    "number of output indexes should be greater than zero.")
        dependencies = list()
        modulus = self.read_attribute('modulus')
        for i in output_indexes:
            dep = int(i // modulus) * modulus
            dependencies.append(dep)
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        dependencies_by_name[self.name] = dependencies
        self.dependencies = dependencies
        new_output_indexes = output_indexes
        new_output_indexes.extend(self.output_indexes)
        new_output_indexes = list(set(new_output_indexes))
        new_output_indexes.sort()
        self.output_indexes = new_output_indexes

    def precompute(self):
        forward_indexes = list()
        modulus = self.read_attribute('modulus')
        for idx in self.output_indexes:
            dep = int(idx // modulus) * modulus
            kaldi_check(dep in self.input_indexes,
                        "%s cannot compute index: %s" % (self.name, dep))
            pos = self.input_indexes.index(dep)
            forward_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes


class ScalesNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            kaldi_check(len(self.inputs) >= 2,
                        "ScalesNode(%s) should have at least 2 inputs." %
                        self.name)
            weight_name = self.inputs[1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[0]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class SpliceNode(Node):

    def is_simple(self):
        return False

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            input_name = self.inputs[0]
            if input_name in dims_by_name:
                input_dim = dims_by_name[input_name]
            else:
                kaldi_check(input_name in nodes_by_name,
                            "Cannot find node: %s" % input_name)
                input_node = nodes_by_name[input_name]
                input_node.inference_dim(dims_by_name, nodes_by_name)
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
        dims_by_name[self.name] = output_dim

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            context = self.read_attribute('context')
            left_context = context[0]
            right_context = context[-1]
            input = self.inputs[0]
            if input in ranges_by_name:
                [input_start, input_end] = ranges_by_name[input]
            else:
                kaldi_check(input in nodes_by_name,
                            "Cannot find node: %s" % input)
                input_node = nodes_by_name[input]
                input_node.inference_range(ranges_by_name, nodes_by_name)
                [input_start, input_end] = input_node.output_range
                self.input_range = [input_start, input_end]
            output_start = input_start - left_context
            output_end = input_end - right_context
            self.input_range = [input_start, input_end]
            self.output_range = [output_start, output_end]
            ranges_by_name[self.name] = self.output_range

    def inference_dependencies(self,
                               output_indexes,
                               dependencies_by_name,
                               nodes_by_name,
                               subsample_factor):
        kaldi_check(len(output_indexes) > 0,
                    "number of output indexes should be greater than zero.")
        dependencies = list()
        context = self.read_attribute('context')
        for i in output_indexes:
            deps = [i + c for c in context]
            dependencies.extend(deps)
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        dependencies_by_name[self.name] = dependencies
        input_indexes = list(dependencies)
        self.dependencies = input_indexes
        new_output_indexes = output_indexes
        new_output_indexes.extend(self.output_indexes)
        new_output_indexes = list(set(new_output_indexes))
        new_output_indexes.sort()
        self.output_indexes = new_output_indexes

    def precompute(self):
        forward_indexes = list()
        forward_const_indexes = list()
        context = self.read_attribute('context')
        const_dim = 0
        if 'const_component_dim' in self.attrs:
            const_dim = self.read_attribute('const_component_dim')
        for idx in self.output_indexes:
            computed_indexes = [idx + c for c in context]
            kaldi_check(set(computed_indexes) <= set(self.input_indexes),
                        "Splice is not computable.")
            forward_index = [self.input_indexes.index(i)
                             for i in computed_indexes]
            forward_indexes.extend(forward_index)
            if const_dim > 0:
                pos = forward_index[0]
                forward_const_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes
        if const_dim > 0:
            self.attrs['forward_const_indexes'] = forward_const_indexes


class StatisticsExtractionNode(Node):

    def is_simple(self):
        return False

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            input_dim = self.read_attribute('input_dim')
            include_variance = self.read_attribute('include_variance')
            output_dim = 1 + input_dim
            if include_variance:
                output_dim += input_dim
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim

    def inference_dependencies(self, output_indexes,
                               dependencies_by_name,
                               nodes_by_name,
                               subsample_factor):
        output_period = int(self.read_attribute('output_period'))
        input_period = int(self.read_attribute('input_period'))
        [input_start, input_end] = self.input_range
        input_indexes = list()
        for index in output_indexes:
            t_start = int(output_period * (index // output_period))
            if t_start > index:
                t_start -= output_period
            t_end = t_start + output_period
            for t in range(t_start, t_end, input_period):
                if t in range(input_start, input_end + 1):
                    input_indexes.append(t)
        dependencies = list(input_indexes)
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        self.dependencies = dependencies
        new_output_indexes = self.output_indexes
        new_output_indexes.extend(output_indexes)
        new_output_indexes = list(set(new_output_indexes))
        new_output_indexes.sort()
        self.output_indexes = new_output_indexes
        dependencies_by_name[self.name] = dependencies

    def precompute(self):
        [start, end] = self.input_range
        input_period = int(self.read_attribute('input_period'))
        in_start_mod = start % input_period

        if in_start_mod > 0:
            in_start_mod -= input_period
        in_start = start - in_start_mod
        input_indexes = [t for t in range(in_start, end + 1, input_period)]

        num_input_indexes = len(input_indexes)
        num_output_indexes = len(self.output_indexes)
        kaldi_check(num_output_indexes > 0,
                    "%s's output indexes is empty." % self.name)
        kaldi_check(num_input_indexes > 0,
                    "%s's input indexes is empty." % self.name)
        forward_indexes = []
        counts = []
        output_period = int(self.read_attribute('output_period'))
        for index in self.output_indexes:
            t_start = int(output_period * (index // output_period))
            if t_start > index:
                t_start -= output_period
            t_end = t_start + output_period
            forward_index = [-1, -1]
            count = 0.0
            for t in range(t_start, t_end, input_period):
                input_idx = t
                if input_idx in input_indexes:
                    pos = input_indexes.index(input_idx)
                    if forward_index[0] == -1:
                        forward_index[0] = pos
                        forward_index[1] = pos + 1
                        count = 1.0
                    else:
                        kaldi_check(forward_index[1] == pos,
                                    "input pos: %s != forward.second: %s"
                                    % (pos, forward_index[1]))
                        forward_index[1] = pos + 1
                        count += 1.0
                    kaldi_check(forward_index[0] != -1,
                                "invalid precomputed forward_index: %s"
                                % forward_index)
            forward_indexes.extend(forward_index)
            counts.append(count)
        self.attrs["forward_indexes"] = forward_indexes
        self.attrs["counts"] = counts
        self.attrs['compute_input_indexes'] = input_indexes


class StatisticsPoolingNode(Node):

    def is_simple(self):
        return False

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            input_dim = self.read_attribute('input_dim')
            num_log_count_features =\
                self.read_attribute('num_log_count_features')
            output_dim = input_dim + num_log_count_features - 1
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim

    def inference_range(self, ranges_by_name, nodes_by_name):
        if self.name not in ranges_by_name:
            input = self.inputs[0]
            if input in ranges_by_name:
                self.input_range = ranges_by_name[input]
            else:
                kaldi_check(input in nodes_by_name,
                            "Cannot find node: %s" % input)
                input_node = nodes_by_name[input]
                input_node.inference_range(ranges_by_name, nodes_by_name)
                [input_start, input_end] = input_node.output_range
                self.input_range = [input_start, input_end]
            ranges_by_name[self.name] = self.output_range

    def inference_dependencies(self,
                               output_indexes,
                               dependencies_by_name,
                               nodes_by_name,
                               subsample_factor):
        input_period = int(self.read_attribute('input_period'))
        left_context = self.read_attribute('left_context')
        right_context = self.read_attribute('right_context')
        [input_start, input_end] = self.input_range
        dependencies = list()
        for middle_t in output_indexes:
            t_start = middle_t - left_context
            t_end = middle_t + right_context
            for t in range(t_start, t_end + 1, input_period):
                if t in range(input_start, input_end + 1):
                    dependencies.append(t)
        if self.name in dependencies_by_name:
            dependencies.extend(dependencies_by_name[self.name])
        dependencies = list(set(dependencies))
        dependencies.sort()
        self.dependencies = dependencies
        new_output_indexes = self.output_indexes
        new_output_indexes.extend(output_indexes)
        new_output_indexes = list(set(new_output_indexes))
        new_output_indexes.sort()
        self.output_indexes = new_output_indexes
        dependencies_by_name[self.name] = dependencies

    def precompute(self):
        num_output_indexes = len(self.output_indexes)
        num_input_indexes = len(self.input_indexes)
        kaldi_check(num_output_indexes > 0, "output indexes is empty.")
        kaldi_check(num_input_indexes > 0, "input indexes is empty.")
        forward_indexes = []
        left_context = self.read_attribute('left_context')
        right_context = self.read_attribute('right_context')
        input_period = int(self.read_attribute('input_period'))
        for index in self.output_indexes:
            t_start = index - left_context
            t_end = index + right_context
            forward_index = [-1, -1]
            for input_t in range(t_start, t_end + 1, input_period):
                if input_t in self.input_indexes:
                    input_pos = self.input_indexes.index(input_t)
                    assert input_pos >= 0
                    if forward_index[0] == -1:
                        forward_index[0] = input_pos
                        forward_index[1] = input_pos + 1
                    else:
                        kaldi_check(forward_index[1] == input_pos,
                                    "input pos: %s != forward.second: %s"
                                    % (input_pos, forward_index[1]))
                        forward_index[1] = input_pos + 1
                    kaldi_check(forward_index[0] != -1,
                                "invalid precomputed forward_index: %s"
                                % forward_index)
            forward_indexes.extend(forward_index)
        self.attrs["forward_indexes"] = forward_indexes


class SubsampleNode(Node):

    def precompute(self):
        forward_indexes = list()
        for idx in self.output_indexes:
            kaldi_check(idx in self.input_indexes,
                        "%s cannot compute index: %s" % (self.name, idx))
            pos = self.input_indexes.index(idx)
            forward_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes


class SumGroupNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            kaldi_check(len(self.inputs) >= 2,
                        "SumGroupNode(%s) should have at least 2 inputs." %
                        self.name)
            weight_name = self.inputs[1]
            kaldi_check(weight_name in self.consts,
                        "Cannot find %s in %s's consts." %
                        (weight_name, self.name))
            weights_shape = self.consts[weight_name].shape
            output_dim = weights_shape[0]
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim


class TargetRMSNormNode(Node):

    def inference_dim(self, dims_by_name, nodes_by_name):
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        else:
            if 'input_dim' in self.attrs:
                input_dim = self.read_attribute('input_dim')
            elif 'dim' in self.attrs:
                input_dim = self.read_attribute('dim')
            else:
                input_name = self.inputs[0]
                if input_name in dims_by_name:
                    input_dim = dims_by_name[input_name]
                else:
                    kaldi_check(input_name in nodes_by_name,
                                "Cannot find node: %s" % input_name)
                    input_node = nodes_by_name[input_name]
                    input_node.inference_dim(dims_by_name, nodes_by_name)
                    input_dim = input_node.output_dim
            add_log_stddev = False
            block_dim = input_dim
            if 'block_dim' in self.attrs:
                block_dim = self.attrs['block_dim']
            if 'add_log_stddev' in self.attrs:
                add_log_stddev = self.attrs['add_log_stddev']
            output_dim = input_dim
            if add_log_stddev:
                output_dim += int(input_dim // block_dim)
        self.output_dim = output_dim
        dims_by_name[self.name] = output_dim
