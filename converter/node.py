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

from __future__ import print_function
import six

from common import *
from utils import *


def make_node(name, type, inputs, attrs=None, consts=None):
    if type == KaldiOpType.Affine.name:
        return AffineNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Append.name:
        return AppendNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Bias.name:
        return BiasNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Constant.name:
        return ConstantNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Conv1d.name:
        return Conv1dNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Convolution.name:
        return ConvolutionNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Dct.name:
        return DCTNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.DynamicLSTM.name:
        return DynamicLSTMNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.ExtractPooling.name:
        return ExtractPoolingNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.IfDefined.name:
        return IfDefinedNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.LstmNonlinear.name:
        return LSTMNonLinearNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Offset.name:
        return OffsetNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.PadContext.name:
        return PadContextNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.ReplaceIndex.name:
        return ReplaceIndexNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.RestrictedAttention.name:
        return RestrictedAttentionNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Round.name:
        return RoundNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Scales.name:
        return ScalesNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.Splice.name:
        return SpliceNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.StatisticsExtraction.name:
        return StatisticsExtractionNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.StatisticsPooling.name:
        return StatisticsPoolingNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.SumGroup.name:
        return SumGroupNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.TargetRMSNorm.name:
        return TargetRMSNormNode(name, type, inputs, attrs, consts)
    elif type == KaldiOpType.PerEltScale.name:
        return PerEltScaleNode(name, type, inputs, attrs, consts)
    else:
        return Node(name, type, inputs, attrs, consts)


class Node(object):

    def __init__(self, name, type, inputs, attrs=None, consts=None):
        self.name = name
        self.type = type
        self.attrs = attrs
        self.consts = consts
        self.nexts = []
        self.output_shape = None
        self._inputs = inputs
        self._outputs = [name]
        self._input_dim = 0
        self._output_dim = 0
        self._input_indexes = []
        self._output_indexes = []
        self.update_attrs()

        self._start_index = 0
        self._end_index = 0

        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = attrs
        if consts is None:
            self.consts = {}
        else:
            self.consts = consts

    def update_attrs(self):
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
                    context = [ t for t in range(-left_context, right_context + 1)]
                    self.attrs['context'] = context

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, start_index):
        self._start_index = start_index
        self.attrs['start_index'] = start_index

    @property
    def end_index(self):
        return self._end_index

    @end_index.setter
    def end_index(self, end_index):
        self._end_index = end_index
        self.attrs['end_index'] = end_index

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if isinstance(inputs, six.string_types):
            self._inputs = [inputs]
        else:
            self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        if isinstance(outputs, six.string_types):
            self._outputs = [outputs]
        else:
            self._outputs = outputs

    @property
    def input_indexes(self):
        return self._input_indexes

    @input_indexes.setter
    def input_indexes(self, input_indexes):
        self._input_indexes = input_indexes
        self.attrs['input_indexes'] = input_indexes

    @property
    def output_indexes(self):
        return self._output_indexes

    @output_indexes.setter
    def output_indexes(self, output_indexes):
        self._output_indexes = output_indexes
        self.attrs['output_indexes'] = output_indexes

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
        print("name:%s type:%s"
              " inputs:%s,"
              " attrs: %s,"
              " shape: %s,"
              " time_index:[%s : %s]" %
              (self.name,
               self.type,
               self.inputs,
               self.attrs,
               self.output_shape,
               self.start_index,
               self.end_index))

    def infer_shape(self, shapes):
        # This is for the simple case, like NonLinear, Normalize etc.
        if self.inputs[0] in shapes:
            input_shape = shapes[self.inputs[0]]
            batch = input_shape[0]
        else:
            batch = 1
        if 'output_dim' in self.attrs:
            output_dim = self.attrs['output_dim']
        elif 'input_dim' in self.attrs:
            output_dim = self.attrs['input_dim']
        elif 'dim' in self.attrs:
            output_dim = self.attrs['dim']
        else:
            kaldi_check(self.inputs[0] in shapes,
                        "Node(%s)'s input(%s) has no shape." % (self.name, self.inputs[0]))
            input_shape = shapes[self.inputs[0]]
            output_dim = input_shape[-1]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = [batch, output_chunk, output_dim]
        self.output_shape = output_shape

    def precompute(self):
        pass

    def infer_index(self, input_indexes, save_index=False):
        kaldi_check(self.inputs[0] in input_indexes,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        output_indexes = input_indexes[self.inputs[0]]
        if save_index:
            self.start_index = output_indexes[0]
            self.end_index = output_indexes[-1]
            self.attrs['input_time_range'] = [output_indexes[0], output_indexes[-1]]
        return output_indexes

    def map_to_input(self, start, end):
        self.attrs['input_time_range'] = [start, end]
        self.start_index = start
        self.end_index = end


class AffineNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        if 'num_repeats' in self.attrs:
            num_repeats = self.attrs['num_repeats']
        else:
            num_repeats = 1
        weights_name = self.inputs[1]
        kaldi_check(weights_name in shapes,
                    "%s is not found in shapes." % weights_name)
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[0] * num_repeats
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class AppendNode(Node):

    def infer_index(self, indexes_by_name, save_index=False):
        start = -1000
        end = 1000
        for input in self.inputs:
            if '.IfDefined' not in input:
                kaldi_check(input in indexes_by_name,
                            "node(%s)'s input(%s) should be computed before this."
                            % (self.name, input))
                input_indexes = indexes_by_name[input]
                start = input_indexes[0] if input_indexes[0] >= start else start
                end = input_indexes[-1] if input_indexes[-1] <= end else end
        if save_index:
            self.start_index = start
            self.end_index = end
            self.attrs['input_time_range'] = [start, end]
        output_indexes = [t for t in range(start, end + 1)]
        return output_indexes

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape." % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        output_dim = 0
        for input in self.inputs:
            if input in shapes and shapes[input] is not None:
                output_dim += shapes[input][-1]
            else:
                output_dim += input_shape[-1]
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        kaldi_check(output_chunk == input_shape[-2],
                    "Append inputs' chunk size are not match.")
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class BiasNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape." % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        weights_name = self.inputs[1]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[0]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = input_shape[0:-2]
        output_shape.extend([output_chunk, output_dim])
        self.output_shape = output_shape


class ConstantNode(Node):

    def infer_shape(self, shapes):
        weights_name = self.inputs[1]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[0]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = [output_chunk, output_dim]
        self.output_shape = output_shape


class ConstantFunctionNode(Node):

    def infer_shape(self, shapes):
        weights_name = self.inputs[1]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[1]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = [output_chunk, output_dim]
        self.output_shape = output_shape


class Conv1dNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        weights_name = self.inputs[1]
        kaldi_check(weights_name in shapes,
                    "%s is not found in shapes." % weights_name)
        weights_shape = shapes[weights_name]
        num_filters = weights_shape[1]
        patch_stride = self.read_attribute('patch_stride')
        patch_step = self.read_attribute('patch_step')
        patch_dim = self.read_attribute('patch_dim')
        num_patches = 1 + (patch_stride - patch_dim) / patch_step
        output_dim = num_filters * num_patches
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class ConvolutionNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        input_x_dim = self.read_attribute('input_x_dim')
        input_y_dim = self.read_attribute('input_y_dim')
        filt_x_dim = self.read_attribute('filt_x_dim')
        filt_y_dim = self.read_attribute('filt_y_dim')
        filt_x_step = self.read_attribute('filt_x_step')
        filt_y_step = self.read_attribute('filt_y_step')
        num_x_steps = 1 + (input_x_dim - filt_x_dim) / filt_x_step
        num_y_steps = 1 + (input_y_dim - filt_y_dim) / filt_y_step
        filter_shape = shapes[self.inputs[1]]
        num_filters = filter_shape[1]
        output_dim = num_x_steps * num_y_steps * num_filters
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class DCTNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        dct_dim = self.read_attribute('dct_dim')
        if 'keep_dct_dim' not in self.attrs:
            dct_keep_dim = dct_dim
        else:
            dct_keep_dim = self.attrs['keep_dct_dim']

        dim = self.read_attribute('dim')
        assert(dct_dim > 0 and dim > 0 and dim % dct_dim == 0)

        output_dim = dct_keep_dim * (dim / dct_dim)
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class DynamicLSTMNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]

        weights_name = self.inputs[3]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[0]
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class ExtractPoolingNode(Node):

    def infer_index(self, indexes_by_name, save_index=True):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))

        input_indexes = indexes_by_name[self.inputs[0]]

        start, end = input_indexes[0], input_indexes[-1]
        input_period = self.read_attribute('input_period')

        self.attrs['input_time_range'] = [start, end]

        in_start_mod = start % input_period
        if in_start_mod > 0:
            in_start_mod -= input_period
        in_start = start - in_start_mod
        input_indexes = [t for t in range(in_start, end + 1, input_period)]
        kaldi_check(len(input_indexes) > 0,
                    " start time: %s, end time: %s" % (start, end))
        self.input_indexes = input_indexes

        input_start = input_indexes[0]
        input_end = input_indexes[-1]
        output_period = self.read_attribute('output_period')
        mod = input_start % output_period
        start = int(input_start / output_period) * output_period
        if mod < 0:
            start -= output_period
        mod = input_end % output_period
        end = int(input_end / output_period) * output_period
        if mod > 0:
            end += output_period

        left = self.read_attribute('left_context')
        right = self.read_attribute('right_context')
        output_start = start - right
        output_end = end + left
        mod = output_start % output_period
        output_start = int(output_start / output_period) * output_period
        if mod > 0:
            output_start += output_period
        mod = output_end % output_period
        output_end = int(output_end / output_period) * output_period
        if mod < 0:
            output_end -= output_period
        pooling_output_indexes = [t for t in range(output_start, output_end)]

        modulus = self.read_attribute('modulus')
        output_indexes = []
        for index in pooling_output_indexes:
            t_start = index * modulus
            t_end = t_start + modulus
            out_indexes = [t for t in range(t_start, t_end)]
            output_indexes.extend(out_indexes)
        return output_indexes

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        input_dim = self.read_attribute('input_dim')
        include_variance = self.read_attribute('include_variance')
        output_dim = 1 + input_dim
        if include_variance:
            output_dim += input_dim

        output_chunk = self.end_index - self.start_index + 1
        kaldi_check(output_chunk > 0,
                    "output chunk should be greater than zero.")
        num_log_count = self.read_attribute('num_log_count')
        output_dim = output_dim + num_log_count - 1
        output_shape = input_shape[0:-2]
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape

    def precompute(self):
        output_period = self.read_attribute('output_period')
        out_start_mod = self.start_index % output_period
        if out_start_mod > 0:
            out_start = self.start_index - out_start_mod
        else:
            out_start = self.start_index + out_start_mod
        output_indexes = [t for t in range(out_start,
                                           self.end_index + 1,
                                           output_period)]
        output_indexes.sort()
        self.output_indexes = output_indexes
        num_output_indexes = len(self.output_indexes)
        num_input_indexes = len(self.input_indexes)
        kaldi_check(num_output_indexes > 0, "output indexes is empty.")
        kaldi_check(num_input_indexes > 0, "input indexes is empty.")
        forward_indexes = []
        counts = []
        left_context = self.read_attribute('left_context')
        right_context = self.read_attribute('right_context')
        input_period = self.read_attribute('input_period')
        for index in self.output_indexes:
            t_start = index - left_context
            t_end = index + right_context
            forward_index = [-1, -1]
            count = 0.0
            for input_t in range(t_start, t_end, input_period):
                if input_t in self.input_indexes:
                    input_pos = self.input_indexes.index(input_t)
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

    def map_to_input(self, start, end):
        self.start_index = start
        self.end_index = end


class IfDefinedNode(Node):
    def infer_index(self, indexes_by_name, save_index=False):
        if self.inputs[0] in indexes_by_name:
            input_indexes = indexes_by_name[self.inputs[0]]
            output_indexes = input_indexes
        else:
            input_indexes = [0]
            output_indexes = [t for t in range(-1000, 1000)]
        if save_index:
            self.start_index = output_indexes[0]
            self.end_index = output_indexes[-1]
            self.attrs['input_time_range'] = [input_indexes[0],
                                              input_indexes[-1]]
        return output_indexes

    def infer_shape(self, shapes):
        input = self.inputs[0]
        if input in shapes:
            self.output_shape = shapes[input]
        else:
            self.output_shape = None


class LSTMNonLinearNode(Node):
    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        params_shape = shapes[self.inputs[1]]
        cell_dim = params_shape[1]
        output_dim = cell_dim * 2
        self.output_dim = output_dim
        output_chunk = self.end_index - self.start_index + 1
        output_shape = input_shape[0:-2]
        output_shape.extend([output_chunk, output_dim])
        self.output_shape = output_shape


class OffsetNode(Node):

    def infer_index(self, indexes_by_name, save_index=False):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        input_indexes = indexes_by_name[self.inputs[0]]
        start = input_indexes[0]
        end = input_indexes[-1]
        offset = self.read_attribute('offset')
        new_start = start + offset
        new_end = end + offset
        if new_start <= start:
            new_start = start
        if new_end >= end:
            new_end = end
        output_indexes = [t for t in range(new_start, new_end + 1)]
        if save_index:
            self.start_index = output_indexes[0]
            self.end_index = output_indexes[-1]
            self.attrs['input_time_range'] = [start, end]
        return output_indexes

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = []
        output_shape.extend(input_shape)
        output_shape[-2] = output_chunk
        self.output_shape = output_shape

    def map_to_input(self, start, end):
        offset = self.read_attribute('offset')
        self.attrs['input_time_range'] = [start - offset, end - offset]
        self.start_index = start
        self.end_index = end


class PadContextNode(Node):

    def infer_index(self, indexes_by_name, save_index=False):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        input_index = indexes_by_name[self.inputs[0]]
        start, end = input_index[0], input_index[-1]

        left = self.read_attribute('left_context')
        right = self.read_attribute('right_context')
        start_index = start - left
        end_index = end + right
        if save_index:
            self.start_index = start_index
            self.end_index = end_index
            self.attrs['input_time_range'] = [start, end]

        output_indexes = [t for t in range(start_index, end_index + 1)]
        return output_indexes

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        chunk_size = input_shape[-2]
        left_context = self.read_attribute('left_context')
        right_context = self.read_attribute('right_context')
        out_chunk_size = chunk_size + left_context + right_context
        output_shape = [n for n in input_shape]
        output_shape[-2] = out_chunk_size
        self.output_shape = output_shape


class PerEltScaleNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        weights_name = self.inputs[1]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[1]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = input_shape[0:-2]
        output_shape.extend([output_chunk, output_dim])
        self.output_shape = output_shape


class PermuteNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        weights_name = self.inputs[1]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[0]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = input_shape[0:-2]
        output_shape.extend([output_chunk, output_dim])
        self.output_shape = output_shape


class ReplaceIndexNode(Node):

    def infer_index(self, indexes_by_name, save_index=False):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        input_indexes = indexes_by_name[self.inputs[0]]
        [start, end] = [input_indexes[0], input_indexes[-1]]
        if save_index:
            self.attrs['input_time_range'] = [start, end]
            self.start_index = -1000
            self.end_index = 1000

        output_indexes = [t for t in range(-1000, 1000)]
        return output_indexes


class RestrictedAttentionNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        num_heads = self.read_attribute('num_heads')
        value_dim = self.read_attribute('value_dim')
        num_left_inputs = self.read_attribute('num_left_inputs')
        num_right_inputs = self.read_attribute('num_right_inputs')
        output_context = self.read_attribute('output_context')
        context_dim = num_left_inputs + 1 + num_right_inputs
        self.set_attribute('context_dim', context_dim)
        output_dim = num_heads * (value_dim + context_dim)\
            if output_context else num_heads * value_dim
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class RoundNode(Node):

    def infer_index(self, indexes_by_name, save_index=False):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        input_indexes = indexes_by_name[self.inputs[0]]
        modulus = self.read_attribute('modulus')
        output_indexes = []
        for index in input_indexes:
            t_start = index * modulus
            t_end = t_start + modulus
            out_indexes = [t for t in range(t_start, t_end)]
            output_indexes.extend(out_indexes)
        if save_index:
            self.start_index = output_indexes[0]
            self.end_index = output_indexes[-1]
            self.attrs['input_time_range'] = [input_indexes[0],
                                              input_indexes[-1]]
        return output_indexes


class ScalesNode(Node):

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        weights_name = self.inputs[1]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[0]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = input_shape[0:-2]
        output_shape.extend([output_chunk, output_dim])
        self.output_shape = output_shape


class SpliceNode(Node):
    def infer_index(self, indexes_by_name, save_index=False):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        input_indexes = indexes_by_name[self.inputs[0]]
        context = self.read_attribute('context')
        left = context[0]
        right = context[-1]
        input_start = input_indexes[0]
        input_end = input_indexes[-1]
        output_start = input_start - left
        output_end = input_end - right
        if save_index:
            self.start_index = output_start
            self.end_index = output_end
            self.attrs['input_time_range'] = [input_start, input_end]
        output_indexes = [t for t in range(output_start, output_end + 1)]
        return output_indexes

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        input_dim = input_shape[-1]

        if 'const_component_dim' in self.attrs:
            const_component_dim = self.attrs['const_component_dim']
        else:
            const_component_dim = 0
        context = self.read_attribute('context')

        output_dim = (input_dim - const_component_dim) * len(context)\
                     + const_component_dim
        output_chunk = self.end_index - self.start_index + 1
        output_shape = input_shape[0:-2]
        output_shape.extend([output_chunk, output_dim])
        self.output_shape = output_shape

    def map_to_input(self, start, end):
        context = self.read_attribute('context')
        left = context[0]
        right = context[-1]
        self.attrs['input_time_range'] = [start + left, end + right]
        self.start_index = start
        self.end_index = end

class StatisticsExtractionNode(Node):
    def infer_index(self, indexes_by_name, save_index=False):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        input_indexes = indexes_by_name[self.inputs[0]]
        input_start = input_indexes[0]
        input_end = input_indexes[-1]
        output_period = self.read_attribute('output_period')
        mod = input_start % output_period
        output_start = int(input_start / output_period) * output_period
        if mod < 0:
            output_start -= output_period
        mod = input_end % output_period
        output_end = int(input_end / output_period) * output_period
        if mod > 0:
            output_end += output_period
        if save_index:
            self.start_index = output_start
            self.end_index = output_end
            self.attrs['input_time_range'] = [input_start, input_end]
        output_indexes = [t for t in range(output_start,
                                           output_end + 1,
                                           output_period)]
        return output_indexes

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        input_dim = self.read_attribute('input_dim')
        include_variance = self.read_attribute('include_variance')
        output_dim = 1 + input_dim
        if include_variance:
            output_dim += input_dim
        output_shape = input_shape[0:-2]
        num_output_indexes = len(self.output_indexes)
        kaldi_check(num_output_indexes > 0,
                    "output_indexes should be greater than zero.")
        output_shape.append(num_output_indexes)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class StatisticsPoolingNode(Node):
    def infer_index(self, indexes_by_name, save_index=False):
        kaldi_check(self.inputs[0] in indexes_by_name,
                    "node(%s)'s input(%s) should be computed before this."
                    % (self.name, self.inputs[0]))
        input_indexes = indexes_by_name[self.inputs[0]]
        input_start = input_indexes[0]
        input_end = input_indexes[-1]
        input_period = self.read_attribute('input_period')
        left = self.read_attribute('left_context')
        right = self.read_attribute('right_context')
        output_start = input_start - right
        output_end = input_end + left
        mod = output_start % input_period
        output_start = int(output_start / input_period) * input_period
        if mod > 0:
            output_start += input_period
        mod = output_end % input_period
        output_end = int(output_end / input_period) * input_period
        if mod < 0:
            output_end -= input_period
        output_indexes = [t for t in range(output_start, output_end + 1)]
        if save_index:
            self.start_index = output_start
            self.end_index = output_end
            self.output_indexes = output_indexes
            self.attrs['input_time_range'] = [input_start, input_end]
        return output_indexes

    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        input_dim = self.read_attribute('input_dim')
        num_log_count_features = self.read_attribute('num_log_count_features')
        output_dim = input_dim + num_log_count_features - 1
        output_shape = input_shape[0:-2]
        num_output_indexes = len(self.output_indexes)
        kaldi_check(num_output_indexes > 0,
                    "output_indexes should be greater than zero.")
        output_shape.append(num_output_indexes)
        output_shape.append(output_dim)
        self.output_shape = output_shape


class SumGroupNode(Node):
    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        weights_name = self.inputs[1]
        weights_shape = shapes[weights_name]
        output_dim = weights_shape[0]
        output_chunk = self.end_index - self.start_index + 1
        output_shape = input_shape[0:-2]
        output_shape.extend([output_chunk, output_dim])
        self.output_shape = output_shape

class TargetRMSNormNode(Node):
    def infer_shape(self, shapes):
        kaldi_check(self.inputs[0] in shapes,
                    "Node(%s)'s input(%s) has no shape."
                    % (self.name, self.inputs[0]))
        input_shape = shapes[self.inputs[0]]
        if 'input_dim' in self.attrs:
            input_dim = self.read_attribute('input_dim')
        elif 'dim' in self.attrs:
            input_dim = self.read_attribute('dim')
        else:
            input_dim = input_shape[-1]
        add_log_stddev = False
        block_dim = input_dim
        if 'block_dim' in self.attrs:
            block_dim = self.attrs['block_dim']
        if 'add_log_stddev' in self.attrs:
            add_log_stddev = self.attrs['add_log_stddev']

        output_dim = input_dim
        if add_log_stddev:
            output_dim += int(input_dim / block_dim)
        kaldi_check(input_shape[-1] == input_dim,
                    "actual input dim should be equal to 'input dim'.")
        output_shape = input_shape[0:-2]
        output_chunk = self.end_index - self.start_index + 1
        output_shape.append(output_chunk)
        output_shape.append(output_dim)
        self.output_shape = output_shape
