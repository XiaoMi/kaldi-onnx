#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/27
"""Nnet3 node."""
import logging

import six

from converter.common import KaldiOpType
from converter.utils import kaldi_check

_LOG = logging.getLogger(__name__)


def make_node(name, node_type, inputs, outputs, attrs=None, consts=None):
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


class SubsampleNode(Node):

    def precompute(self):
        forward_indexes = list()
        for idx in self.output_indexes:
            kaldi_check(idx in self.input_indexes,
                        "%s cannot compute index: %s" % (self.name, idx))
            pos = self.input_indexes.index(idx)
            forward_indexes.append(pos)
        self.attrs['forward_indexes'] = forward_indexes
