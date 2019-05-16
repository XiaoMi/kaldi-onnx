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
python parser.py --input=path/to/kaldi_model.mdl --nnet-type=(2 or 3)
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from common import *
from utils import *
import re
import sys


class Nnet2Parser(object):

    def __init__(self, line_buffer):

        self._additive_noise_actions = {
            '<StdDev>': (read_float, 'std_dev'),
        }

        self._affine_actions = {
            '<LinearParams>': (read_matrix, 'params'),
            '<BiasParams>': (read_vector, 'bias'),
            '<Alpha>': (read_float, 'alpha'),
            '<NumRepeats>': (read_int, 'num_repeats'),
            '<NumBlocks>': (read_int, 'num_blocks'),
        }

        self._basic_actions = {
            '<Dim>': (read_int, 'dim'),
            '<InputDim>': (read_int, 'input_dim'),
            '<OutputDim>': (read_int, 'output_dim')
        }

        self._bias_actions = {
            '<Bias>': (read_vector, 'bias'),
        }

        self._convolutional1d_actions = {
            '<PatchDim>': (read_int, 'patch_dim'),
            '<PatchStep>': (read_int, 'patch_step'),
            '<PatchStride>': (read_int, 'patch_stride'),
            '<AppendedConv>': (read_bool, 'appended_conv'),
            '<FilterParams>': (read_matrix, 'params'),
            '<BiasParams>': (read_vector, 'bias'),
        }

        self._dct_actions = {
            '<DctDim>': (read_int, 'dct_dim'),
            '<Reorder>': (read_bool, 'reorder'),
            '<DctKeepDim>': (read_bool, 'dct_keep_dim'),
        }

        self._dropout_actions = {
            '<DropoutScale>': (read_float, 'dropout_scale'),
            '<DropoutProportion>': (read_float, 'dropout_proportion'),
        }

        self._linear_actions = {
            '<CuMatrix>': (read_matrix, 'params'),
        }

        self._maxpooling_actions = {
            '<PoolSize>': (read_int, 'pool_size'),
            '<PoolStride>': (read_int, 'pool_stride'),
        }

        self._no_actions = {}

        self._nonlinear_actions = {
            '<ValueSum>': (read_vector, 'value_avg'),
            '<DerivSum>': (read_vector, 'deriv_avg'),
            '<Count>': (read_float, 'count'),
        }

        self._permute_actions = {
            '<Reorder>': (read_vector, 'reorder'),
        }

        self._pnorm_actions = {
            '<P>': (read_float, 'p'),
        }

        self._power_actions = {
            '<Power>': (read_float, 'power'),
        }

        self._fixed_scale_actions = {
            '<Scales>': (read_vector, 'scales'),
        }

        self._scale_actions = {
            '<Scale>': (read_float, 'scale'),
        }

        self._splice_actions = {
            '<LeftContext>': (read_int, 'left_context'),
            '<RightContext>': (read_int, 'right_context'),
            '<Context>': (read_vector_int, 'context'),
            '<ConstComponentDim>': (read_int, 'const_component_dim'),
        }

        self._sumgroup_actions = {
            '<Sizes>': (read_vector_int, 'sizes'),
        }

        self._component_actions = {
            NNet2Component.AdditiveNoiseComponent.name: self._additive_noise_actions,
            NNet2Component.AffineComponent.name: self._affine_actions,
            NNet2Component.AffineComponentPreconditioned.name: self._affine_actions,
            NNet2Component.AffineComponentPreconditionedOnline.name: self._affine_actions,
            NNet2Component.BlockAffineComponent.name: self._affine_actions,
            NNet2Component.BlockAffineComponentPreconditioned.name: self._affine_actions,
            NNet2Component.Convolutional1dComponent.name: self._convolutional1d_actions,
            NNet2Component.DctComponent.name: self._dct_actions,
            NNet2Component.DropoutComponent.name: self._dropout_actions,
            NNet2Component.FixedAffineComponent.name: self._affine_actions,
            NNet2Component.FixedBiasComponent.name: self._bias_actions,
            NNet2Component.FixedLinearComponent.name: self._linear_actions,
            NNet2Component.FixedScaleComponent.name: self._fixed_scale_actions,
            NNet2Component.LogSoftmaxComponent.name: self._nonlinear_actions,
            NNet2Component.MaxoutComponent.name: self._no_actions,
            NNet2Component.MaxpoolingComponent.name: self._maxpooling_actions,
            NNet2Component.NonlinearComponent.name: self._nonlinear_actions,
            NNet2Component.NormalizeComponent.name: self._nonlinear_actions,
            NNet2Component.PermuteComponent.name: self._permute_actions,
            NNet2Component.PnormComponent.name: self._pnorm_actions,
            NNet2Component.PowerComponent.name: self._power_actions,
            NNet2Component.RectifiedLinearComponent.name: self._nonlinear_actions,
            NNet2Component.ScaleComponent.name: self._scale_actions,
            NNet2Component.SigmoidComponent.name: self._nonlinear_actions,
            NNet2Component.SoftHingeComponent.name: self._nonlinear_actions,
            NNet2Component.SoftmaxComponent.name: self._nonlinear_actions,
            NNet2Component.SpliceComponent.name: self._splice_actions,
            NNet2Component.SpliceMaxComponent.name: self._splice_actions,
            NNet2Component.SumGroupComponent.name: self._sumgroup_actions,
            NNet2Component.TanhComponent.name: self._nonlinear_actions,
        }

        self._current_component_id = 0
        self._components = []
        self._nodes = []
        self._num_components = 0
        self._line_buffer = line_buffer
        # self._transition_model = []

    def run(self):
        line = next(self._line_buffer)
        pos = 0
        if line.startswith("<TransitionModel>"):
            self.parse_transition_model(line)
            line = next(self._line_buffer)
        assert line.startswith(NNet2Header)
        line, pos = self.check_header(line, pos)
        self.parse_component_lines(line, pos)
        self.add_input_node()
        print("finished parse nnet2 (%s) components." %
              len(self._components))
        return self._components

    def add_input_node(self):
        first_component = self._components[0]
        inputs = first_component['input']
        if 'input_dim' in first_component:
            input_dim = first_component['input_dim']
        else:
            assert 'dim' in first_component,\
                   "'input dim' or 'dim' attribute is required" \
                   " in the first node: %s." % first_component
            input_dim = first_component['dim']

        input_component = {'type': 'Input',
                           'name': inputs[0],
                           'dim': input_dim}
        self._components.insert(0, input_component)

    def parse_component_lines(self, line, pos):
        tok, pos = read_next_token(line, pos)
        assert(tok == "<Components>")
        while True:
            tok, pos = read_next_token(line, pos)
            if tok == "</Components>":
                break
            elif tok == NNet2End:
                print("finished parse nnet2 components.")
                break
            elif tok[1:-1] in self._component_actions:
                component_type = tok[1:-1]
                component, line, pos =\
                    self.parse_component(component_type, line, pos)
                self._components.append(component)
                line = next(self._line_buffer)
                pos = 0
            else:
                raise Exception(
                    "{0}: error reading Component: at position {1},"
                    "unrecognised component: {2}."
                    .format(sys.argv[0], pos, tok))
        return line, pos

    def parse_component(self, component_type, line, pos):
        component_id = self._current_component_id
        terminating_token = "</" + component_type + ">"
        terminating_tokens = {terminating_token}
        action_dict = self._component_actions[component_type]
        action_dict.update(self._basic_actions)
        op_dict, pos = read_generic(line,
                                    pos,
                                    self._line_buffer,
                                    terminating_tokens,
                                    action_dict)
        op_dict['name'] = str(component_id + 1)
        op_dict['input'] = [str(component_id)]
        op_dict['type'] = KaldiOpRawType[component_type]
        self._current_component_id += 1
        return op_dict, line, pos

    def parse_transition_model(self, line):
        # self._transition_model.append(line)
        while True:
            line = next(self._line_buffer)
            # self._transition_model.append(line)
            if line.startswith("</TransitionModel>"):
                break

    # parse nnet header and get components number
    def check_header(self, line, pos):
        tok, pos = read_next_token(line, pos)
        assert(tok == NNet2Header)
        tok, pos = read_next_token(line, pos)
        assert(tok == '<NumComponents>')
        num_components, pos = read_int(line, pos, self._line_buffer)
        self._num_components = num_components
        return line, pos

    def parse_end_of_component(self, component, line):
        end_component = "</" + component + ">"
        while end_component not in line:
            line = next(self._line_buffer)

    @staticmethod
    def parse_priors(line):
        vector = Nnet2Parser.parse_vector(line.partition('[')[2])
        return vector

    @staticmethod
    def parse_vector(line):
        vector = line.strip().strip("[]")
        return np.array([float(x) for x in vector.split()], dtype="float32")


class Nnet3Parser(object):

    def __init__(self, line_buffer):

        self._affine_actions = {
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

        self._bachnorm_actions = {
            '<Dim>': (read_int, 'dim'),
            '<BlockDim>': (read_int, 'block_dim'),
            '<Epsilon>': (read_float, 'epsilon'),
            '<TargetRms>': (read_float, 'target_rms'),
            '<Count>': (read_float, 'count'),
            '<StatsMean>': (read_vector, 'stats_mean'),
            '<StatsVar>': (read_vector, 'stats_var'),
            '<TestMode>': (read_bool, 'test_mode'),
        }

        self._backprop_truncation_actions = {
            '<Dim>': (read_int, 'dim'),
            '<Scale>': (read_float, 'scale'),
        }

        self._basic_actions = {
            '<Dim>': (read_int, 'dim'),
            '<InputDim>': (read_int, 'input_dim'),
            '<OutputDim>': (read_int, 'output_dim')
        }

        self._bias_actions = {
            '<Bias>': (read_vector, 'bias'),
        }

        self._constant_actions = {
            '<InputDim>': (read_int, 'input_dim'),
            '<Output>': (read_vector, 'constants'),
        }

        self._convolution_actions = {
            '<InputXDim>': (read_int, 'input_x_dim'),
            '<InputYDim>': (read_int, 'input_y_dim'),
            '<InputZDim>': (read_int, 'input_z_dim'),
            '<FiltXDim>': (read_int, 'filt_x_dim'),
            '<FiltYDim>': (read_int, 'filt_y_dim'),
            '<FiltZDim>': (read_int, 'filt_z_dim'),
            '<FiltXStep>': (read_int, 'filt_x_step'),
            '<FiltYStep>': (read_int, 'filt_y_step'),
            '<InputVectorization>': (read_int, 'input_vectorization'),
            '<FilterParams>': (read_matrix, 'params'),
            '<BiasParams>': (read_vector, 'bias'),
        }

        self._linear_actions = {
            '<Params>': (read_matrix, 'params'),
            '<RankInOut>': (read_int, 'rank_inout'),
            '<UpdatedPeriod>': (read_int, 'updated_period'),
            '<NumSamplesHistory>': (read_float, 'num_samples_history'),
            '<Alpha>': (read_float, 'alpha')
        }

        self._lstm_actions = {
            '<Params>': (read_matrix, 'params'),
            '<ValueAvg>': (read_matrix, 'value_avg'),
            '<DerivAvg>': (read_matrix, 'deriv_avg'),
            '<SelfRepairConfig>': (read_vector, 'self_repair_config'),
            '<SelfRepairProb>': (read_vector, 'self_repair_prob'),
            '<Count>': (read_float, 'count')
        }

        self._maxpooling_actions = {
            '<InputXDim>': (read_int, 'input_x_dim'),
            '<InputYDim>': (read_int, 'input_y_dim'),
            '<InputZDim>': (read_int, 'input_z_dim'),
            '<PoolXSize>': (read_int, 'pool_x_size'),
            '<PoolYSize>': (read_int, 'pool_y_size'),
            '<PoolZSize>': (read_int, 'pool_z_size'),
            '<PoolXStep>': (read_int, 'pool_x_step'),
            '<PoolYStep>': (read_int, 'pool_y_step'),
            '<PoolZStep>': (read_int, 'pool_z_step'),
        }

        self._no_actions = {}

        self._nonlinear_actions = {
            '<Dim>': (read_int, 'dim'),
            '<BlockDim>': (read_int, 'block_dim'),
            '<ValueAvg>': (read_vector, 'value_avg'),
            '<DerivAvg>': (read_vector, 'deriv_avg'),
            '<OderivRms>': (read_vector, 'oderiv_rms'),
            '<Count>': (read_float, 'count'),
            '<OderivCount>': (read_float, 'oderiv_count')
        }

        self._normalize_actions = {
            '<Dim>': (read_int, 'dim'),
            '<InputDim>': (read_int, 'input_dim'),
            '<BlockDim>': (read_int, 'block_dim'),
            '<TargetRms>': (read_float, 'target_rms'),
            '<AddLogStddev>': (read_bool, 'add_log_stddev')
        }

        self._per_elt_offset_actions = {
            '<Dim>': (read_int, 'dim'),
            '<Offsets>': (read_vector_int, 'offsets'),
        }

        self._per_elt_scale_actions = {
            '<Params>': (read_matrix, 'params'),
            '<Rank>': (read_int, 'rank'),
            '<UpdatedPeriod>': (read_int, 'updated_period'),
            '<NumSamplesHistory>': (read_float, 'num_samples_history'),
            '<Alpha>': (read_float, 'alpha'),
        }

        self._permute_actions = {
            '<ColumnMap>': (read_vector, 'column_map'),
        }

        self._restricted_attention_actions = {
            '<NumHeads>': (read_int, 'num_heads'),
            '<KeyDim>': (read_int, 'key_dim'),
            '<ValueDim>': (read_int, 'value_dim'),
            '<NumLeftInputs>': (read_int, 'num_left_inputs'),
            '<NumRightInputs>': (read_int, 'num_right_inputs'),
            '<TimeStrides>': (read_int, 'time_stride'),
            '<NumLeftInputsRequired>':
                (read_int, 'num_left_inputs_required'),
            '<NumRightInputsRequired>':
                (read_int, 'num_right_inputs_required'),
            '<OutputContext>': (read_bool, 'output_context'),
            '<KeyScale>': (read_float, 'key_scale'),
            '<StatsCount>': (read_float, 'stats_count'),
            '<EntropyStats>': (read_vector, 'entropy_stats'),
            '<PosteriorStats>': (read_matrix, 'posterior_stats'),
        }

        self._scale_actions = {
            '<Scales>': (read_vector, 'scales'),
        }

        self._scale_offset_actions = {
            '<Dim>': (read_int, 'dim'),
            '<Scales>': (read_vector, 'scales'),
            '<Offsets>': (read_vector, 'offsets'),
            '<Rank>': (read_int, 'rank'),
            '<UseNaturalGradient>': (read_bool, 'use_natural_gradient'),
        }

        self._statistics_extraction_actions = {
            '<InputDim>': (read_int, 'input_dim'),
            '<InputPeriod>': (read_int, 'input_period'),
            '<OutputPeriod>': (read_int, 'output_period'),
            '<IncludeVarinance>': (read_bool, 'include_variance'),
        }

        self._statistics_pooling_actions = {
            '<InputDim>': (read_int, 'input_dim'),
            '<InputPeriod>': (read_int, 'input_period'),
            '<LeftContext>': (read_int, 'left_context'),
            '<RightContext>': (read_int, 'right_context'),
            '<NumLogCountFeatures>': (read_int, 'num_log_count_features'),
            '<VarianceFloor>': (read_float, 'variance_floor'),
            '<OutputStddevs>': (read_bool, 'output_stddevs'),
        }

        self._sumblock_actions = {
            '<InputDim>': (read_int, 'input_dim'),
            '<OutputDim>': (read_int, 'output_dim'),
            '<Scale>': (read_float, 'scale'),
        }

        self._sumgroup_actions = {
            '<InputDim>': (read_int, 'input_dim'),
            '<OutputDim>': (read_int, 'output_dim'),
            '<Sizes>': (read_vector_int, 'sizes'),
        }

        self._tdnn_actions = {
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

        self._sumblock_actions = {
            '<InputDim>': (read_int, 'input_dim'),
            '<OutputDim>': (read_int, 'output_dim'),
            '<Scale>': (read_float, 'scale'),
        }

        self._sumgroup_actions = {
            '<InputDim>': (read_int, 'input_dim'),
            '<OutputDim>': (read_int, 'output_dim'),
            '<Sizes>': (read_vector_int, 'sizes'),
        }

        self._tdnn_actions = {
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

        self._timeheight_conv_actions = {
            '<LinearParams>': (read_matrix, 'params'),
            '<BiasParams>': (read_vector, 'bias'),
            '<MaxMemoryMb>': (read_float, 'max_memory_mb'),
            '<NumMinibatchesHistory>': (read_float, 'num_minibatches_history'),
            '<OrthonormalConstraint>': (read_float, 'orthonormal_constraint'),
            '<UseNaturalGradient>': (read_bool, 'use_natrual_gradient'),
            '<RankInOut>': (read_int, 'rank_inout'),
            '<NumSamplesHistory>': (read_float, 'num_samples_history'),
            '<Alpha>': (read_float, 'alpha'),
            '<AlphaInOut>': (read_float, 'alpha_inout'),
        }

        self._component_parsers = {
            NNet3Component.AffineComponent.name: self._affine_actions,
            NNet3Component.BackpropTruncationComponent.name:
                self._backprop_truncation_actions,
            NNet3Component.BatchNormComponent.name: self._bachnorm_actions,
            NNet3Component.BlockAffineComponent.name: self._affine_actions,
            NNet3Component.BlockAffineComponentPreconditioned.name:
                self._affine_actions,
            NNet3Component.ClipGradientComponent.name: self._basic_actions,
            NNet3Component.ConstantComponent.name: self._constant_actions,
            NNet3Component.ConvolutionComponent.name:
                self._convolution_actions,
            NNet3Component.DistributeComponent.name: self._basic_actions,
            NNet3Component.DropoutComponent.name: self._basic_actions,
            NNet3Component.DropoutMaskComponent.name: self._basic_actions,
            NNet3Component.ElementwiseProductComponent.name:
                self._basic_actions,
            NNet3Component.FixedAffineComponent.name: self._affine_actions,
            NNet3Component.FixedBiasComponent.name: self._bias_actions,
            NNet3Component.FixedScaleComponent.name: self._scale_actions,
            NNet3Component.GeneralDropoutComponent.name: self._basic_actions,
            NNet3Component.LinearComponent.name: self._linear_actions,
            NNet3Component.LogSoftmaxComponent.name: self._nonlinear_actions,
            NNet3Component.LstmNonlinearityComponent.name: self._lstm_actions,
            NNet3Component.MaxpoolingComponent.name: self._maxpooling_actions,
            NNet3Component.NaturalGradientAffineComponent.name:
                self._affine_actions,
            NNet3Component.NaturalGradientPerElementScaleComponent.name:
                self._per_elt_scale_actions,
            NNet3Component.NaturalGradientRepeatedAffineComponent.name:
                self._affine_actions,
            NNet3Component.NonlinearComponent.name: self._nonlinear_actions,
            NNet3Component.NoOpComponent.name: self._basic_actions,
            NNet3Component.NormalizeComponent.name: self._normalize_actions,
            NNet3Component.PerElementOffsetComponent.name:
                self._per_elt_offset_actions,
            NNet3Component.PerElementScaleComponent.name:
                self._per_elt_scale_actions,
            NNet3Component.PermuteComponent.name: self._permute_actions,
            NNet3Component.PnormComponent.name: self._basic_actions,
            NNet3Component.RandomComponent.name: self._basic_actions,
            NNet3Component.RectifiedLinearComponent.name:
                self._nonlinear_actions,
            NNet3Component.RepeatedAffineComponent.name:
                self._affine_actions,
            NNet3Component.RestrictedAttentionComponent.name:
                self._restricted_attention_actions,
            NNet3Component.ScaleAndOffsetComponent.name:
                self._scale_offset_actions,
            NNet3Component.SigmoidComponent.name:
                self._nonlinear_actions,
            NNet3Component.SoftmaxComponent.name:
                self._nonlinear_actions,
            NNet3Component.StatisticsExtractionComponent.name:
                self._statistics_extraction_actions,
            NNet3Component.StatisticsPoolingComponent.name:
                self._statistics_pooling_actions,
            NNet3Component.SumBlockComponent.name: self._sumblock_actions,
            NNet3Component.SumGroupComponent.name: self._sumgroup_actions,
            NNet3Component.TanhComponent.name: self._nonlinear_actions,
            NNet3Component.TdnnComponent.name: self._tdnn_actions,
            NNet3Component.TimeHeightConvolutionComponent.name:
                self._timeheight_conv_actions,
        }

        # self._configs = []
        self._components_by_name = dict()
        self._component_names = []
        self._components = []
        self._num_components = 0
        self._line_buffer = line_buffer
        self._pos = 0
        self._current_id = 0

    def run(self):
        self.check_header()
        self.parse_configs()
        self.parse_component_lines()
        self._components = []
        for component_name in self._components_by_name:
            self._components.append(self._components_by_name[component_name])
        return self._components

    def print_components_info(self):
        for component in self._components:
            print(component)

    def check_header(self):
        line = next(self._line_buffer)
        assert line.startswith(NNet3Header)

    def parse_configs(self):
        while True:
            line = next(self._line_buffer, 'Parser_EOF')
            if line == 'Parser_EOF':
                raise Exception('No <NumComponents> in File.')
            if line.startswith("<NumComponents>"):
                self._num_components = int(line.split()[1])
                break
            config_type, parsed_config = self.parse_nnet3_config(line)
            if config_type is not None:
                parsed_config['node_type'] = config_type

                if 'input' in parsed_config:
                    input = parsed_config['input']
                    parsed_input = self.parse_input_descriptor(input)
                    if isinstance(parsed_input, list):
                        parsed_config['input'] = parsed_input
                    else:
                        parsed_config['input'] = [parsed_input]

                if config_type in ['output-node',
                                   'input-node',
                                   'dim-range-node']:
                    parsed_config['type'] = KaldiOpRawType[config_type]
                    kaldi_check('name' in parsed_config,
                                "Expect 'name' value in %s" % parsed_config)
                    if 'input-node' in parsed_config:
                        parsed_config['input'] = [parsed_config['input-node']]
                    if 'dim-offset' in parsed_config:
                        parsed_config['offset'] = parsed_config['dim-offset']
                self._current_id += 1
                parsed_config['id'] = self._current_id
                self.add_component(parsed_config)

    def add_component(self, component):
        kaldi_check('name' in component or 'component' in component,
                    "'name' or 'component' is required.")
        if 'component' in component:
            component_name = component['component']
        else:
            component_name = component['name']
        self._components.append(component)
        self._components_by_name[component_name] = component
        self._component_names.append(component['name'])

    def parse_input_descriptor(self, input_str):
        sub_components = []
        input_str = input_str.replace(' ', '')
        type = self.check_sub_inputs(input_str)
        if type is not None:
            input_name = self.parse_descriptor(type, input_str, sub_components)
            for item in sub_components:
                self.add_component(item)
        else:
            input_name = input_str
        return input_name

    @staticmethod
    def check_sub_inputs(input_str):
        try:
            type_end_index = input_str.index('(')
            if type_end_index > 0:
                type = input_str[0: type_end_index]
                return type
            else:
                return None
        except ValueError:
            return None

    def parse_descriptor(self, type, input_str, sub_components):
        input = input_str[len(type) + 1: -1]
        if type == NNet3Descriptor.Offset.name:
            return self.parse_offset_descp(input, sub_components)
        elif type == NNet3Descriptor.Round.name:
            return self.parse_round_descp(input, sub_components)
        elif type == NNet3Descriptor.Switch.name:
            return self.parse_switch_descp(input, sub_components)
        elif type == NNet3Descriptor.Sum.name:
            return self.parse_sum_descp(input, sub_components)
        elif type == NNet3Descriptor.Failover.name:
            return self.parse_failover_descp(input, sub_components)
        elif type == NNet3Descriptor.IfDefined.name:
            return self.parse_ifdefine_descp(input, sub_components)
        elif type == NNet3Descriptor.Scale.name:
            return self.parse_scale_descp(input, sub_components)
        elif type == NNet3Descriptor.Const.name:
            return self.parse_const_descp(input, sub_components)
        elif type == NNet3Descriptor.ReplaceIndex.name:
            return self.parse_replace_index_descp(input, sub_components)
        elif type == NNet3Descriptor.Append.name:
            return self.parse_append_descp(input, sub_components)
        else:
            raise Exception(
                'Does not support this descriptor type: {0} in input: {1}'
                    .format(type, input_str))

    def parse_append_descp(self, input_str, sub_components):
        input_str = input_str.replace(" ", "")
        items = parenthesis_split(input_str, ",")
        num_inputs = len(items)
        kaldi_check(num_inputs >= 2,
                    "Append should have at least two inputs.")
        append_inputs = []
        offset_components = []
        offset_inputs = []
        offset_indexes = []
        offsets = []
        for item in items:
            type = self.check_sub_inputs(item)
            if type in NNet3Descriptors:
                sub_comp_name = self.parse_descriptor(type, item, sub_components)
                sub_comp = sub_components[-1]
                append_inputs.append(sub_comp_name)
                if type == NNet3Descriptor.Offset.name:
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
        splice_indexes = splice_continous_numbers(offset_indexes)

        if num_inputs == len(offset_inputs) and len(pure_inputs) == 1:
            # print("Fuse Append to Splice 1.")
            self._current_id += 1
            comp_name = 'splice_' + str(self._current_id)
            component = {
                'id': self._current_id,
                'type': 'Splice',
                'name': comp_name,
                'input': pure_inputs,
                'context': offsets}
            for item in offset_components:
                sub_components.remove(item)
            sub_components.append(component)
        else:
            if len(pure_inputs) == 1 and len(splice_indexes) == 1 and\
                    len(offset_inputs) > 1:
                # print("Fuse Append to Splice 2.")
                self._current_id += 1
                splice_comp_name = 'splice_' + str(self._current_id)
                splice_component = {
                    'id': self._current_id,
                    'type': 'Splice',
                    'name': splice_comp_name,
                    'context': offsets,
                    'input': pure_inputs}
                new_append_inputs = []
                for i in range(num_inputs):
                    if i not in offset_indexes:
                        new_append_inputs.append(append_inputs[i])
                    elif i == offset_indexes[0] :
                        new_append_inputs.append(splice_comp_name)
                append_inputs = new_append_inputs
                for item in offset_components:
                    sub_components.remove(item)
                sub_components.append(splice_component)
            self._current_id += 1
            comp_name = 'append_' + str(self._current_id)
            component = {
                'id': self._current_id,
                'type': 'Append',
                'name': comp_name,
                'input': append_inputs}
            sub_components.append(component)
        return comp_name

    def parse_offset_descp(self, input, sub_components):
        items = parenthesis_split(input, ",")
        kaldi_check(len(items) == 2, 'Offset descriptor should have 2 items.')
        sub_type = self.check_sub_inputs(items[0])
        if sub_type is not None:
            input_name = self.parse_descriptor(sub_type,
                                               items[0],
                                               sub_components)
        else:
            input_name = items[0]
        offset = int(items[1])
        self._current_id += 1
        comp_name = input_name + '.Offset.' + str(offset)
        component = {
            'id': self._current_id,
            'type': 'Offset',
            'name': comp_name,
            'input': [input_name],
            'offset': offset
        }
        sub_components.append(component)
        return comp_name

    def parse_scale_descp(self, input, sub_components):
        items = parenthesis_split(input, ",")
        kaldi_check(len(items) == 2, 'Scale descriptor should have 2 items.')
        scale = float(items[0])
        sub_type = self.check_sub_inputs(items[1])
        if sub_type is not None:
            input_name = self.parse_descriptor(sub_type,
                                               items[1],
                                               sub_components)
        else:
            input_name = items[1]
        comp_name = input_name + '.Scale.' + items[0]
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'Scale',
            'name': comp_name,
            'input': [input_name],
            'scale': scale
        }
        sub_components.append(component)
        return comp_name

    def parse_const_descp(self, input, sub_components):
        items = parenthesis_split(input, ",")
        kaldi_check(len(items) == 2, 'Const descriptor should have 2 items.')
        value = float(items[0])
        dimension = int(items[1])
        comp_name = 'Const' + '_' + items[0] + '_' + items[1]
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'Const',
            'name': comp_name,
            'input': [],
            'value': value,
            'dim': dimension,
        }
        sub_components.append(component)
        return comp_name

    def parse_round_descp(self, input, sub_components):
        items = parenthesis_split(input, ",")
        kaldi_check(len(items) == 2, 'Round descriptor should have 2 items.')
        sub_type = self.check_sub_inputs(items[0])
        if sub_type is not None:
            input_name = self.parse_descriptor(sub_type,
                                             items[0],
                                             sub_components)
        else:
            input_name = items[0]
        modulus = int(items[1])
        comp_name = input_name + '.Round.' + str(modulus)
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'Round',
            'name': comp_name,
            'input': [input_name],
            'modulus': modulus
        }
        sub_components.append(component)
        return comp_name

    def parse_switch_descp(self, input, sub_components):
        items = parenthesis_split(input, ",")
        kaldi_check(len(items) >= 2, 'Switch descriptor should have 2 items.')
        sub_type = self.check_sub_inputs(items[0])
        if sub_type is not None:
            input_name = self.parse_descriptor(sub_type,
                                               items[0],
                                               sub_components)
        else:
            input_name = items[0]
        sub_type = self.check_sub_inputs(items[1])
        if sub_type is not None:
            other_name = self.parse_descriptor(sub_type,
                                             items[1],
                                             sub_components)
        else:
            other_name = items[1]
        comp_name = input_name + '.Switch.' + other_name
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'Switch',
            'name': comp_name,
            'input': items,
        }
        sub_components.append(component)
        return comp_name, component

    def parse_sum_descp(self, input, sub_components):
        items = input.split(",")
        kaldi_check(len(items) == 2, 'Sum descriptor should have 2 items.')
        sub_type = self.check_sub_inputs(items[0])
        if sub_type is not None:
            input_name = self.parse_descriptor(sub_type,
                                               items[0],
                                               sub_components)
        else:
            input_name = items[0]
        sub_type = self.check_sub_inputs(items[1])
        if sub_type is not None:
            other_name = self.parse_descriptor(sub_type,
                                               items[1],
                                               sub_components)
        else:
            other_name = items[1]

        comp_name = input_name + '.Sum.' + other_name
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'Sum',
            'name': comp_name,
            'input': [input_name, other_name],
        }
        sub_components.append(component)
        return comp_name

    def parse_failover_descp(self, input, sub_components):
        items = parenthesis_split(input, ",")
        kaldi_check(len(items) == 2, 'Failover descriptor should have 2 items.')
        sub_type = self.check_sub_inputs(items[0])
        if sub_type is not None:
            input_name = self.parse_descriptor(sub_type,
                                               items[0],
                                               sub_components)
        else:
            input_name = items[0]
        sub_type = self.check_sub_inputs(items[1])
        if sub_type is not None:
            other_name = self.parse_descriptor(sub_type,
                                               items[1],
                                               sub_components)
        else:
            other_name = items[1]
        comp_name = input_name + '.Failover.' + other_name
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'Failover',
            'name': comp_name,
            'input': [input_name, other_name],
        }
        sub_components.append(component)
        return comp_name

    def parse_replace_index_descp(self, input, sub_components):
        items = parenthesis_split(input, ",")
        kaldi_check(len(items) == 3, 'ReplaceIndex descriptor should have 3 items.')
        sub_type = self.check_sub_inputs(items[0])
        if sub_type is not None:
            input_name = self.parse_descriptor(sub_type,
                                               items[0],
                                               sub_components)
        else:
            input_name = items[0]
        var_name = items[1]
        value = int(items[2])
        inputs = [input_name]
        comp_name = input_name + '.ReplaceIndex.' + items[1] + items[2]
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'ReplaceIndex',
            'name': comp_name,
            'input': inputs,
            'var_name': var_name,
            'modulus': value}
        sub_components.append(component)
        return comp_name

    def parse_ifdefine_descp(self, input, sub_components):
        if input.startswith('Offset('):
            sub_input = input[7:-1]
            items = sub_input.split(",")
            kaldi_check(len(items) == 2, 'IfDefined descriptor should have 2 items.')
            sub_type = self.check_sub_inputs(items[0])
            if sub_type is not None:
                input_name = self.parse_descriptor(sub_type,
                                                   items[0],
                                                   sub_components)
            else:
                if items[0] in self._component_names:
                    input_name = items[0]
                else:
                    input_name = items[0] + '.IfDefined'
            offset = int(items[1])
        else:
            input_name = input
            offset = 0
        comp_name = input_name + '.' + str(offset)
        # input_name = input_name + ".IfDefined"
        self._current_id += 1
        component = {
            'id': self._current_id,
            'type': 'IfDefined',
            'name': comp_name,
            'input': [input_name],
            'offset': offset,
        }
        sub_components.append(component)
        return comp_name

    def parse_component_lines(self):
        """Parse all components lines before </Nnet3>"""
        num = 0
        while True:
            line = next(self._line_buffer)
            pos = 0
            tok, pos = read_next_token(line, pos)
            if tok is None:
                line = next(self._line_buffer)
                pos = 0
                if line is None:
                    print("unexpected EOF on line:\n", line)
                    break
                else:
                    tok, pos = read_next_token(line, pos)
            if tok == '<ComponentName>':
                component_pos = pos
                component_name, pos = read_next_token(line, pos)

                component_type, pos = read_component_type(line, pos)
                assert is_component_type(component_type)
                component_dict, line, pos = self.read_component(line,
                                                                pos,
                                                                component_type)
                if component_dict is not None:
                    if component_name in self._components_by_name:
                        config_dict = self._components_by_name[component_name]
                        new_dict = merge_two_dicts(config_dict, component_dict)
                        self._components_by_name[component_name] = new_dict
                        num += 1
                else:
                    print("{0}: error reading component with name {1}"
                          " at position {2}"
                          .format(sys.argv[0],
                                  component_name,
                                  component_pos),
                          file=sys.stderr)
            elif tok == NNet3End:
                print("finished parsing nnet3 (%s) components." % num)
                assert num == self._num_components
                break
            else:
                print("{0}: error reading Component:"
                      " at position {1}, expected <ComponentName>,"
                      " got: {2}"
                      .format(sys.argv[0], pos, tok), file=sys.stderr)
                break

    def read_component(self, line, pos, component_type):
        terminating_token = "</" + component_type[1:]
        terminating_tokens = {terminating_token, '<ComponentName>'}

        type = component_type[1:-1]
        if type in self._component_parsers:
            action_dict = self._component_parsers[type]
            d, pos = self.read_generic(line,
                                       pos,
                                       self._line_buffer,
                                       terminating_tokens,
                                       action_dict)
            if d is not None:
                d['type'] = KaldiOpRawType[type]
                d['raw-type'] = component_type[1:-10]  # e.g. 'Linear'
            return d, line, pos
        else:
            print("Component: %s not supported yet." % type)
            return None, line, pos

    @staticmethod
    def read_generic(line, pos, line_buffer, terminating_token, action_dict):
        if isinstance(terminating_token, str):
            terminating_tokens = set([terminating_token])
        else:
            terminating_tokens = terminating_token
            assert isinstance(terminating_tokens, set)
        assert isinstance(action_dict, dict)

        # d will contain the fields of the object.
        d = dict()
        orig_pos = pos
        while True:
            tok, pos = read_next_token(line, pos)
            if tok in terminating_tokens:
                break
            if tok is None:
                line = next(line_buffer)
                if line is None:
                    print("{0}: error reading object starting at position {1},"
                          " got EOF "
                          "while expecting one of: {2}"
                          .format(sys.argv[0], orig_pos, terminating_tokens),
                          file=sys.stderr)
                    break
                else:
                    pos = 0
                    continue
            if tok in action_dict:
                func, name = action_dict[tok]
                obj, pos = func(line, pos, line_buffer)
                d[name] = obj
        return d, pos

    @staticmethod
    def parse_nnet3_config(line):
        if re.search(
            '^input-node|^component|^output-node|^component-node|^dim-range-node',
            line.strip()) is None:
            return [None, None]
        parts = line.split()
        config_type = parts[0]
        fields = []
        prev_field = ''
        for i in range(1, len(parts)):
            if re.search('=', parts[i]) is None:
                prev_field += ' ' + parts[i]
            else:
                if not (prev_field.strip() == ''):
                    fields.append(prev_field)
                sub_parts = parts[i].split('=')
                if len(sub_parts) != 2:
                    raise Exception('Malformed config line {0}'.format(str))
                fields.append(sub_parts[0])
                prev_field = sub_parts[1]
        fields.append(prev_field)

        parsed_string = {}
        try:
            while len(fields) > 0:
                value = re.sub(',$', '', fields.pop().strip())
                key = fields.pop()
                parsed_string[key.strip()] = value.strip()
        except IndexError:
            raise Exception('Malformed config line {0}'.format(str))
        return [config_type, parsed_string]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model file")
    parser.add_argument('--nnet-type', type=int,
                        dest='nnet_type', help='nnet type', default=3)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.input:
        with open(args.input, 'r') as f:
            if args.nnet_type == 3:
                parser = Nnet3Parser(f)
            else:
                parser = Nnet2Parser(f)
            parser.run()
    else:
        print("invalid file path ", args.input)
        sys.exit(1)

if __name__ == "__main__":
    main()
