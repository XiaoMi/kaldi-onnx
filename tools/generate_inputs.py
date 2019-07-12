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

from __future__ import division
from __future__ import print_function

import argparse
import numpy as np


def generate_features(batch, chunk, dim):
    data = np.random.rand(batch, chunk, dim)
    print "genearted data shape:", data.shape
    return data


def pad_context(input_data,
                left_context,
                right_context):
    if left_context > 0:
        first = np.expand_dims(input_data[:, 0, :], axis=1)
        first = np.repeat(first, left_context, axis=1)
        out_data = np.concatenate((first, input_data), axis=1)
    else:
        out_data = input_data
    if right_context > 0:
        last = np.expand_dims(input_data[:, -1, :], axis=1)
        last = np.repeat(last, right_context, axis=1)
        out_data = np.concatenate((out_data, last), axis=1)
    print "genearted padded context data shape:", out_data.shape
    return out_data


def save_mace_input(data, file_path):
    # np.save(file_path, data)
    data.astype(np.float32).tofile(file_path)


def save_kaldi_input(data, shape, out_path):
    with open(out_path, 'w') as f:
        for b in xrange(shape[0]):
            header = 'utterance-id' + str(b) + '  [\n'
            f.write(header)
            for n in xrange(shape[1]):
                d = data[b, n, :]
                d_str = " ".join(str(x) for x in d)
                if n < shape[1] - 1:
                    d_str = d_str + '\n'
                else:
                    d_str = d_str + ' ]\n'
                f.write(d_str)
            # f.write('\n')


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dim', type=int, dest='input_dim',
                        help='input dim', default=40)
    parser.add_argument('--batch', type=int, dest='batch',
                        help='batch size', default=1)
    parser.add_argument('--left-context', type=int, dest='left_context',
                        help='left context', required=True)
    parser.add_argument('--right-context', type=int, dest='right_context',
                        help='right context size', required=True)
    parser.add_argument('--chunk-size', type=int, dest='chunk_size',
                        help='chunk size', required=True)
    parser.add_argument("--kaldi-file", required=True, type=str,
                        dest="kaldi_data_file",
                        help="kaldi data file path")
    parser.add_argument("--mace-file", required=True, type=str,
                        dest="mace_data_file",
                        help="mace data file path")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data = generate_features(args.batch, args.chunk_size, args.input_dim)
    save_kaldi_input(data, [args.batch, args.chunk_size,
                            args.input_dim], args.kaldi_data_file)
    mace_data = pad_context(data, args.left_context, args.right_context)
    save_mace_input(mace_data, args.mace_data_file)


if __name__ == "__main__":
    main()
