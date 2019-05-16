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

import sys, os, argparse
import numpy as np


def generate_features(batch, chunk, dim):
    data = np.random.rand(batch, chunk, dim)
    print "genearted data shape:", data.shape
    return data

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
    parser.add_argument('--input_dim', type=int, help='input dim', default=40)
    parser.add_argument('--batch', type=int, help='batch size', default=1)
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=20)
    parser.add_argument("--kaldi_input_file", required=True, type=str,
                        help="kaldi input data file path")
    parser.add_argument("--mace_input_file", required=True, type=str,
                        help="mace input data file path")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data = generate_features(args.batch, args.chunk_size, args.input_dim)
    save_mace_input(data, args.mace_input_file)
    save_kaldi_input(data, [args.batch, args.chunk_size, args.input_dim], args.kaldi_input_file)

if __name__ == "__main__":
    main()
