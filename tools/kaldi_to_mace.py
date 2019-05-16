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

import numpy as np, os, sys, argparse


def save_to_txt(data, shape, out_path):
	header = 'utterance-id1  [\n'
	with open(out_path, 'w') as f:
		f.write(header)
		for n in xrange(shape[0]):
			d = data[n, :]
			d_str = " ".join(str(x) for x in d)
			if n < shape[0] - 1:
				d_str = d_str + '\n'
			else:
				d_str = d_str + ' ]\n'
			f.write(d_str)


def get_args():
	"""Parse commandline."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=True, type=str,
						help="kaldi data file path")
	parser.add_argument("--output", required=True, type=str,
						help="mace data file path")
	args = parser.parse_args()
	return args


def read_kaldi_output(file_path):
	data_lines = []
	with open(file_path, 'r') as f:
		lines = f.readlines()
		for l in lines:
			if '[' not in l:
				tmp = l.split()
				if ']' in l:
					del tmp[-1]
				data_line = [float(x) for x in tmp]
				data_lines.append(data_line)
	return np.array(data_lines)

def main():
	args = get_args()
	kaldi_data = read_kaldi_output(args.input)
	kaldi_data.astype(np.float32).tofile(args.output)

if __name__ == "__main__":
	main()
