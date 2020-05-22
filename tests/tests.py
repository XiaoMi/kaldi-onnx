#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Test converter."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


def _test_one_model(model_dir, left_context, right_context):
  """Convert one model and check output.

  Args:
    model_dir: model dir for test.
    left_context: left context of kaldi model.
    right_context: right context of kaldi model.

  Returns:
    max err between tensorflow pb output and kaldi output.
  """
  with TemporaryDirectory() as tmp_dir:
    kaldi_model_path = model_dir / "final.txt"
    pb_path = Path(tmp_dir) / "tf.pb"
    Converter(kaldi_model_path, left_context, right_context, pb_path).run()

    with tf.compat.v1.Session() as session:
      with gfile.FastGFile(pb_path, 'rb') as pb_file:
        tf.compat.v1.reset_default_graph()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(pb_file.read())
        tf.import_graph_def(graph_def)

      feat_input = np.loadtxt(model_dir / "input.txt", dtype=np.float32)
      feat_ivector = np.loadtxt(model_dir / "ivector.txt", dtype=np.float32)
      feed_dict = {"input:0": feat_input, "ivector:0": feat_ivector}
      out_tensor = session.graph.get_tensor_by_name("output.affine:0")
      output = session.run(out_tensor, feed_dict)

  kaldi_output = np.loadtxt(model_dir / "output.txt", dtype=np.float32)
  return np.amax(np.absolute(np.subtract(output, kaldi_output)))


if __name__ == '__main__':
  unittest.main()
