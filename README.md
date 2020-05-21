# Kaldi-ONNX-Tensorflow Converter

This is a tool for converting [Kaldi](https://github.com/kaldi-asr/kaldi) model to tensorflow model for inference based on XiaoMi's [Kaldi-ONNX](https://github.com/XiaoMi/kaldi-onnx) project.

## Background

XiaoMi's project can convert Kaldi model to onnx model. Usually, onnx model can be easily convert to tensorflow model using [ONNX-TF](https://github.com/onnx/onnx-tensorflow) toolkit.

However, their implementation of converted onnx node can only support [MACE](https://github.com/XiaoMi/mace) framework.

Based on their project, Kaldi node is implemented using common onnx node, so that it can be easily converted to tensorflow model for inference.

## Support

The inference of Kaldi's most popular TDNN-F networks in nnet3 is supported.

## Usage

### 1. Environment

python3.6

### 2. Dependencies

```sh
pip3 install -r requirements.txt
```

### 3. Model Preparation

This tool only supports Kaldi's text model as an input.

If you have a binary model, Kaldi's `nnet3-copy` tool can help you get a text one:

```sh
kaldi/src/nnet3bin/nnet3-copy --binary=false --prepare-for-test=true final.mdl text.mdl
```

Don't forget to use the `--prepare-for-test=true` option.

Before converting, you need to use `nnet3-am-info` to get left_context and right_context,
which is required for converting.

### 4. Convert

```sh
python3 -m converter.convert <input_kaldi_model> <left_context> <right_context> <out_tensorflow_pb> 
```

### 5. Graph review

After converting, there is a graphic tool for you to review the onnx model: [ONNX Model Viewer](https://lutzroeder.github.io/netron/).

## Acknowledgement

* [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi).
* [Kaldi-ONNX](https://github.com/XiaoMi/kaldi-onnx)
* [ONNX](https://github.com/onnx/onnx).
* [ONNX-TF](https://github.com/onnx/onnx-tensorflow).
