# Kaldi-ONNX Converter

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[English](README.md) | [中文](README_zh.md)

**Kaldi-ONNX** is a tool for porting [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi) 
neural network models to [ONNX](https://github.com/onnx/onnx) models for inference.
With the converted ONNX model, you can use [MACE](https://github.com/XiaoMi/mace)
to speedup the inference on **Android**, **iOS**, **Linux** or **Windows** devices with
highly optimized NEON kernels (more heterogeneous devices will be supported in the future).

This tool supports converting both Nnet2 and Nnet3 models. Almost all components
in Nnet2 and Nnet3 models are supported, and the available components are listed
in converter/common.py. 

Besides DNN, many speech recognition models are using RNN or TDNN networks.
To make them more portable and efficient, this tool will convert these RNN and
TDNN networks to DAG-like networks.

## Usage

### 1. Dependencies

```sh
pip install -r requirements.txt
```

### 2. Prepare models
This tool only supports Kaldi's text model as an input.

If you have a binary model, Kaldi's `nnet-am-copy` or `nnet3-copy` tool can help you get a text one:

Nnet2 Model

```sh
path/to/kaldi/src/nnet2bin/nnet-am-copy --binary=false          \
                                        final.raw text.mdl
```

Nnet3 Model

```sh
path/to/kaldi/src/nnet3bin/nnet3-copy --binary=false          \
                                      --prepare-for-test=true \
                                      final.raw text.mdl
```

Don't forget to use the `--prepare-for-test=true` option to optimize the model.

More details about kaldi's tools  are in [Kaldi's Documentation](http://kaldi-asr.org/doc/).


### 3. Convert
One command to convert:

```sh
python converter/convert.py --input=models/kaldi/model_name.mdl  \
                            --output=models/onnx/model_name.onnx \
                            --chunk-size=20 \
                            --nnet-type=3
```


### 4. Graph review
After converting, there is a graphic tool for you to review the onnx model: [ONNX Model Viewer](https://lutzroeder.github.io/netron/).


### 5. Deployment

To run the ONNX models, an inference framework with Kaldi specific operators support is needed.
Currently, [MACE](https://github.com/XiaoMi/mace) has Kaldi-ONNX support which is primarily optimized for mobile phones and IoT devices.

For more details about deployment, please refer to [MACE documents](https://mace.readthedocs.io/en/latest/).


### 6. Validation

Since MACE already supports most frequently used components in Kaldi.
 We can use it to validate converted Kaldi models.
The validation process is giving the same inputs to Kaldi and MACE's computation and
 check if they will output the same results.

**Generate random input**

We provide a tool to generate the same random input data for Kaldi and MACE's computation.

```sh
python tools/generate_inputs.py --input_dim=40 \
                                --chunk_size=20 \
                                --kaldi_input_file=path/to/kaldi/input/test_input.ark \
                                --mace_input_file=path/to/mace/input/test_input.npy

```
'input_dim' and 'chunk_size' are model's input dim and chunk, can be specified by case.
'kaldi_input_file' and 'mace_input_file' are paths for saving generated data files. 

**Compute in Kaldi**

Kaldi has command line tools for computing the model's propogation.

Nnet2:

```sh
path/to/kaldi/src/nnet2bin/nnet-am-compute  path/to/kaldi/model/file/final.mdl \
                                            ark,t:/path/to/kaldi/input/file/test_input.ark \
                                            ark,t:path/to/save/output/data/test_output.ark

```

Nnet3:

```sh
path/to/kaldi/src/nnet3-compute  path/to/kaldi/model/file/final.mdl \
                                 ark,t:/path/to/kaldi/input/file/test_input.ark \
                                 ark,t:path/to/save/output/data/test_output.ark

```
**Convert output data file**

After running `nnet-compute`, we'll get Kaldi's output data in text format .
Because MACE supports numpy data file as input or output, we need to convert Kaldi's output data file to numpy format.

The script tools/kaldi_to_mace.py will help you doing this convertion.

```sh
python tools/kaldi_to_mace.py  --input=path/to/kaldi/data/file.ark \
                               --output=path/to/save/numpy/data/file.npy
```

**Prepare config file for MACE**

Here is an example yaml config file for fisher english model.

```yaml
# fisher_english.yml

library_name: fisher_english_8
target_abis: [armeabi-v7a, arm64-v8a]
model_graph_format: file
model_data_format: file
models:
  fisher_english_8:
    platform: onnx
    model_file_path: https://cnbj1.fds.api.xiaomi.com/mace/miai-models/onnx/kaldi/nnet2/fisher_english_8_nnet_a.onnx
    model_sha256_checksum: e27d8147995b0a68e1367d060dc4f41c0f434043992a52548ff961e4e1e87e6c
    subgraphs:
      - input_tensors:
          - 0
        input_shapes:
          - 1,20,140
        output_tensors:
          - 17
        output_shapes:
          - 1,20,7880
        backend: kaldi
        input_data_formats: NONE
        output_data_formats: NONE
        validation_inputs_data:
            - https://cnbj1.fds.api.xiaomi.com/mace/miai-models/onnx/kaldi/data/kaldi_input_20_140.npy
        validation_outputs_data:
            - https://cnbj1.fds.api.xiaomi.com/mace/miai-models/onnx/kaldi/data/test_fisher_english_8_20_140_out.npy
    backend: kaldi
    runtime: cpu
    limit_opencl_kernel_time: 0
    nnlib_graph_mode: 0
    obfuscate: 0

```

**Validate**

Convert and validate the model in MACE:

```sh
cd path/to/mace
python tools/converter.py convert --config=path/to/fisher_english.yml
python tools/converter.py run --config=path/to/fisher_english.yml --validate

```
The command will give you the similarity between MACE's and Kaldi's output results.

More details about how to use MACE, please step to [MACE documents](https://mace.readthedocs.io/en/latest/).

### 5. Examples

We have converted numbers of Kaldi's Nnet2 and Nnet3 models, and put them in [MACE Model Zoo](https://github.com/XiaoMi/mace-models).


## Communication
* GitHub issues: bug reports, usage issues, feature requests


## License
[Apache License 2.0](LICENSE).


## Contributing
Any kind of contribution is welcome. For bug reports, feature requests,
please just open an issue without any hesitation. For code contributions, it's
strongly suggested to open an issue for discussion first.


## Acknowledgement
kaldi-onnx learned a lot from the following projects during the development:
* [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi),
* [ONNX](https://github.com/onnx/onnx).
