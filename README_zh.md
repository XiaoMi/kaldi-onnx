# Kaldi-ONNX

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build Status](https://api.travis-ci.org/XiaoMi/kaldi-onnx.svg?branch=master)](https://travis-ci.org/XiaoMi/kaldi-onnx)

[English](README.md) | [中文](README_zh.md)


**Kaldi-ONNX** 是一个将[Kaldi](https://github.com/kaldi-asr/kaldi)的模型文件转换为[ONNX](https://github.com/onnx/onnx)模型的工具。
转换得到的ONNX模型可以借助[MACE](https://github.com/XiaoMi/mace)框架部署到Android, iOS, Linux或者Windows设备端进行推理运算。

此工具支持Kaldi的Nnet2和Nnet3模型，大部分Nnet2和Nnet3组件都已支持，
目前已支持的组件清单可以在文件converter/commmon.py中查看。
此外，针对NNet3, 这个工具也支持将部分描述符(Descriptor)转换为ONNX的Node。
如有未支持的组件，欢迎提Commits和Issue，会及时更新完善。

除了DNN，越来越多的语音识别模型使用RNN或TDNN网络，为了使模型更易于移植和推理过程更加高效，
此工具将这类模型转成有向无环图的网络模型。


## 使用方法

### 1. 安装依赖
```sh
pip install -r requirements.txt
```

### 2. 准备源模型
目前Kaldi-ONNX工具支持Kaldi的文本模型作为输入。
如果你的模型文件是二进制格式的，请使用Kaldi的命令行工具`nnet-am-copy`
或`nnet3-copy`将其转换为文本格式。

Nnet2 模型
```sh
path/to/kaldi/src/nnet2bin/nnet-am-copy --binary=false          \
                                        final.raw final-test.mdl
```

Nnet3 模型
```sh
path/to/kaldi/src/nnet3bin/nnet3-copy --binary=false          \
                                      --prepare-for-test=true \
                                      final.raw final-test.mdl
```
注意！请务必设置`--prepare-for-test=true`，这个参数会对模型进行优化，
更多的Kaldi工具细节请参考[Kaldi项目文档](http://kaldi-asr.org/doc/)。


### 3. 转换
```sh
python converter/convert.py --input=path/to/kaldi/model/final.mdl  \
                            --output=path/to/save/onnx/model/final.onnx \
                            --chunk-size=20 \
                            --nnet-type=3
```

### 4. 查看模型网络结构
转换之后，可以使用[模型可视化工具](https://lutzroeder.github.io/netron/)查看网络结构。


### 5. 部署和测试

MACE框架已支持Kaldi算子以及ONNX格式模型，所以推荐使用MACE对转换好的模型进行部署和测试。
这里是[nnet2_fisher_english](https://github.com/XiaoMi/mace-models/tree/master/onnx-models)在Android设备上部署的例子。
更多的部署和使用方法可参考[MACE文档](https://mace.readthedocs.io/en/latest/)。


### 6. 验证
因为MACE已经支持了Kaldi大部分常用的组件和ONNX格式模型，
可以使用MACE框架对转换好的Kaldi-ONNX模型进行验证。
验证过程是使用相同的输入数据，检查模型在Kaldi上计算得到的结果和MACE上计算得到的结果是否一致。

**生成随机输入数据**
这里提供了脚本工具用来产生Kaldi和MACE的随机输入数据。
```sh
python tools/generate_inputs.py --input_dim=40 \
                                --chunk_size=20 \
                                --kaldi_input_file=path/to/kaldi/input/test_input.ark \
                                --mace_input_file=path/to/mace/input/test_input.npy
```
`input_dim` 和 `chunk_size` 是模型的input dim 和 chunk，可以根据不同的模型或需求设置为不同的值.
`kaldi_input_file` 和 `mace_input_file` 用来保存生成的输入数据文件. 

**在Kaldi上计算**
Kaldi提供了命令行工具可以指定输入数据对模型进行推理计算。

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

**转换输出数据格式**
运行上面的命令后，会得到Kaldi对模型进行推理计算后的结果，这个输出数据文件是文本格式的。
由于MACE支持numpy格式的数据作为输入和输出，为了方便验证，
需要将文本格式的Kaldi输出数据文件转换为numpy格式。
这里也提供了一个python脚本工具帮助完成这个转换。
```sh
python tools/kaldi_to_mace.py  --input=path/to/kaldi/data/file.ark \
                               --output=path/to/save/numpy/data/file.npy
```

**准备模型的配置文件**
下面是一个Fisher English 模型的示例MACE配置文件：

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

** 验证**
使用MACE进行转换和验证:

```sh
cd path/to/mace
python tools/converter.py convert --config=path/to/fisher_english.yml
python tools/converter.py run --config=path/to/fisher_english.yml --validate

```
上述命令会给出模型在MACE和Kaldi上的输出结果的相似程度。
关于如何使用MACE的更多详细信息，请参考[MACE使用文档](https://mace.readthedocs.io/en/latest/)


## 示例
我们已转换了一批Kaldi的Nnet2和Nnet3模型，放在[MACE Model Zoo](https://github.com/XiaoMi/mace-models)项目里.


## 交流与反馈
* 欢迎通过Github Issues提交问题报告与建议
* 欢迎提交代码，代码格式请遵照[PEP8风格Python代码规范](https://www.python.org/dev/peps/pep-0008/),
  可以使用[pycodestyle](https://github.com/PyCQA/pycodestyle)工具进行检查 `pycodestyle $(find . -name "*.py")`。

## License
[Apache License 2.0](LICENSE)
