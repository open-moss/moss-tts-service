# MOSS TTS合成服务

## 配置要求

建议使用显卡推理，至少2GB显存，建议6GB以上为佳，推理文本越长越消耗显存，过低有溢出风险。

## 模型下载

下载模型将 moss.pth 和 config.json 两个文件放置到仓库根目录的 **model** 目录内。

[MOSS vits模型 (900 epochs)](https://github.com/open-moss/moss-vits-model)

## 创建conda环境

```shell
conda create -n moss-tts python=3.8
conda activate moss-tts
```

## 安装依赖

```shell
pip install -r requirements.txt
```

如果pytorch依赖和显卡CUDA版本不匹配将导致无法使用GPU加速推理，需要卸载已安装的torch、torchvision、torchaudio后根据[pytorch官网](https://pytorch.org/)选择对应版本的pip安装命令安装。

## 构建monotonic_align
Windows的二进制文件已经预构建，无需执行此步骤。
```shell
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```

## 启动服务

```shell
python server.py
```

## 调用TTS合成

```
POST /text_to_speech
{
    "text": "大家好，我是MOSS",
    "speechRate": 1
}
```