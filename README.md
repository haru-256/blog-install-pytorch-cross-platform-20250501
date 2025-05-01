# uvでcross-platformなCPU/GPU用PyTorchを自動インストール

## 概要

`uv` の **index マッピング機能** を使うと、
1 つの lock ファイルで **CPU 版 / CUDA 版 PyTorch** を自動切り替えできます。
本記事では **uv ≥ 0.5.3** を前提に、CUDA 12.8 対応 PyTorch 2.7 を例に設定手順を解説します。

ソースコードは以下のリポジトリにあります。

## はじめに

PyTorch のwheelは **CPU 用** と **GPU（CUDA 版）** が異なる index に配置されています。
OS とアクセラレータ別の組み合わせは次のとおりです。

| OS \\ Compute | CPU | GPU (CUDA 12.8) |
|--------------|-------------------------------------------|----------------------------------------------|
| **Linux**    | <https://download.pytorch.org/whl/cpu>    | <https://download.pytorch.org/whl/cu128>     |
| **Windows**  | <https://pypi.python.org/simple>          | <https://download.pytorch.org/whl/cu128>     |
| **macOS**    | <https://pypi.python.org/simple>          | —（CUDA ビルドなし / Apple Silicon は MPS 対応） |

私の知る限り、`pipenv`ではOSごとにindexを指定することができないため、OSごとにlockファイルを分ける必要がありました。

- <https://pipenv.pypa.io/en/latest/indexes.html>

そのため、従来は OS ごとに lock ファイルを分割したり、手動で `--index-url` を切り替えたりしていましたが、`uv` の **optional-dependencies** 機能を使えば lockファイルを1つにすることが可能です。
これにより、OSやCUDAの有無を意識せず、1つのコマンドで PyTorch をインストールできるようになります。

## 要件と事前条件

要件は以下のとおりです。

- OSとCPU/CPUに応じてインストールするindexを自動的に選択する
- OSは自動的に認識してインストールするindexを選択し、CPU/GPUはCUDAの有無でindexを選択する
- コマンド一発でOSとCPU/GPUに応じてインストールするindexを選択してPyTorchをインストールする

事前条件は以下のとおりです。

- uv ≥ 0.5.3 （`uv --version` で確認）
- NVIDIA GPU マシンは CUDA 12.8 ドライバーをインストール済み

## 設定

### pyproject.tomlの設定

uv のoptional-dependenciesを使用して、OSごとにindexを変更することができます。
optional-dependenciesは必須ではない依存関係を指定するためのセクションです。`torch`自体は本当は必須なのですが、これをoptional-dependenciesに指定することで、ごとにindexを変更することができます。

具体的な`pyproject.toml`の設定は以下のようになります。

```toml
[project]
name = "blog-install-pytorch-cpu-gpu"
version = "0.1.0"
description = "CPU/GPUのPyTorchをインストール方法を解説するブログのサンプルコード"
readme = "README.md"
requires-python = ">=3.12"
dependencies = ["numpy~=2.2.5", "loguru~=0.7.3"]

[project.optional-dependencies]
cpu = ["torch==2.7.0"]
gpu = ["torch==2.7.0"]

[dependency-groups]
dev = ["pytest~=8.3.5", "pytest-mock~=3.14.0", "deepdiff~=8.4.2"]
lint = ["ruff~=0.11.7", "mypy~=1.15.0"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "gpu" }]]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu-mac", extra = "cpu", marker = "platform_system == 'macOS'" },
  { index = "pytorch-cpu", extra = "cpu", marker = "platform_system != 'macOS'" },
  { index = "pytorch-gpu", extra = "gpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu-mac"
url = "https://pypi.python.org/simple"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

上記のpyproject.tomlをもとに`uv lock` で `uv.lock` を作成すると、OSやCPU/GPUに応じてインストール時に使用するindexが自動的に選択された`uv.lock`が作成されます。

```lock
[package.optional-dependencies]
cpu = [
    { name = "torch", version = "2.7.0", source = { registry = "https://download.pytorch.org/whl/cpu" }, marker = "(platform_system != 'macOS' and sys_platform == 'darwin' and extra == 'extra-6-sample-cpu') or (platform_system == 'macOS' and extra == 'extra-6-sample-cpu' and extra == 'extra-6-sample-gpu') or (sys_platform != 'darwin' and extra == 'extra-6-sample-cpu' and extra == 'extra-6-sample-gpu')" },
    { name = "torch", version = "2.7.0", source = { registry = "https://pypi.python.org/simple" }, marker = "(platform_system == 'macOS' and extra == 'extra-6-sample-cpu') or (extra == 'extra-6-sample-cpu' and extra == 'extra-6-sample-gpu')" },
    { name = "torch", version = "2.7.0+cpu", source = { registry = "https://download.pytorch.org/whl/cpu" }, marker = "(platform_system != 'macOS' and sys_platform != 'darwin' and extra == 'extra-6-sample-cpu') or (platform_system == 'macOS' and extra == 'extra-6-sample-cpu' and extra == 'extra-6-sample-gpu') or (sys_platform == 'darwin' and extra == 'extra-6-sample-cpu' and extra == 'extra-6-sample-gpu')" },
]
gpu = [
    { name = "torch", version = "2.7.0+cu126", source = { registry = "https://download.pytorch.org/whl/cu126" } },
]
```

### CPU/GPUの有無でIndexを変更する(Makefile)

`pyproject.toml`の設定によりOSによって自動的にindexを切り替えることができるようになりました。
しかし、ユーザーから明示的なextraの指定がないとCPU/GPUはindexを変更することはできません。
そこで、Makefileを使用して、CPU/GPUの有無でindexを変更する方法を紹介します。

具体的には、以下のようなMakefileを作成します。
まず、`nvcc`コマンドが存在するかどうかを確認し、存在する場合はGPU対応のpytorchがインストールされるようにします。

```makefile
HAS_CUDA := $(shell command -v nvcc 2> /dev/null && echo 1 || echo 0)

.PHONY: install
install: ## Setup the project
 @if [ $(HAS_CUDA) -eq 1 ]; then \
  echo "=== Install with GPU support ==="; \
  uv sync --all-groups --extra=gpu; \
 else \
  echo "=== Install with CPU support ==="; \
  uv sync --all-groups --extra=cpu; \
 fi
```

## 使い方

`uv.lock`を作成したら、`make install`を実行することで、OSやCPU/GPUに応じてインストール時に使用するindexを自動的に選択してPyTorchをインストールすることができます。

```bash
uv lock # 初回だけ。cross-platform な uv.lock を生成
make install # 以降は OS / GPU 判定込みで
```

## まとめ

- `uv`を使用することで、OSやCPU/GPUに応じてインストール時に使用するindexを自動的に選択することができるようになりました。
- これにより、OSごとにlockファイルを分ける必要がなくなり、1つのlockファイルで全てのOS/CPU/GPUに対応することができるようになり、 CI/CD もシンプルになります。
- また、Makefileを使用することで、CPU/GPUの有無でindexを変更することができるようになりました。
- これにより、ユーザーから明示的なextraの指定がなくても、CPU/GPUに応じてindexを変更することができるようになりました。

## 最後に

`uv pip install torch --torch-backend=auto` というpreview機能もあり、CUDA バージョンを解析して最適な index を選ぶことができます。この機能を使えばCUDAが12.6かどうかを意識せずインストールすることができます。ただこの場合、1つのlockファイルで全てのOS/CPU/GPUに対応することはできません。

<https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection>
