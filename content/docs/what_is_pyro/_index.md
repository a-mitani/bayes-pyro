---
title: Pyroとは
weight: 1
#bookFlatSection: true
# bookCollapseSection: true
---
# Pyroとは
[Pyro](https://pyro.ai/)はUber AI labにより開発されている、オープンソースの確率的プログラミング言語(Probablistic Programing Language: PPL)です[^ppl]。Python上で動作し、Pythonのコードを実装する要領で、確率変数やそれらを組み合わせた独自の確率モデルを構築することを可能にしてくれます。

また同時に確率モデルの推論を行うための多くのアルゴリズムが実装されており、本書のメイントピックであるベイズ機械学習（以降、ベイズ学習）を非常にシンプルな形で実装することが可能になります。特に変分推論においては学習時に最大化していく対象であるELBO(Evidence Lower Bound)をユーザーが構築した独自の確率モデルに従って自動で計算する機能を備えており、Pyroを用いると変分推論の実装が非常に容易になります[^dnn]。このような機能により深層学習とベイズ学習を組み合わせた深層ベイズ学習などの実装も非常に容易になります。

Pyroはそのバックエンドに深層学習フレームワークの[Pytorch](https://pytorch.org/)を利用しているため、Pytorchのもつ自動微分(autograd)機能やGPU計算機能を利用することで高速な学習が可能になっているところに特徴があります。

# Pyroのインストール
## ローカル環境へのPyroのインストール
自身のローカル環境でPyroを利用する場合、まずは、まず[Pytorchをインストール](https://pytorch.org/get-started/locally/)する必要があります。Pytorchをインストール後、
```python
pip install pyro-ppl
```
コマンドによりPyroのインストールします。

## Google ColabratoryでのPyroのインストール
以降の本書のサンプルコードは全て[Google Colabratory](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja)上で動作させることを前提としています。ColabratoryはPytorchは事前にインストールされているため、下記の「!」を冒頭につけたpipコマンドをセル上で動かすだけででPyroをインストールすることが可能です。
```python
!pip install pyro-ppl
```
次節以降で、ベイズ学習の基礎とPyroを用いた実装方法を学んでいきます。

[^ppl]: 「プログラミング言語」と名乗ってはいますが、Pythonで動作するライブラリの位置付けなので、確率的プログラミングをするための「フレームワーク」と考えた方がわかりやすいかもしれません。
[^dnn]: これは言うなれば深層学習においてTensorflowやPytorchがユーザーが独自に構築したニューラルネットワークの損失関数の勾配を自動で計算してくれることに対応していると考えれば、この機能の便利さが容易に想像できます。