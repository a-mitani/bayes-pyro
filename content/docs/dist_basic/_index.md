---
title: 確率分布の取り扱い
weight: 1
---
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>

# 確率分布の取り扱い
前節でPyroに用意された正規分布やベルヌーイ分布の関数（`dist.Normal`、`dist.Bernoulli`）を利用しましたが、本節ではもう少し詳しくPyroでの確率分布の取り扱いについて見ていくことにします。

## ■ 実現値のサンプリング
Pyroでの確率分布はPytorchの確率分布関数の薄いラッパークラス[^mixin]として定義されているため、Pytorchの確率分布クラスで定義されている各種機能が利用可能です。そのため確率分布からの実現値は以下のように`sample()`関数を用いてサンプリングします。
```python
normal_1d = dist.Normal(0.0, 1.0)
print(normal_1d.sample())

##Output
# tensor(-0.6117)
```
多次元の確率分布も同様です。例えば２次元の正規分布関数からは２次元のサンプリング値が出力されます。
```python
normal_2d = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
print(normal_2d.sample())

## Output
# tensor([2.5499, 0.2041])
```
また、`sample`関数に`torch.Size`を引数に渡すことで指定のshapeでサンプリングすることができます。
```python
normal_1d = dist.Normal(0.0, 1.0)
samples = normal_1d.sample(torch.Size([2, 3]))
print(samples)

## Output
tensor([[ 0.7036,  2.2816,  0.5640],
        [ 0.3072, -1.0661, -1.6618]])
```
ここでサンプリングされた値間は**独立同分布（IID）の関係**にあることに注意してください。

前節のように確率モデリングをする場合、後々の推論などで確率変数の名前をつけると扱いやすいです。その場合は`pyro.sample`関数を用いて確率変数に名前をつけた上でサンプリングを行うことが可能です。下の例ではベルヌーイ分布の分布をもつ確率変数`X`からの実現値が変数`x`に格納される動作になります。
```python
x = pyro.sample('X', dist.Bernoulli(0.5))
print(x)

## Output
# tensor(1.)
```

## ■ 確率（密度）の取得
確率分布から指定された実現値がとる確率を計算するには`log_prob`関数を利用します。これは対数確率密度を計算するので生の確率密度値を計算する場合はこの出力に対して`np.exp()`を計算します。
```python
bel_dist = dist.Bernoulli(0.6)
x = torch.tensor([1.0, 0.0])
log_prob_x = bel_dist.log_prob(x)
print(np.exp(log_prob_x))

## Output
# tensor([0.6000, 0.4000])
```

## ■ batch_shapeとevent_shape
### batch_shape
確率分布のパラメータとして複数の値をTensorとして与えることで、同一の分布だが異なるパラメータで特徴付けられた複数の分布を同時に定義することが出来ます。例えば下記のコードでは
* 平均: 0.0、標準偏差: 1.0
* 平均: 3.0、標準偏差: 0.5

の２つの分布を１つの確率分布変数として定義していいます。
```python
locs = torch.tensor([0.0, 3.0])
sds = torch.tensor([1.0, 0.5])
normal_1d_cond = dist.Normal(locs, sds)
print("batch_shape =", normal_1d_cond.batch_shape)

## Output
# batch_shape = torch.Size([2])
```
ここで`batch_shape`という属性が出てきました。これは確率分布変数に定義された分布の種類の数を示しており、上記例では二種類のパラメータを指定しているので「2」になります。また以下のコードのように`sample`関数を呼ぶことにより、定義された２つのパラメータでの確率分布の実現値がサンプリングされます。ここで各パラメータのサンプル値`samples[:, 0]`、`samples[:, 1]`のそれぞれ独立同分布であり、一方でお互いは独立ではあるが同分布ではないことになります。

```python
samples = normal_1d_cond.sample(torch.Size([10000]))
print("samples.shape =", samples.shape)

plt.hist(samples[:, 0], alpha=0.3)
plt.hist(samples[:, 1], alpha=0.3)
plt.xlabel("X")
plt.ylabel("Frequency")
plt.show()

## Output
# samples.shape = torch.Size([10000, 2])
```
<center>
<img src="hist.png" width="400">
</center>

また以下のように確率密度を求めた場合も、各パラメータでの確率密度が出力されます。最後の行のprint文に示したように**確率密度のshapeは確率分布変数の`batch_shape`と等しい**ことに注意してください。

```python
log_prob_x = normal_1d_cond.log_prob(torch.Tensor([1.0]))
print(np.exp(log_prob_x))
print(log_prob_x.shape == normal_1d_cond.batch_shape)

## Output
# tensor([0.2420, 0.0003])
# True
```

### event_shape
二次元正規分布を考えてみます。この分布の`event_shape`を出力すると「２」となります。この`event_shape`とは従属な変数の数を示しています。つまり２つの変数が決まって初めて１つの確率密度が求まる分布であることを示しています（以下コードの最後の`print`文）。
```python
normal_2d = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
print(normal_2d.event_shape)
sampled = normal_2d.sample()
print("(x, y)=", sampled)
print("P(x, y) =", np.exp(normal_2d.log_prob(sampled)))

## Output
# torch.Size([2])
# (x, y)= tensor([1.9866, 0.4698])
# P(x, y) = tensor(0.0198)
```

### 従属変数化 `to_event()`
Pyroでは`to_event()`関数を用いて、単変数確率分布を組み合わせて多変数確率分布に変形することが可能です。
以下のようにベルヌーイ分布を定義します。2×2の計4つのパラメータの分布を指定して流ので`batch_shape`も2×2になります。当然、確率分布はそれぞれの確率分布で計算されます。
```python
ps = torch.Tensor([[0.3, 0.8], [0.1, 1.0]])
bern_dist = dist.Bernoulli(ps)
print("bern_dist.batch_shape =", bern_dist.batch_shape)
print("bern_dist.event_shape =", bern_dist.event_shape)

val = torch.Tensor([1.0])
print("P(x=1) =", np.exp(bern_dist.log_prob(val)))

## 
# bern_dist.batch_shape = torch.Size([2, 2])
# bern_dist.event_shape = torch.Size([])
# P(x=1) = tensor([[0.3000, 0.8000],
#         [0.1000, 1.0000]])
```
この確率分布変数に対して`to_event()`を適用します。
```python
bern_dist2 = bern_dist.to_event(1)
print("bern_dist2.batch_shape =", bern_dist.batch_shape)
print("bern_dist2.event_shape =", bern_dist.event_shape)

val = torch.Tensor([1.0])
print("P(x=1) =", np.exp(bern_dist2.log_prob(val)))

## Output
# bern_dist2.batch_shape = torch.Size([2, 2])
# bern_dist2.event_shape = torch.Size([])
# P(x=1) = tensor([0.2400, 0.1000])
```
ここで`.to_event(1)`の引数`1`は`batch_shape`の右から１つ目だけを従属化するという指定になります。

## ■ ベイズ学習へ
ここまででPyroを用いて対象の事象に合わせた確率モデル、いわゆる生成モデルを定義することを行ってきました。ベイズ学習ではこの確率モデルに観測されたデータを組み合わせることで、未知のパラメータを学習・推論することになります。例えば、前節の赤玉白玉問題の場合、取り出された玉の色のデータをもとに袋の中の赤玉の数を推定していくことを行います。

Pyroを用いてベイズ学習を実装していく前に、必要最小限のベイズ学習の知識を復習していきましょう。

[^mixin]:つまり[`torch.distributions.distribution.Distribution`クラス](https://pytorch.org/docs/master/distributions.html#torch.distributions.distribution.Distribution)と、[`TorchDistributionMixin`](https://docs.pyro.ai/en/dev/distributions.html#pyro.distributions.torch_distribution.TorchDistributionMixin)の多重継承サブクラスとして実装されています。
