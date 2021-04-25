---
title: 変分推論の基礎
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

# 変分推論の基礎
前節でベイズ学習の枠組みの３つの要点を簡単に説明しました。つまり

1. ベイズ学習とは観測データ$\mathbf{X}$が得られたという条件下での未知のパラメータ$\mathbf{w}$の確率分布、すなわち$p(\mathbf{w}|\mathbf{X})$を学習する作業であること。

2. 一般的にベイズ学習は①対象とする事象の発生過程のモデル化（確率モデルの構築）を行ったうえで、②確率モデルと観測データをもとに未知パラメータを推論する、という枠組みで行うこと。

3. 条件付き確率の定義から、$W$が離散変数の場合、$$p(\mathbf{w}|\mathbf{X})=\frac{p(\mathbf{w},\mathbf{X})}{p(\mathbf{X})}=\frac{p(\mathbf{w},\mathbf{X})}{\sum_{\mathbf{w}} p(\mathbf{w},\mathbf{X})}\tag{1}$$もしくは$W$が連続変数の場合、$$p(\mathbf{w}|\mathbf{X})=\frac{p(\mathbf{w},\mathbf{X})}{p(\mathbf{X})}=\frac{p(\mathbf{w},\mathbf{X})}{\int_{\mathbf{w}} p(\mathbf{w},\mathbf{X})d\mathbf{w}}\tag{2}$$を計算することで$p(\mathbf{w}|\mathbf{X})$を推論することができること。

の３点です。

しかし、前節で例示したような簡単な例を除いて、現実のほとんどの問題では、(1)式や(2)式の周辺分布（つまり和や積分の部分）は計算量が膨大であったり積分が解析的に不可能であり、厳密に計算できません。そこでこの周辺分布の計算を近似的にかつ現実的な時間内で計算する手法としてサンプリングや変分近似の手法が考案されてきました。ここではその１つである変分近似の手法について解説します。

※ 以降では確率変数が連続変数の場合に限って説明をしていきますが離散変数の場合でも同様の議論が可能です。

## 変分近似
我々は$p(\mathbf{w}|\mathbf{X})$を求めたいわけですが、この未知の確率分布がなんらかシンプルな関数$q(\mathbf{w})$で表現できないかと考えます。この$q(\mathbf{w})$を本来求めたい確率分布$p(\mathbf{w}|\mathbf{X})$に近づけていくことで$p(\mathbf{w}|\mathbf{X})$を近似的に求めてやろうという方法が**変分近似**です。

ここで$p(\mathbf{w}|\mathbf{X})$と近似関数$q(\mathbf{w})$の類似度合いの指標としてKLダイバージェンスを採用すると変分近似は、
$$ q_{opt.}(\mathbf{w})=\underset{q}{\operatorname{argmax}} \operatorname{KL}(q(\mathbf{w})||p(\mathbf{w}|\mathbf{X}))$$
の最適化問題として定式化できます。

しかしKLダイバージェンスに未知の関数である$p(\mathbf{w}|\mathbf{X})$が入っているため、このままでは$q_{opt.}$を求めることはできません。そこでこのKLダイバージェンスの最小化問題を、数学的なトリックを使って別の計算可能な量の最大化問題に書き換えることで間接的にKLダイバージェンスを最小化する関数を求めることを行います。ではどのようにするのでしょうか？

## ELBO
ここで対数周辺尤度$\ln{p(\mathbf{X})}$を以下のように書き換えることができることに着目します。
$$
\begin{align} 
\ln{p(\mathbf{X})} & = \ln{p(\mathbf{X})} \int_{\mathbf{w}} q(\mathbf{w}) d\mathbf{w}\newline
& = \int_{\mathbf{w}} q(\mathbf{w}) \ln\frac{p(\mathbf{X}, \mathbf{w})}{p(\mathbf{w} \vert \mathbf{X})} d\mathbf{w}\newline & = \int_{\mathbf{w}} q(\mathbf{w}) \ln \frac{p(\mathbf{X}, \mathbf{w})~q(\mathbf{w})}{p(\mathbf{w} \vert \mathbf{X}) ~q(\mathbf{w})} d\mathbf{w}\newline 
& = \int_{\mathbf{w}} q(\mathbf{w}) \ln \frac{p(\mathbf{X}, \mathbf{w})}{q(\mathbf{w})} d\mathbf{w} + \int_{\mathbf{w}} q(\mathbf{w}) \ln \frac{q(\mathbf{w})}{p(\mathbf{w} \vert \mathbf{X})} d\mathbf{w}\newline 
& = \mathcal{L}(\mathbf{X}) + \operatorname{KL}(q\vert \vert p) 
\end{align}
$$
ここで１行目は
$$
\int_{\mathbf{w}} q(\mathbf{w}) d\mathbf{w} = 1
$$
を使っています。

ここで$\mathcal{L}(\mathbf{X})$は、我々が仮定する関数$q(\mathbf{w})$と確率モデル$p(\mathbf{X}, \mathbf{w})$から構成されているため計算が可能であることに注意してください。周辺尤度$p(\mathbf{X})$は確率モデルと観測データが与えられれば一意的に値が決まる量なので、上の式はこの$\mathcal{L}(\mathbf{X})$を最大化する$q(\mathbf{w})$を求めれば、それが自動的にKLダイバージェンスを最小化する$q(\mathbf{w})$を求めていることになることを示しています。つまり未知の関数を含むKLダイバージェンスを直接最小化できなくても、代わりに$\mathcal{L}(\mathbf{X})$を最大化すればその目的が達成できるのです！

この$\mathcal{L}(\mathbf{X})$をELBO(Evidence Lower BOund)と呼びます。


ここで$q(\mathbf{w})$をなんらかパラメータ$\boldsymbol{\alpha}$で特徴づけられる関数を仮定しているとすると、ELBO$\mathcal{L}(\mathbf{X})$の最大化、言い換えて$-\mathcal{L}(\mathbf{X})$の最小化はパラメータに対する偏微分$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\alpha}}$を求めて勾配降下法で最適化計算を行っていくことができことになります。

以上、変分近似について解説してきました。要点は、
1. 事後確率$p(\mathbf{w}|\mathbf{X})$をパラメータ$\boldsymbol{\alpha}$で特徴づけられる関数（変分関数）$q(\mathbf{w})$で表現する。
2. 求めたい事後確率分布に変分関数を似せる（つまりKLダイバージェンスを最小化する）ことで事後分布を近似的に求める。
3. ただしKLダイバージェンスを直接計算できないので、代わりにELBOを最大化することで最適な変分関数$q_{opt.}(\mathbf{w})$を求める。
4. ELBOの最大化は勾配降下法を用いて行う。
ということになります。

次節以降でPyroで変分近似を行う例を具体的に示していきます。
