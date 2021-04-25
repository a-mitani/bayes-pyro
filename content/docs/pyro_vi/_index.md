---
title: Pyroで変分推論
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

#Pyroで変分推論
前節で変分近似はKLダイバージェンスを最小化する代わりにELBOを最大化することで事後確率を近似計算するという旨を解説しました。

しかし実際の確率モデルこのELBOを人の手で計算するのは非常に複雑になり困難です。そこでPyroの出番になります。Pyroはユーザーが確率モデルを定義すればそのモデルに応じたELBOやその偏微分を自動計算してくれる機能を持つためユーザーはELBOの具体的な形を意識せずに変分近似の計算が可能になります。

では具体的にPyroを用いてどのように計算を行っていくのかを見ていきましょう。


これはPytorchなどの深層学習用のフレームワークがユーザーが定義したニューラルネットワークの計算グラフをモデルとして保持し、その計算グラフに合わせて損失関数のパラメータの偏微分を自動計算してくれるのと似ています。Pyroではユーザーが定義する確率モデルを保持し、そのモデルに合わせてELBOやその偏微分を自動計算し勾配降下での最適化を自動で行ってくれるのです。
