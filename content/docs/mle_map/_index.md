---
title: MAP推定と最尤推定
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

# MAP推定と最尤推定
前節までで変分近似を用いてベイズ推論を行ってきました。ベイズ推論は観測出来ない潜在パラメータを確率変数と捉え、ベイズの定理を利用して観測データ$\mathbf{X}$からそのパラメータの**確率分布を推定する**ものでした。つまり潜在パラメータの確率変数を$\Theta$とすると下記の関係性を用いて事後分布$p(\Theta|\mathbf{X})$を求めることがベイズ推定です。
$$
p(\Theta|\mathbf{X}) = \frac{p(\mathbf{X}|\Theta)p(\Theta)}{p(\mathbf{X})}\tag{1}
$$

一方でわざわざ確率分布まで求めなくても、確率分布の最も頻度が高くなる一点を簡易的に求めれば事足りる場合も多々あります。そのような手法としてMAP推定や最尤推定と呼ばれる手法があります。本節ではこれらを紹介したうえでPyroでの計算方法を紹介します。

## ■MAP推定
MAP推定のMAPはmaximum a posterioriの略であり、日本語に訳すと最大事後確率推定と呼ばれます。その名のとおり観測データに対して事後分布$p(\Theta|\mathbf{X})$が最大となるパラメータ値を推定する、つまり、
$$
\theta_{MAP} = \underset{\theta}{\operatorname{argmax}}~p(\Theta=\theta|\mathbf{X})=\underset{\theta}{\operatorname{argmax}}~p(\mathbf{X}|\Theta=\theta)p(\theta)\tag{2}
$$
となる$\theta_{MAP}$を求める作業になります。ここで２つ目の等式は(1)式の関係性から容易に導けるでしょう。

さて、これまで変分推論では事後確率に近しいと想定される近似関数$q(\theta)$を用意し、その近似関数の形が本来求めたい事後確率分布に近づくように近似関数のパラメータを最適化しました。その変分推論の枠組みに当てはめたとき、(2)式は近似関数$q(\theta)$をデルタ関数$\delta(\theta-\theta_{MAP})$と仮定して変分推論することとして解釈が可能です。

ここでデルタ関数$\delta(\theta-\theta_{MAP})$は$\theta=\theta_{MAP}$で∞になり、それ以外の$\theta$ではゼロをとる無限に尖った関数であり$\int\delta(\theta-\theta_{MAP})d\theta=1$となります。

以上のとおりMAP推定をデルタ関数を近似関数とした変分推論と解釈できるならPyroでも`guide`関数にデルタ関数を指定してあげれることでMAP推定を行うことができます。実際に前節の赤玉白玉の混合比率を例にPyroでMAP推定してみましょう。

### 試行データ生成
試行結果データを生成するコードは前節と全く同様です。
```python
# 試行データ作成（前節と同一コード）
def create_data(red_num, white_num):
    red = torch.tensor(1.0)
    white = torch.tensor(0.0)
    data = []
    for _ in range(red_num):
        data.append(red)
    for _ in range(white_num):
        data.append(white)
    random.shuffle(data)
    data = torch.tensor(data)
    return data

data = create_data(6, 4)
```

### 最適化ヘルパー関数を定義
後ほど最尤推定でも再利用できるように最適化の一連の処理を関数化します。
```python
# 最適化計算用のヘルパー関数
#　引数として指定されたmodelの関数とguide関数を用いてELBOの最大化を行う
def optimize_param(model_fn, guide_fn):
    # グローバル変数として保存されているパラメータを削除
    pyro.clear_param_store()

    # Optimizerの定義と設定（Adamの利用が推奨されている）
    adam_params = {"lr": 0.001, "betas": (0.95, 0.999)}
    optimizer = Adam(adam_params)

    # 推論アルゴリズムとLoss値を定義
    # ここでは組み込みのELBOの符号反転をLoss値とする`Trace_ELBO()`を利用しています。
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())

    # 最適化の逐次計算
    # ここではAdamで勾配降下を1000回繰り返すことになる。
    n_steps = 1000
    losses = []
    for step in range(n_steps):
        loss = svi.step(data)
        losses.append(loss)
        if step % 100 == 0:
            print('#', end='')

    plt.plot(losses)
    plt.show()
```
### 確率モデルと変分関数
上述のようにMAP推定は変分推論の枠組みで変分関数としてデルタ関数を仮定するものというお話をしました。そのため確率モデルは前節と全く同じでよく、また変分関数を規定する`guide`関数はデルタ分布を用います。
```python
# 確率モデルの定義
def model(data):
    # 事前確率分布は比率0.5に穏やかなピークを持つ関数を仮定する。
    alpha0 = torch.tensor(2.0)
    beta0 = torch.tensor(2.0)
    f = pyro.sample("Theta", dist.Beta(alpha0, beta0))

    # 観測データのプレート定義
    with pyro.plate('observation'):
      pyro.sample('X', dist.Bernoulli(f), obs=data)

# MAP推定や最終推定は変分関数としてデルタ関数を仮定する。
def guide_delta(data):
    theta_opt = pyro.param("theta_opt", torch.tensor(0.5),
                       constraint=constraints.unit_interval)
    pyro.sample("Theta", dist.Delta(theta_opt))
```
確率モデルと変分関数が用意できたら、変分関数のパラメータ（今回の場合`theta_opt`）の最適化を行います。
```python
#　MAP推定用のmodel関数とguide関数を指定して最適化を実施
optimize_param(model_map, guide_delta)

# 最適化後の変分パラメータを取得する
theta = pyro.param("theta_opt").item()
print("theta_opt = {:.3f}".format(theta))
## Output
# theta_opt = 0.583
```
上記の結果、`thata_opt`つまり$\theta_{MAP}$は`0.583`となりました。前節で行ったベイズ推定結果の事後確率分布の最頻値とほぼ同じ値が求まったことに注意してください。これはMAP推定が全体の確率分布までは求めないけど確率分布の最頻値をとる潜在パラメータの値を求める推定方法であることを考えると納得がいくでしょう。

また、試行データを作成する際に、`data = create_data(60, 40)`として、合計100回の試行データを作成すると`theta_opt = 0.599`となります。これは試行回数が増えれば増えるほど、最初に想定していた事前確率分布よりも試行データのほうが重みが増えて試行データの結果の比に近づいていくことを示しています。


## ■最尤推定
単に観測データ$\mathbf{X}$に対する尤度が最大となる$\theta$を求める、すなわち
$$
\theta_{MLE} = \underset{\theta}{\operatorname{argmax}}~p(\mathbf{X}|\Theta=\theta)
$$
となる$\theta_{MLE}$を求めるのが最尤推定（Maximum Likelihood Estimation: MLE）です。

これは(2)式と見比べると潜在パラメータの事前分布を$p(\Theta)$を一定値とした（すなわち事前情報がないとした）MAP推定をしていると解釈することができます。

以上のことからPyroで最尤推定を行うには場合、事前分布を一様分布としてMAP推定を行えばよいことが分かります。確率モデルは下記のように定義することができます。0~1の範囲で一様分布となる`dist.Uniform(0.0, 1.0)`で事前分布が定義されているのに注意してください。
```python
# 最尤推定用に確率モデルの定義（ベイズ推定の際と全く同じことに注意）
def model_mle(data):
    # 事前確率分布は一定値（無情報）として一様分布を指定する
    f = pyro.sample("Theta", dist.Uniform(0.0, 1.0))

    # 観測データのプレート定義
    with pyro.plate('observation'):
      pyro.sample('X', dist.Bernoulli(f), obs=data)
```
上記の確率モデル関数を用いて最適化を以下のように行うことができます。
```python
#　MAP推定用のmodel関数とguide関数を指定して最適化を実施
optimize_param(model_mle, guide_delta)

# 最適化後の変分パラメータを取得する
theta = pyro.param("theta_opt").item()
print("theta_opt = {:.3f}".format(theta))
## Output
theta_opt = 0.600
```

`thata_opt`つまり$\theta_{MAP}S$は`0.600`となりました。

ちなみに`data = create_data(3, 2)`というように試行回数を幾ら減らしたデータを用いても`theta_opt=0.600`という結果は変わりません。これは最尤推定では例えば比率が半分である可能性が高いといったような事前情報を導入していないため、試行データだけをもとに混合比率を推定したことに起因します。その結果「赤玉と白玉がそれぞれ3回と2回取り出された」という結果に過剰適合（Overfitting）した結果になります。



