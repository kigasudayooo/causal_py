# 機械学習による因果推論について

## EconMLとは
- EconMLは、Microsoft Researchが開発した因果推論のためのPythonパッケージ
- Heterogeneous Treatment Effect Estimation(異質処置効果推定)を行うことができる


 ### 参考資料集
- [EconML](https://econml.azurewebsites.net/index.html)
- [EconMLのサンプルコード](https://github.com/py-why/EconML/tree/main/notebooks)
- [EconMLパッケージの紹介 (meta-learners編)](https://usaito.hatenablog.com/entry/2019/04/07/205756)
- [機械学習で因果推論~Meta-LearnerとEconML~](https://zenn.dev/s1ok69oo/articles/1eeebe75842a50)
- [CATEを推定するMeta-Learnersの特徴と比較](https://saltcooky.hatenablog.com/entry/2020/08/16/003950)
- [Uplift Modelingで介入効果を最適化する](https://qiita.com/usaito/items/af3fa59d0ee153a70350)
  * A/Bテスト (RCT)によって収集された学習データがあることを前提とする効果検証手法
- [めっさ分かりやすい因果推論](https://www.ariseanalytics.com/activities/report/20210409/)
- [異質な因果効果とその推定方法](https://qiita.com/ssugasawa/items/15ca8ae09477c5023c1e#x-learner)
- [BART: Bayesian additive regression treesによる因果推論](https://tmitani-tky.hatenablog.com/entry/2019/11/14/014056)



### Meta-Learners
- EconMLにおける異質処置効果推定の手法

#### T-Learner
- 処置群の結果を予測するモデルと非処置群の結果を予測するモデルの2つのモデルを用意する
$$
\hat{\tau}(x) = \hat{\mu}_{1}(x) - \hat{\mu}_{0}(x) 
$$
ここで、$\hat{\mu}_{0} = E[Y^{(0)} | X = x]$,$\hat{\mu}_{1} = E[Y^{(1)} | X= x]$となっている。つまり、対照群と処置群のそれぞれについて、関心のある共変量Xの下での反応の推定を行い、その結果を比較する。

- T-learnerでは処理群と対照群の観測データをプールして利用していないため、処理群と対照群のそれぞれのデータ生成過程の違いが推定性能に影響を与える。
- 処理群と対照群のそれぞれのデータ生成過程が等しい場合は、不利になる傾向になる。
- 他方で、処置効果の構造が非常に複雑で、処理群と対照群のそれぞれのデータ生成過程に共通の傾向がない場合には、特に優れた性能を発揮する傾向にある。

#### S-Learner
- 処置群の結果を予測するモデルと非処置群の結果を予測するモデルの2つのモデルを用意する
$$
\hat{\tau}(x) = \hat{\mu}(x,1) - \hat{\mu}(x,0) 
$$
ここで、$\mu(x,w) := E[Y^{obs}|X = x, W = w]$となっている。つまり、推定したモデルの対象に対して、wに0/1を代入した差分を考える。
- S-learnerでは、処置変数を他の共変量の同様に扱い、処置変数には特別な役割はない。
- そのため、lassoやRandomForestのようなアルゴリズムは、治療の割り当てを完全に無視して、治療の割り当てを選択しないこともできる。
- シミュレーション結果から、データ生成過程が等しい場合とCATEが多くの場所で0である場合において、最も良い推定を行うことが確認できる。


#### X-Learner
- まず、T-learner同様に$\hat{\mu}_{0} = E[Y^{(0)} | X = x]$,$\hat{\mu}_{1} = E[Y^{(1)} | X= x]$を考える。次に、$\hat{\mu}_{0}$を用いて、処置群の個人の処置を行わない場合の結果の推定を行う。この推定値と、観測された対照における結果の差を、個人の介入効果とする（$\tilde D^{0}_{i}$）。同様に処置群の効果も推定する。

$$
\tilde D^{1}_{i} := Y^{1}_{i} - \hat{\mu}_{0}(X^{1}_{i}) \\
\tilde D^{0}_{i} := Y^{1}_{i} - \hat{\mu}_{0}(X^{1}_{i})
$$

- そして介入群のみからなるデータと、対照群からなるデータそれぞれを用いて、介入効果を推定するモデルを作成する（二段階目のベース学習器）。

$$
\hat{\tau}_{0} = E[\tilde D^{0}_{i}  | X = x]\\
\hat{\tau}_{1} = E[\tilde D^{1}_{i}  | X= x]
$$

- 最後に、得られたベース学習器について、傾向スコア($g(x)$)を用いた重み付き平均を求めることで、介入効果を推定する。
$$
\hat{\tau}_{X}(x) = g(x)\hat{\tau}_{0}(x) + (1-g(x))\hat{\tau}_{1}(x)
$$

- X-learnerは，CATEに構造的な仮定がある場合や，一方の処置群が他方の処置群よりもはるかに大きい場合に特に優れた性能を発揮する。
- 期待されるCATEがほとんど0であるという強い信念がない限り、小さなデータサイズの場合はBARTを用いたX-learnerを、大きなデータサイズの場合にはRandomForestを用いるべきであるとしている。


#### R-Learner([githubのページ](https://github.com/xnie/rlearner))
- [Nie and Wager (2021)](https://arxiv.org/abs/1712.04912)によって提案された方法
- HTEを推定するために、ロビンソン分解（[Robinson (1988)](https://www.jstor.org/stable/1912705)）を用いる
  * [Kaddour et al.(2021)](https://arxiv.org/abs/2106.01939)もまた参考文献としてありそう。
$$
Y_i - m(X_i) = (T_i - g(X_i)) \tau(X_i) + \varepsilon_i
$$
ここで$m(X_i) = E[Y_i|X_i]$はアウトカムの条件付き期待値で、$g(X_i)$は傾向スコア。
これらの$m(X_i)$及び$g(X_i)$を用いて、R-learnerは以下の式を最小化するときの$X_i$を用い、介入効果($\tau(X_i)$)を推定する。なお、$\varLambda_n(\tau(\cdot))$は正則化項。
$$
\sum_{i=1}^{n} [(Y_i - m(X_i)) - (T_i - \pi(X_i)) \tau(X_i)]^2 + \varLambda_n(\tau(\cdot)) \\
$$
- 実務上は$m(X_i)$及び$g(X_i)$は道なので、観測データから推計する。手法を提案した研究では、$m(X_i)$及び$g(X_i)$の推定と$\tau(X_i)$の推定は、それぞれ元のデータセットを分割した、別々のデータセットを用いて行うことを提案している(cross-fitting)。

#### その他の手法
- DA-Learner (Domain Adaptation Learner)
  * DA-Learnerは, X-Learnerにおける$μ^0$,$μ^1$の学習に共変量シフトを用いた手法
- DR-Learner (Doubly Robust Learner)
  *  Doubly Robustを用いてCATEを代替するようなsurrogate outcomeを作り, それをXに回帰する方法
- [Targeted Maximum Likelihood Estimation:TMLEについて](https://saltcooky.hatenablog.com/entry/TLME22)
  * なんじゃこりゃ。super learnerとかいうのもあるらしい。


### Meta-Leanerの利点
- 特定のベース学習器に依存しない
  * 線形モデルではなくても良いことが利点。
  * 他方で、ベース学習器の選択をどのように行うかが重要となる。
  * 以下のような特徴がある、らしい。。。
    - データの生成構造が大域的に線形である状況やデータセットが小さい場合には、[BART](https://tmitani-tky.hatenablog.com/entry/2019/11/14/014056)のように大域的に作用する推定器が大きな優位性を持つ。
      * $Y = \sum_{i=1}^{m} g(x:T,M) + \varepsilon, \varepsilon ∼
N(0,ρ)$というm個の木による加法的予測モデル（これを森と呼ぶ）をベイズで求めることを考える。初期値は適当なpriorを考え、森を作成したのちに木を一つずつMCMCで更新していく。
      * これが収束したら、事後分布に基づいたK個の森をサンプリングして、予測する際はK個の森を使うことで予測値の事後分布を得る。
      * 基本的には因果推論として特別なことをするわけではなく，treatmentも他のcovariateと同列に扱った予測モデルを構築し，その上でtreatment Z={0,1}とした予測値の差で因果効果を推定するアプローチ
      * authorのHillは，事前の知識や因果構造についての推論などをモデル化に活用すると，予測していなかった重要な結果をマスクしてしまったり，推定にバイアスを生んでしまう，という立場を取っている．なので事前知識に応じた既知の因果構造を取り込んだpriorを設定するといった方法論の記載は論文内にない．
    - 大域的な構造がない場合やデータセットが大きい場合には、Random Forestのような高次元の交互作用を用いることができるモデルが有利になる。
      * 結局これまでやってきたような介入効果は、簡単な構造でより効率的に（バリアンスにつ強い）推計をできる一方で、より複雑なデータの生成過程がある場合は、今回のようなより複雑なモデルを用いる必要があるということ？
      * 線形回帰モデルのあてはめは構造がシンプルすぎるのが問題ということらしい。概念としてはS-learnerが線形回帰モデルによる因果推論を内包していると考えられる？

![シミュレーションの設定](https://cdn-ak.f.st-hatena.com/images/fotolife/s/saltcooky/20200815/20200815162631.png)
![シミュレーションの結果](https://cdn-ak.f.st-hatena.com/images/fotolife/s/saltcooky/20200815/20200815011615.png)

## Double/Debiased Machine Learningについて

### 参考資料
- [Double/Debiased Machine Learning for Treatment and Structural Parameters](https://arxiv.org/abs/1608.00060)

### 概要