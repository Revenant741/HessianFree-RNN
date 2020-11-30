# HessianFree_RNN
HessianFreeをRNNに適応することを目的にしています

# 実行方法

  `$python3 setup.py`
  
  `$docker start -i my-esn`
  
  通常のRNNの実行
  `$python3 src/train.py`
  hessianfree法のRNN適応の実行
  `$python3 src/hessian_train.py`

# 目的の流れ

  ESNのリザバー層の最適化に遺伝的アルゴリズムは不適格

  すべての層を含めた一次収束で精度を出す

  すべての層を含めた二次収束で精度を出す←今ここ

# 現状の問題

1:Hessian-free法の適応が上手く行かない
  torch.autograd.grad()の地点で，10回の誤差に対して10回の出力を微分してもらうために，
  allow_unused=Trueを引数にして無理やり動かすと，出てくる勾配が0になってしまう．
  かといって，各誤差に対して各出力を微分してもらうようにすると2epoch以降はlossがnanになる．
  おそらくガウスニュートン行列を求める部分が正しく作動してない？
  
2:RNNを普通に動かすだけで
  精度は十分高いようだけど，そうなるとhessian-free法の実装は意味があるのか？


#　実装におけるチェックポイント

  データ・セットと正解データが上手く出来ているか？

  内部状態は正しく保持されているか？

  Lossは正しく溜め込まれ，更新されているか？

  学習率，最適化手法はそれで正しいか？