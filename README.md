# my_esn
拘束条件付きESNの実装テスト

# 実行方法

  `$python3 setup.py`
  
  `$docker start -i my-esn`
  
  `$python3 src/train.py`

  `$tensorboard --logdir=./logs`

# 目的の流れ

  生物的尤度を持つ学習モデルの提唱

  従来手法としてbpttがある←今ここ

  取り敢えずhessian-freeで精度が上がっているのを確認することでアイディアの有用性を確認

# 現状の問題

  精度が低すぎる，lossが下がらない 
  原因
  実装の方式が未だ不十分
  実装の手法を平川さんの実装を参考に学ぶ
  (精度は十分高いようだけど，そうなるとhessian-free法の実装は意味があるの？)

#　実装におけるチェックポイント

  データ・セットと正解データが上手く出来ているか？

  内部状態は正しく保持されているか？

  Lossは正しく溜め込まれ，更新されているか？

  学習率，最適化手法はそれで正しいか？