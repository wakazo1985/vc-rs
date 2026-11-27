# RVC VoiceBank Windows ML
1. vc-vst3.vst3/models/toddler、girl、male、robot にそれぞれ1個のPTHと対応するINDEXを配置。commonにはContentVec、RMVPEのONNXを配置。
2. vc-vst3.vst3フォルダ全体をWindowsのVST3フォルダに配置。設定・モデル生成・ログのため書き込み権限が必要。既存モデルや設定を削除しないこと。
3. Wave Linkにエフェクトを追加。最初のリアルタイム音声処理で自動変換・登録・読み込み。ONNXが既にあれば再変換しない。RVC v2 F0対応のみ。失敗時はbundle内logsを確認。INDEXはL2 Flat／IVFFlat形式（最大512 MiB）に対応し、音声特徴検索に使用。検索結果の混合率の初期値は0.5。未対応形式の読み込みはエラーとして記録されます。
4. F7元の声、F8幼児、F9少女、F10低音男性、F11ロボット、F12ホールドエコー。control.htmlは未接続時にも表示。設定変更は接続画面で行う。URLはhttp://127.0.0.1:18765/control.html。
5. Windows App SDK Runtime 2.1以上が必要。実際のモデルとWave Linkは利用者環境で確認すること。benchmarkフォルダは任意の測定ツール。

INDEX実装の参考: https://github.com/facebookresearch/faiss/blob/v1.8.0/faiss/impl/index_write.cpp および https://github.com/RVC-Project/Retrieval-based-Voice-Conversion/blob/develop/rvc/modules/vc/pipeline.py 。Pythonや外部FAISS DLLは不要。実際のモデルの形式・音質・処理時間は未確認。

配布ZIPは1回解凍するだけです。ソースコードはsourceフォルダに展開済みです。HTMLをローカルファイルから開いた場合、接続リンクから設定画面に移動できます。HTTP APIは同一オリジンのブラウザまたはローカルのネイティブHTTPクライアントから操作可能です。拒否時はHTTP_REJECTログを参照してください。
