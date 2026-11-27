# 工程03 音質調整（暫定値・比較方法）
初期候補は幼児 +14 / index 0.4 / protect 0.25 / RMS 0.75、少女 +12 / 0.4 / 0.25 / 0.75、男性 0 / 0.5 / 0.33 / 0.8、ロボット 0 / 0.7 / 0.33 / 1.0。全て50ms、フォルマント0、参照100ms。RMSは入力音量強制一致による過増幅を減らすための暫定値。作者の確定推奨値ではなく、聴感で最適化済みでもない。
配布元: https://amitaro.net/synth/rvc/ ; https://v2.voice-models.com/creator/VofVoices ; https://huggingface.co/ttttdiva/rvc_okiba/blob/main/vc-samples.md ; https://voice-models.com/model/94r
男性の同名モデル候補のピッチ0使用例あり。同一性は未確定。
ローカルのモデルは再配布しない。比較用スクリプトはインストール済みモデル・同一入力WAVで基準と各パラメーター単独変更を比較する。Wave Linkを終了してから tools/voicebank-benchmark.ps1 を実行する。出力は bundle/measurements 以下。WAV、性能JSON、接続CSV、summary.csvを保存。実機の測定値と聴感評価を合わせて選定する。自動で設定を上書きしない。オフライン結果はWave Linkの実時間保証ではない。
