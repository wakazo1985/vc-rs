# 新規制作 工程01 — モデル仕様書（公開配布物解析）
対象：ユーザー保有の4モデルとファイル名・容量が一致する公開配布物。原本のSHA-256は未取得なので同一性は確定しない。モデル重みの転送・変換・再配布はしていない。公開ファイルの部分読み取りによるメタデータ解析のみ。旧vc-rsの実装は採用しない。
## 解析証跡
- https://github.com/wakazo1985/vc-rs/actions/runs/46041242448 （全4件のFULL_META、TENSOR_COUNT、KEY_SHAPES、INDEX_META）
- https://github.com/wakazo1985/vc-rs/actions/runs/46040125854 （るんるんREADME、公開D照合）
- https://github.com/wakazo1985/vc-rs/actions/runs/46040265909 （ロボットZIP照合）

## モデル別
|項目|幼児 Fantastic Max|少女 るんるん|D（女性モデル）|Speak and Spell|
|---|---|---|---|---|
|ZIP/PTHサイズ bytes|ZIP 160446052|ZIP 365440882|PTH 55232523|ZIP 96253214|
|PTH| maxwellyoung(FantasticMax)_250e_19250s.pth|Amitaro_runrun_V.1.0_499e_22455s_best_epoch.pth|D.pth|speakandspell2333333.pth|
|INDEX|maxwellyoung(FantasticMax).index|Amitaro_runrun_V.1.0.index|D.index (454130739 bytes)|added_IVF309_Flat_nprobe_1_v2.index|
|PTH sr|48000|40000|40000|40000|
|version/f0|v2/true|v2/true|v2/1|v2/1|
|epoch|250|499|250|300|
|step|19250|22455|記録なし|記録なし|
|dataset length|11:41|35:47|モデルカードによれば概ね1時間、個別値なし|記録なし|
|embedder|contentvec（明記）|contentvec（明記）|記録なし、標準768次元経路で検証|記録なし、標準768次元経路で検証|
|vocoder|HiFi-GAN（明記）|HiFi-GAN（明記）|記録なし、NSF構造と公式デフォルトからHiFi-GAN候補|同左|
|学習F0方式|RMVPE（配布ページ）|PTH記録なし、作者の学習方式の一次資料で追加確認|記録なし|記録なし|
|テンソル数|457|457|457|457|
|パラメータ要素数|28716994|27537346|27537346|27537346|
|INDEXベクトル数|36532|106899|143759|12057|
|nlist|936|2741|3686|309|
|nprobe|1|1|1|1|

全INDEX: IndexIVFFlat、768次元、学習済み、二乗L2、量子化器 IxF2 (IndexFlatL2)、1ベクトル3080 bytes（float32特徴3072+ID8）。公式Applioリアルタイムの検索k=8、逆距離二乗重みを正規化しindex_rateで混合。実データ全体の破損検査・検索実行は未実施。
ロボットZIPのtotal_fea.npyは学習・再構築向け素材で、通常のINDEX読み込み推論には別途不要。
Dの配布元は女性モデルと明記。低音男性ではない。ユーザーの指定なしにman-D等へ置換しない。

## PTH configの全18項目
|項目|幼児|少女/D/ロボット|
|---|---|---|
|spec_channels|1025|1025|
|segment_size|32|32|
|inter_channels|192|192|
|hidden_channels|192|192|
|filter_channels|768|768|
|n_heads|2|2|
|n_layers|6|6|
|kernel_size|3|3|
|p_dropout|0|0|
|resblock|"1"|"1"|
|resblock_kernel_sizes|[3,7,11]|[3,7,11]|
|resblock_dilation_sizes|[[1,3,5],[1,3,5],[1,3,5]]|同左|
|upsample_rates|[12,10,2,2]|[10,10,2,2]|
|upsample_initial_channel|512|512|
|upsample_kernel_sizes|[24,20,4,4]|[16,16,4,4]|
|spk_embed_dim|109|109|
|gin_channels|256|256|
|sr|48000|40000|

テンソルグループ: enc_p 113、dec 243、flow 100、emb_g 1（全4モデル）。主要形状: enc_p.emb_phone.weight [192,768]、enc_p.emb_pitch.weight [256,192]、emb_g.weight [109,256]、dec.conv_pre.weight [512,192,7]、dec.conv_post.weight [1,32,7]、dec.m_source.l_linear.weight [1,1]。主要テンソルはHalfStorage。109は埋め込みテーブル行数で、109人が学習済みという意味ではない。幼児/少女はspeakers_id=1。基準推論では公式のsid=0を検証する。デコーダのアップサンプリング積は幼児480、他400。10msのF0フレームに対応する。学習用enc_q重みは含まれない。学習用設定と推論時調整値は別。

## 推論設定 — 出典別、確定値ではない
- Applioファイル変換API既定: pitch 0、index_rate 0.75、volume_envelope 1.0、protect 0.5、RMVPE、contentvec、hop_length 128（RMVPEの実フレーム間隔とは別）、sid 0、追加エフェクトOFF。
- Applioリアルタイムvoice_conversion関数既定: pitch 0、index_rate 0.5、protect 0.5、volume_envelope 1.0、autotune=false、proposed_pitch=false。UI初期値とは区別。
- RVCリアルタイムの既定は実行バージョンに依存するため基準実行時にコミット固定・実際の値を記録する。
- D作者の男声VC例 pitch +15、別の発話例 index 1.0、protect 0.33、Harvest。歌唱例 index 0.75。同一試行の設定と扱わない。
- 幼児/少女/ロボットの作者固定推奨値は確認できない。pitch等の試験候補は工程02/03で原音と出力を比較して決定する。現段階では固定値を書き込まない。
- RMVPEの公式Applioリアルタイム呼び出し閾値0.03、F0 50–1100Hz、16000Hz/160 samples。推論時F0方式は学習時方式とは独立。
- 10/20msの処理単位は工程06で実測する目標、総遅延や音質の保証ではない。

## 互換性
Applio Synthesizerはcheckpoint config、version、f0、vocoderを使う。HiFi-GAN+F0はNSFデコーダ。RVC公式v2+F0もSynthesizerTrnMs768NSFsidを選ぶ。4モデルは構造上の候補に一致するが、実際のstate_dictロード（missing/unexpected keys）、同一入力での推論は未実施。基準はApplioで学習した2モデルをApplio経路、D/ロボットを標準RVC経路として比較する。モデル形式変換は不要、実行方式は工程04で確定。

## 利用条件
- Fantastic Max: https://new.voice-models.com/model/9TZ 作者はクレジットを要求。第三者キャラクターの権利まで許諾するものではない。
- るんるん: https://amitaro.net/synth/rvc/ および https://amitaro.net/voice/voice_rule/ 。クレジット付き商用/配信可、モデル再配布不可、ソフト組み込みは事前相談。作者の禁止用途を遵守。
- D: https://huggingface.co/ttttdiva/rvc_okiba CreativeML Open RAIL-M表記、自己責任。 https://huggingface.co/ttttdiva/rvc_okiba/blob/main/vc-samples.md
- Speak and Spell: https://huggingface.co/analogspiderweb/analogspiderweb_RVC リポジトリのOpenRAIL表記。第三者権利と個別モデルの利用条件は別途確認が必要。
- いずれも公開ZIPにモデルを同梱しない。

## 参照ソース
- https://github.com/IAHispano/Applio/blob/main/rvc/realtime/pipeline.py
- https://github.com/IAHispano/Applio/blob/main/rvc/infer/infer.py
- https://github.com/IAHispano/Applio/blob/main/rvc/lib/algorithm/synthesizers.py
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/rtrvc.py
- https://github.com/facebookresearch/faiss/blob/v1.7.4/faiss/impl/index_read.cpp

## 残事項
原本SHA-256照合はライブラリの実ファイルへのアクセスが必要。GitHubはユーザーのライブラリに直接アクセスできず、大容量モデルを移送しない制約を維持。公開配布物のファイル名・サイズは一致、原本と同一と断定しない。PTH記録のmodel_hashはファイル全体のSHA-256とは別。Dとロボットの学習時特徴抽出器・F0方式・ボコーダーの明示メタデータがない。公開されていない設定は未記載として保持し、工程02/03で音声の検証により決める。工程01の公開メタデータ調査は完了、原本照合は未完了。