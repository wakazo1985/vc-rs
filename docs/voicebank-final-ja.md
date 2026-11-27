RVC VoiceBank Windows ML — 使い方
================================

【1. 最初に】
ZIPは1回だけ解凍します。「すべて展開」を選び、展開先にあるこの説明書を開いてください。
Windows 64ビット版、Wave Link 3、Windows App SDK Runtime 2.1以上が必要です。
Python、DAW、仮想オーディオケーブル、Stream Deckは不要です。
この配布物には学習済みモデル、ContentVec、RMVPEは含まれません。手持ちのデータを使います。
ソースコードは別の開発用配布物です。通常の利用には不要です。

【2. モデルを入れる】
手持ちのモデルのZIPを解凍してください。各キャラのフォルダに、対応するPTHとINDEXを1個ずつ配置します。ファイル名は元のままで構いません。
下記の例は配置場所を示しています。例示のファイル名に変更する必要はありません。
  vc-vst3.vst3
    models
      toddler   幼児 / Maxwell Young
        *.pth
        *.index
      girl      少女 / あみたろ・るんるん
        *.pth
        *.index
      male      低音男性
        D.pth
        D(1).index
      robot     ロボット / Speak & Spell
        *.pth
        *.index
      common
        contentvec.onnx
        rmvpe.onnx
commonの共有モデルは必須です。以前のファイル contentvec-f.onnx と rmvpe_20231006.onnx を2つとも入れてください。ファイル名を変える必要はありません。4キャラのPTH/INDEXだけではRVC変換できません。
キャラモデルはRVC v2・F0ありの対応形式に限ります。INDEXはFAISS L2 Flat／IVFFlat、512 MiB以下。元のPTH/INDEXは削除しません。
ONNXをすでに所有している場合、対応するキャラフォルダにtoddler.onnx、girl.onnx、male.onnx、robot.onnxを配置できます。
同じキャラフォルダに複数のPTHやINDEXを混在させないでください。

【3. VST3をWave Linkに登録】
Wave Linkを終了します。
vc-vst3.vst3フォルダを、中のContents、models、toolsごとコピーします。中のVST3ファイルだけをコピーしないでください。フォルダ名も変更しないでください。
一般的なWindows VST3フォルダ:
  C:\Program Files\Common Files\VST3
エクスプローラーのアドレス欄に入力してください。Wave Linkが認識するVST3フォルダを使用します。
このプラグインは自身のフォルダ内にモデル・設定・ログを書き込みます。実行ユーザーが書き込めることが必要です。Program Files配下で書き込めない場合は、認識される書き込み可能な場所を使用するか、対象のプラグインフォルダだけのアクセス権を確認してください。Wave Linkを管理者実行することを前提にしていません。
旧版がある場合は、モデル・settings.toml・logsを退避してから更新してください。旧版と新版を同時に有効にしないでください。
Wave Linkを起動し、マイク入力のオーディオエフェクトからプラグインを追加します。VC-RS RVCを選び、エフェクトを有効にします。表示名やスキャン画面はWave Link側のバージョンにより異なります。
最初のリアルタイム音声処理でモデルの検出・変換・読み込みが行われます。初回は変換に時間がかかる場合があります。既にONNXがある場合は再変換しません。
プラグインの設定画面で読み込み状態を確認してください。読み込まれない場合はWave Linkを終了してモデルの配置を確認し、再起動してください。必要に応じてLoad / Reloadで登録済みモデルを読み込み直せます。
通常利用でtools内のEXEを手動起動する必要はありません。

【4. ショートカット】
F7   元の声（変換なし）
F8   幼児
F9   少女
F10  低音男性
F11  ロボット
F12  エコー（押している間だけ）
F7～F11は1回押すと選択が切り替わります。F12を離すとエコーが止まり、選択中の声に戻ります。
保存済みのカスタムショートカットがある場合は、その設定が優先されます。

【5. HTML設定画面】
解凍したフォルダのcontrol.htmlをダブルクリックしてください。
ローカルファイル表示では初期ショートカットとHTTP URLの確認ができます。
Wave Link内でVST3を有効にしてから「Wave Link起動後、接続画面を開く」をクリックしてください。
またはブラウザのアドレス欄に次のURLを直接入力します。
  http://127.0.0.1:18765/control.html
「接続中」を確認します。
キャラクター切替ボタンを押すと、選択要求が送信されます。
ショートカット設定ではキーを選び、必要に応じてCtrl、Alt、Shift、Winを選択し、「設定を保存」を押します。
「現在の設定を取得」でVST3に保存されている設定を再表示します。
同じキーの重複登録はできません。設定保存に失敗した場合は画面下部のメッセージを確認します。

【6. たぬえさ3などのHTTP連携】
送信方法はPOST、本文は空欄です。URLはcontrol.html内でもコピーできます。
元の声   http://127.0.0.1:18765/api/voice/original
幼児     http://127.0.0.1:18765/api/voice/toddler
少女     http://127.0.0.1:18765/api/voice/girl
低音男性 http://127.0.0.1:18765/api/voice/male
ロボット http://127.0.0.1:18765/api/voice/robot
エコーはF12ホールド操作のみです。HTTPによるエコー制御はありません。
HTTP 202は選択要求の受付であり、実際の音声変換成功を保証するものではありません。
ブラウザのアドレス欄にAPI URLを入れるだけではGETになるため切り替わりません。
WindowsのコマンドプロンプトでHTTPの受付を確認する例:
  curl.exe -i -X POST http://127.0.0.1:18765/api/voice/girl
接続状態だけ確認する場合:
  curl.exe http://127.0.0.1:18765/api/state
ローカルのネイティブHTTPクライアント、または設定画面からの操作を許可します。外部サイトからのブラウザAPI操作は拒否されます。

【7. 設定とログの場所】
実際にWave Linkに登録したvc-vst3.vst3フォルダの直下にsettings.tomlとlogsが作成されます。
ダウンロードして解凍した元フォルダではなく、実際のインストール先を確認してください。
  MODEL_CONVERSION  PTH変換結果
  MODEL_REGISTERED  キャラ登録
  INDEX_REGISTERED  INDEX認識
  INDEX_LOADED      INDEX読み込み
  MODEL_LOAD        モデル読み込み
  HTTP_START        HTTP待受開始
  HTTP_VOICE        キャラ切替要求
  HTTP_REJECT       HTTP拒否理由
  ERROR             エラーの詳細
変換ツールのstdout/stderrもlogs内に生成されます。
詳細ログを確認する場合はプラグイン設定の診断機能を利用します。

【8. トラブル対処】
・設定画面にアクセスできない: Wave Link内のプラグインが有効か確認し、実際のインストール先logsのHTTP_START、HTTP_ERROR、HTTP_REJECTを確認。同じプロセス内の複数インスタンスはポート18765を共有します。別プロセスや旧版がポートを使用している場合は終了してください。
・キャラの声が出ない: control.htmlの読み込み済み数とエラーを確認。COMMON_MODEL_MISSINGがあれば共通ONNXをmodels/commonに配置。未準備のキャラでは元の声へフォールバックします。新規設定のチャンクは200ms・クロスフェード40ms。以前の500ms/85ms標準設定も移行します。実際の遅延は推論負荷によって変わります。
・変換に失敗: PTHがRVC v2・F0ありか、重複ファイルがないか、フォルダ書き込み権限とconversionログを確認。
・INDEX読み込み失敗: 対応形式、512 MiB以下、モデルの特徴次元との一致を確認。
・モデルを変更する: Wave Linkを終了し、旧PTH/INDEX/ONNXを別フォルダへ退避して新しいデータを配置。古いONNXがあると再変換されません。再起動後にログを確認。
・エフェクトが一覧にない: VST3フォルダの位置、Contents内のモジュール、Windows App SDK Runtimeの有無、Wave Link側のスキャン結果を確認。
・設定保存失敗: 実際のVST3フォルダへの書き込み権限を確認。

【9. 音質・負荷の測定（任意）】
通常使用にはbenchmarkフォルダは不要です。
START_VOICEBANK_BENCHMARK.cmdを実行し、画面に従って音声WAVとモデルを指定すると測定結果が作成されます。

【10. 検証状況】
Windowsのビルド環境でのテストと、実際のWave Link上の動作・音質検証は別です。
実際の手持ちの4モデルとの互換性、GPU負荷、Wave Link・たぬえさ3実機連携は未確認です。
INDEX混合率の初期値は0.5です。外部FAISS DLLやPython常駐は不要です。
