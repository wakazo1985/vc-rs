# HTTPコントロール
Wave LinkでVST3が有効になった状態で http://127.0.0.1:18765/control.html を開くか、同梱control.htmlを開く。別アプリは不要。POST /api/voice/original, toddler, girl, male, robot は引数不要。202は選択要求の受理でありモデルのロード成功ではない。ショートカットはHTMLから全6機能を編集・保存可能。エコーは従来通り物理キーのホールド。設定はVST3フォルダ内settings.toml、ログはlogs。ポート固定18765、ループバックのみ、クロスオリジン拒否。複数インスタンスのうち最初にポートを取得したものが対象。ポート競合時はHTTP_ERRORログを確認。モデル読み込みは従来通り必要。
