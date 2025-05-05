# Climbing App プロジェクト

## 実装済み機能
- [x] ChromaDB用Dockerfileの作成
  - chromadb/chroma:latestイメージを使用
  - データ永続化のためのボリューム設定
  - ポート8000の公開設定
- [x] 🔴 app_climbing アプリケーションのドキュメント更新
  - README.md の更新
- [x] 🔴 app_climbing アプリケーションの現状を Git に反映
  - README.md, app_climbing/app.py のコミットとプッシュ
- [x] refactor: README.md と todo.md を app_climbing ディレクトリに移動

## 実装予定機能
- [ ] 🟡 Docker Composeファイルの作成
  - ChromaDBサービスの設定
  - データボリュームのマウント設定
- [ ] 🟢 ChromaDBへのデータロード機能の改善/確認 (`load_knowledge.py`)
  - README との整合性確認
  - 必要に応じてスクリプトの改善
- [ ] 🟡 `requirements.txt` の整理
  - ルートにあった不要なファイルを削除 (実施済み)
  - `app_climbing/requirements.txt` の内容精査

## バグ修正
- [ ] 現在特になし

## 技術的負債
- [ ] 🟢 環境変数管理の改善
  - .envファイルによる設定の外部化検討
- [ ] 🟢 一時動画ファイル (`temp_videos`) のクリーンアップ機構検討 