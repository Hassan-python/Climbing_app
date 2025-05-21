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
- [ ] 🟡 [バックエンド移行] Google Cloud Runへのデプロイ
  - [ ] Dockerfileの確認・修正 (システム依存関係、uvicorn起動コマンド)
  - [ ] GCPプロジェクト設定 (API有効化、Artifact Registry設定)
  - [ ] 既存ChromaDBへの接続設定 (CHROMADB_URL環境変数の設定)
  - [ ] 動画アップロード/分析処理をGoogle Cloud Storage利用に修正
    - [ ] main.py の /upload, /analyze エンドポイント修正
  - [ ] コンテナイメージのビルドとArtifact Registryへのプッシュ
  - [ ] Cloud Runサービスへのデプロイと環境変数設定
  - [ ] デプロイされたAPIエンドポイントの動作確認
  - [ ] フロントエンドからの接続先URL変更

## バグ修正
- [ ] 現在特になし

## 技術的負債
- [ ] 🟢 環境変数管理の改善
  - .envファイルによる設定の外部化検討
- [ ] 🟢 一時動画ファイル (`temp_videos`) のクリーンアップ機構検討 