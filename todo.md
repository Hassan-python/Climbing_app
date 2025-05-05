# ChromaDBプロジェクト

## 実装済み機能
- [x] ChromaDB用Dockerfileの作成
  - chromadb/chroma:latestイメージを使用
  - データ永続化のためのボリューム設定
  - ポート8000の公開設定

## 実装予定機能
- [ ] 🟡 Docker Composeファイルの作成
  - ChromaDBサービスの設定
  - データボリュームのマウント設定
- [ ] 🟢 ChromaDBへのデータロード自動化スクリプト
  - 初期データの準備と投入プロセス
- [x] 🔴 app_climbing アプリケーションのドキュメント更新
  - README.md の更新 (進行中 -> 完了へ)
- [x] 🔴 app_climbing アプリケーションの現状を Git に反映
  - README.md, app_climbing/app.py のコミットとプッシュ (進行中)

## バグ修正
- [ ] 現在特になし

## 技術的負債
- [ ] 🟢 環境変数管理の改善
  - .envファイルによる設定の外部化 