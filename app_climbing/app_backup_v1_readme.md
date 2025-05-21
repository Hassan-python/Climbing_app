# ボルダリング動画分析API (`app_backup_v1.py`)

## 概要

このアプリケーションは、FastAPIを使用して構築されたボルダリング動画分析APIです。
ユーザーがアップロードしたボルダリングの動画（5秒以内）に対し、AIによる画像分析、関連知識の検索、そして具体的な改善アドバイスを提供します。

## 特徴

- **動画アップロード**: Google Cloud Storage (GCS) と連携し、動画ファイルを安全に保存します。
- **フレーム抽出**: 動画の指定された区間から分析用のフレームを抽出します。
- **AIによる画像分析**: Google Gemini APIを利用して、抽出されたフレームからクライマーの動きや体勢を客観的に分析します。
- **関連知識のベクトル検索**: ChromaDBに格納されたボルダリング知識データベースから、ユーザーの課題やAI分析結果に関連する情報を検索します。
- **アドバイス生成**: OpenAI API (GPTモデル) を使用し、上記の分析結果と検索された知識に基づいて、具体的かつ実践的な改善アドバイスを生成します。
- **多言語対応**: リクエストヘッダー `X-Language` (`ja` または `en`) に基づいて、アドバイスを日本語または英語で提供します。

## APIエンドポイント

### 1. `POST /upload`

ビデオファイルをアップロードし、GCSに保存します。

- **リクエストボディ**:
  - `video`: アップロードするビデオファイル (multipart/form-data)
- **成功レスポンス (200 OK)**:
  ```json
  {
    "gcsBlobName": "videos/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.mp4",
    "videoId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  }
  ```
- **制約**:
  - アップロードするビデオは5秒以内でなければなりません。
- **エラーレスポンス**:
  - `400 Bad Request`: ビデオファイルが提供されていない、またはビデオ長が5秒を超えている場合。
  - `500 Internal Server Error`: GCSバケット名が設定されていない、またはアップロード処理中にエラーが発生した場合。

### 2. `POST /analyze`

GCSに保存されたビデオの分析を要求し、アドバイスを取得します。

- **リクエストヘッダー**:
  - `X-Language` (任意): アドバイスの言語を指定します (`ja` で日本語、`en` で英語。デフォルトは英語)。
- **リクエストボディ** (`AnalysisSettings` モデル):
  ```json
  {
    "problemType": "string (例: 特定のホールドが取れない)",
    "crux": "string (例: 次の一手が遠い、バランスが悪い)",
    "startTime": "float (動画内の分析開始時間(秒))",
    "gcsBlobName": "string (アップロード時に取得したGCSのBlob名)"
  }
  ```
- **成功レスポンス (200 OK)** (`AnalysisResponse` モデル):
  ```json
  {
    "advice": "string (生成されたアドバイス)",
    "sources": [
      {
        "name": "string (関連知識ソースの識別子)",
        "content": "string (関連知識の内容)"
      }
    ],
    "geminiAnalysis": "string (Geminiによる客観的な画像分析結果)"
  }
  ```
- **エラーレスポンス**:
  - `400 Bad Request`: `gcsBlobName` が提供されていない場合。
  - `404 Not Found`: 指定された `gcsBlobName` のビデオがGCSに見つからない場合。
  - `500 Internal Server Error`: GCSバケット名が設定されていない、ChromaDBのURLが設定されていない、または分析処理中にエラーが発生した場合。

### 3. `GET /chroma-status`

ChromaDBの接続状態と、指定されたコレクション内のアイテム数を確認します。

- **成功レスポンス (200 OK)**:
  ```json
  {
    "status": "✅ ChromaDB 接続成功 (`bouldering_advice`: 123 アイテム)"
  }
  ```
- **失敗レスポンス (200 OK - ステータス内容で判別)**:
  ```json
  {
    "status": "❌ ChromaDB connection failed: [エラー詳細]"
  }
  ```

## セットアップと実行方法

1.  **リポジトリのクローン**:
    ```bash
    # git clone ... (リポジトリがある場合)
    ```
2.  **依存関係のインストール**:
    プロジェクトルートで以下のコマンドを実行し、必要なライブラリをインストールします。
    ```bash
    pip install fastapi uvicorn pydantic python-dotenv google-generativeai langchain-openai langchain-community chromadb Pillow google-cloud-storage moviepy opencv-python numpy
    ```
3.  **環境変数の設定**:
    プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定します。
    ```env
    GCS_BUCKET_NAME="your-gcs-bucket-name"
    GEMINI_API_KEY="your-gemini-api-key"
    CHROMA_DB_URL="http://your-chromadb-host:port" # 例: http://localhost:8000
    OPENAI_API_KEY="your-openai-api-key"
    OPENAI_MODEL_NAME="gpt-4.1-nano" # (任意、デフォルトは gpt-4.1-nano)
    ```
4.  **ChromaDBの準備**:
    ChromaDBサーバーを起動し、`bouldering_advice` という名前のコレクションが利用可能であることを確認してください。
5.  **アプリケーションの起動**:
    ```bash
    uvicorn app_climbing.app_backup_v1:app --host 0.0.0.0 --port 8000 --reload
    ```
    `--reload` は開発時に便利です。

## 必要な環境変数

-   `GCS_BUCKET_NAME`: 動画ファイルを保存するGoogle Cloud Storageのバケット名。
-   `GEMINI_API_KEY`: Google Gemini APIを利用するためのAPIキー。
-   `CHROMA_DB_URL`: ChromaDBサーバーのURL。
-   `OPENAI_API_KEY`: OpenAI API (Embeddings, Chat LLM) を利用するためのAPIキー。
-   `OPENAI_MODEL_NAME`: (任意) アドバイス生成に使用するOpenAIのモデル名。指定しない場合は `gpt-4.1-nano` が使用されます。

## 依存ライブラリ

-   fastapi
-   uvicorn
-   pydantic
-   python-dotenv
-   google-generativeai
-   langchain-openai
-   langchain-community (Chroma vector store)
-   chromadb
-   Pillow (PIL)
-   google-cloud-storage
-   moviepy
-   opencv-python
-   numpy

## 注意点・制限事項

-   **動画長**: `/upload` エンドポイントでアップロードできる動画は最大5秒です。
-   **フレーム分析**: Geminiによる画像分析に使用されるフレーム数は、動画の長さや設定に応じて最大10フレームに調整されます。
-   **一時ファイル**: `/upload` 時および `/analyze` 時に、一時的に動画ファイルがサーバーの `/tmp/` ディレクトリにダウンロードされます。処理後には削除されます。
-   **CORS設定**: 現在、CORS (Cross-Origin Resource Sharing) は全てのオリジン (`"*"`) に対して許可されています。本番環境で運用する際は、セキュリティ向上のため、許可するオリジンを具体的に指定してください。
-   **GCS認証**: `google-cloud-storage` ライブラリは、実行環境に応じた認証情報（例:環境変数 `GOOGLE_APPLICATION_CREDENTIALS` やサービスアカウントキー）が設定されている必要があります。 