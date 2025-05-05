# 🧗 ボルダリング動画分析＆アドバイス アプリ

このアプリケーションは、アップロードされたボルダリング動画の一部を分析し、AIが改善のためのアドバイスを提供する Streamlit Web アプリケーションです。

## ✨ 主な機能

*   **動画アップロード:** MP4, MOV, AVI 形式の動画ファイルをアップロードできます。
*   **分析範囲選択:** スライダーを使って、動画内の分析したい5秒間の開始位置を指定できます。
*   **課題情報入力 (任意):** 分析の精度向上のため、課題の種類や難しいと感じるポイントを入力できます。
*   **フレーム抽出・表示:** 指定された範囲から一定間隔でフレームを抽出し、画面に表示します（90度回転）。
*   **AIによるアドバイス生成:** 抽出されたフレーム情報（現在はフレーム数のみ活用）とユーザーの入力に基づき、LangChain と OpenAI の LLM (GPT-4 Nano) を使用して RAG (Retrieval-Augmented Generation) パイプラインを実行し、アドバイスを生成します。
*   **ChromaDB連携:** アドバイス生成の知識ベースとして、外部で動作する ChromaDB ベクトルデータベースに接続します。
*   **デバッグモード:** アドバイス生成時に参照された知識ソース（ChromaDB から取得したドキュメント）を表示するデバッグモードがあります。

## 🛠️ 技術スタック

*   **フロントエンド:** Streamlit (`streamlit==1.37.0`)
*   **バックエンド/コアロジック:** Python
*   **LLM/Embeddings:** OpenAI (GPT-4 Nano, OpenAI Embeddings via `langchain-openai`, `openai`, `tiktoken`)
*   **AIオーケストレーション:** LangChain (`langchain`, `langchain-community`, `langchain-core`)
*   **ベクトルデータベース:** ChromaDB (`chromadb==1.0.8`) - HTTP クライアント経由で接続。別途サーバーインスタンスの実行が必要。
*   **動画処理:** MoviePy (`moviepy==1.0.3`), OpenCV (`opencv-python-headless`), Imageio (`imageio`, `imageio-ffmpeg`), NumPy (`numpy`)
*   **データベース互換性:** `pysqlite3-binary==0.5.4` (Streamlit Cloud 環境などでの `sqlite3` バージョン問題を回避するため)
*   **Webフレームワーク (依存関係):** FastAPI (`fastapi==0.115.9` - ChromaDB の依存関係)
*   **デプロイ環境 (想定):** Streamlit Cloud
*   **システム依存関係 (Streamlit Cloud):** `ffmpeg`, `build-essential` (`packages.txt` で指定)

## 🚀 セットアップ & 実行

### 前提条件

*   Python (3.9 以降推奨)
*   Git

### 手順

1.  **リポジトリをクローン:**
    ```bash
    git clone https://github.com/Hassan-python/Climbing_app.git
    cd Climbing_app/app_climbing
    ```

2.  **Secrets の設定:**
    プロジェクトルート (`app_climbing` ディレクトリ) に `.streamlit` ディレクトリを作成し、その中に `secrets.toml` ファイルを作成します。以下の内容を記述し、実際のキーと URL に置き換えてください。

    ```toml
    # .streamlit/secrets.toml

    [openai]
    api_key = "sk-YOUR_OPENAI_API_KEY"

    [chromadb]
    url = "YOUR_CHROMA_DB_HTTP_URL" # 例: "https://your-chroma-service-xxxx.run.app"
    ```

3.  **依存関係のインストール:**
    ```bash
    pip install -r requirements.txt
    ```
    *(注意: `opencv-python-headless` など、環境によっては追加のシステムライブラリが必要になる場合があります)*

4.  **アプリケーションの実行:**
    ```bash
    streamlit run app.py
    ```
    Web ブラウザで表示されたローカルアドレス (通常 `http://localhost:8501`) にアクセスします。

### Streamlit Cloud へのデプロイ

1.  リポジトリを GitHub にプッシュします。
2.  Streamlit Cloud ([https://share.streamlit.io/](https://share.streamlit.io/)) にアクセスし、"New app" からリポジトリ (`Hassan-python/Climbing_app`)、ブランチ (`main`)、メインファイル (`app_climbing/app.py`) を指定してデプロイします。
3.  Streamlit Cloud のアプリケーション設定画面で、`secrets.toml` と同様の内容を Secrets として設定します。
4.  `requirements.txt` と `packages.txt` が Streamlit Cloud によって自動的に読み込まれ、依存関係がインストールされます。

## 📝 注意点

*   **ChromaDB サーバー:** このアプリケーションは、実行中の ChromaDB サーバーインスタンスが別途必要です。`secrets.toml` で指定された URL に、アクセス可能な ChromaDB サーバーが起動している必要があります (例: Google Cloud Run などでデプロイ)。
*   **知識ベース:** アドバイスの質は ChromaDB に格納されている知識ベースの内容に大きく依存します。現状、知識ベースの構築・更新はこのアプリケーションの範囲外です。
*   **API コスト:** OpenAI API の利用にはコストが発生します。
*   **フレーム分析:** 現在、AI アドバイス生成には抽出されたフレームの「数」のみが情報として渡されています。将来的にはフレームの内容自体を分析に含めることで、より精度の高いアドバイスが期待できます。
*   **SyntaxWarning:** `moviepy` に関する `SyntaxWarning` がログに出力されることがありますが、これは `moviepy` 内部の問題であり、通常はアプリケーションの動作に影響しません。