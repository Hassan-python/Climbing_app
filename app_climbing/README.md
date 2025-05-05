# Climbing Analysis & Advice App

## 概要

このアプリケーションは、アップロードされたボルダリング動画（5秒以内）を分析し、改善のためのアドバイスを生成する Streamlit アプリケーションです。

動画からキーフレームを抽出し、Google Gemini Vision API を使用してフレーム画像を分析します。その分析結果と、事前にリモートの ChromaDB に格納されたボルダリングに関する知識ベースを組み合わせ、ユーザーが入力した「課題の種類」や「難しい点」も考慮して、OpenAI GPT モデルが具体的なアドバイスを生成します。

SQLite の実装として `pysqlite3` を使用するように試みます（環境にインストールされている場合）。

## 主な機能

*   **動画アップロード:** ユーザーはクライミング動画ファイル (mp4, mov, avi, 5秒以内) をアップロードできます。
*   **動画プレビューと分析範囲指定:** アップロードされた動画をプレビューし、分析を開始したい時間を指定できます（デフォルトは動画開始から1秒間）。
*   **キーフレーム抽出:** 指定された開始時間から一定期間（デフォルト1秒）の動画フレームを抽出します。
*   **画像分析 (Gemini Vision):** 抽出された代表フレーム画像（最大10フレーム）を Google Gemini Vision API (`gemini-2.5-pro-preview-03-25`) に送信し、クライマーの動きに関する客観的な観察結果を取得します。タイムアウトは180秒に設定されています。
*   **RAG (Retrieval-Augmented Generation):**
    *   Gemini Vision による分析結果と、ユーザーが入力した課題情報（種類、難しい点）を組み合わせます。
    *   リモートの ChromaDB に格納されたボルダリング知識ベースから関連情報を検索します。
    *   Geminiの分析結果、検索された知識、ユーザー入力を統合し、最終的なアドバイス生成のためのプロンプトを作成します。
*   **アドバイス生成 (OpenAI GPT):** 最終プロンプトを OpenAI API (デフォルト: `gpt-4.1-nano`、Secretsで変更可能) に送信し、ユーザーへの具体的で実践的なアドバイスを生成・表示します。タイムアウトは120秒に設定されています。自信のない表現を避けるようにプロンプトで指示しています。
*   **デバッグ情報表示:** デバッグモードを有効にすると、Secrets のキー、ChromaDB の接続ステータス、Gemini の分析結果、参照した知識ソースを表示します。
*   **エラーハンドリング:** APIキーの不足、動画ファイルの問題、API呼び出しエラーなどを検知し、ユーザーにエラーメッセージを表示します。

## 技術スタック

*   **Webフレームワーク:** [Streamlit](https://streamlit.io/)
*   **言語モデル:** [OpenAI GPT (e.g., gpt-4.1-nano)](https://openai.com/), [Google Gemini 2.5 Pro Preview](https://ai.google.dev/)
*   **ベクトルデータベース:** [ChromaDB](https://www.trychroma.com/) (リモートサーバーへ HttpClient で接続)
*   **LLM オーケストレーション:** [LangChain](https://www.langchain.com/)
*   **動画処理:** [MoviePy](https://zulko.github.io/moviepy/), [OpenCV (`opencv-python-headless`)](https://opencv.org/)
*   **データベース:** [SQLite](https://www.sqlite.org/index.html) (必要に応じて [pysqlite3-binary](https://github.com/coleifer/pysqlite3) を使用)
*   **依存関係管理:** `requirements.txt`
*   **設定管理:** Streamlit Secrets (`secrets.toml`)
*   **その他:** NumPy, Pillow

## セットアップ

1.  **リポジトリをクローン:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **(推奨) 仮想環境の作成と有効化:**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **依存関係のインストール:**
    `app_climbing/requirements.txt` に `pysqlite3-binary` が含まれていることを確認してください（特に Streamlit Cloud など、システム標準の SQLite が古い場合に必要）。
    ```bash
    pip install -r app_climbing/requirements.txt
    ```

4.  **シークレットファイルの設定:**
    *   プロジェクトのルートディレクトリ（または`.streamlit`ディレクトリ内）に `secrets.toml` ファイルを作成します。
    *   以下の内容を参考に、ご自身の API キーと ChromaDB URL を入力します:
        ```toml
        # secrets.toml

        [openai]
        api_key = "YOUR_OPENAI_API_KEY_HERE"
        # model_name = "gpt-4o-mini" # オプション: デフォルト以外を使用する場合

        [gemini]
        api_key = "YOUR_GOOGLE_GEMINI_API_KEY_HERE"

        [chromadb]
        url = "YOUR_CHROMA_DB_URL_HERE" # 例: "https://your-chroma-service-url.run.app"
        ```
    *   **注意:** この `secrets.toml` ファイルは `.gitignore` に登録されているため、Git リポジトリにはコミットされません。

## ナレッジベース (ChromaDB)

このアプリケーションは、事前にベクトル化され ChromaDB に格納されたボルダリングに関する知識を利用します。データはリモートの ChromaDB インスタンスに格納されている必要があります。

*   **コレクション名:** `bouldering_advice` (アプリケーションコード内の `CHROMA_COLLECTION_NAME` 定数で定義)
*   **データの準備とロード:** `app_climbing/knowledge_base` ディレクトリに知識ソースとなるテキストファイルを配置し、`app_climbing/load_knowledge.py` スクリプトを実行して ChromaDB にロードします。詳細は `load_knowledge.py` のコードやコメントを参照してください。

## 実行

セットアップが完了したら、以下のコマンドで Streamlit アプリケーションを起動します:

```bash
streamlit run app_climbing/app.py
```

Web ブラウザで表示されるローカル URL (通常 `http://localhost:8501`) にアクセスします。

## デプロイ (Streamlit Cloud)

このアプリケーションは Streamlit Cloud にデプロイできます。

1.  **必要なファイル:**
    *   `requirements.txt`: Python の依存関係。`pysqlite3-binary` が含まれていることを確認してください。
    *   `app_climbing/app.py`: Streamlit アプリケーション本体。
    *   （必要であれば）`packages.txt`: OSレベルの依存関係（例: `build-essential`）。

2.  **Streamlit Cloud でのリポジトリ接続:** GitHub リポジトリを Streamlit Cloud アプリに接続します。

3.  **シークレットの設定:**
    *   Streamlit Cloud のアプリケーション設定画面 ([Settings] > [Secrets]) で、`secrets.toml` と同じ形式でシークレットを入力します。
        ```toml
        [openai]
        api_key = "YOUR_OPENAI_API_KEY_HERE"
        # model_name = "gpt-4.1-nano" # オプション

        [gemini]
        api_key = "YOUR_GOOGLE_GEMINI_API_KEY_HERE"

        [chromadb]
        url = "YOUR_CHROMA_DB_URL_HERE"
        ```
    *   アプリケーションは、これらのキーを Streamlit Secrets (`st.secrets`) 経由で読み込みます。

4.  **デプロイ:** 設定が完了したら、アプリケーションをデプロイします。

## 注意点

*   動画ファイルは一時的にサーバー（または実行環境）の `temp_videos` ディレクトリに保存されます。このディレクトリのクリーンアップは自動では行われません。
*   Gemini API と OpenAI API の呼び出しにはタイムアウトが設定されていますが、ネットワーク状況やAPI側の負荷によっては時間がかかる場合があります。
*   ChromaDB への接続情報は Secrets から取得されます。URL が正しいか、また ChromaDB サーバーが稼働しているか確認してください。 