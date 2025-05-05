# Climbing Analysis & Advice App

## 概要

このアプリケーションは、アップロードされたクライミング動画を分析し、改善のためのアドバイスを生成する Streamlit アプリケーションです。

動画からキーフレームを抽出し、Google Gemini Vision API を使用してフレーム画像を分析します。その分析結果と、事前に ChromaDB に格納されたボルダリングに関する知識ベース（ナレッジベース）を組み合わせ、ユーザーのパフォーマンスに対する具体的なアドバイスを生成します。

## 主な機能

*   **動画アップロード:** ユーザーはクライミング動画ファイル (mp4, mov, avi) をアップロードできます。
*   **キーフレーム抽出:** 動画から指定された間隔（デフォルト3秒）で代表的なフレームを抽出します。
*   **画像分析 (Gemini Vision):** 抽出された代表フレーム画像を Google Gemini Vision API (gemini-1.5-flash) に送信し、姿勢、動き、改善点などのテキスト記述を取得します。
*   **RAG (Retrieval-Augmented Generation):**
    *   Gemini Vision による分析結果とユーザーの質問（またはデフォルトプロンプト）を組み合わせます。
    *   事前にリモートの ChromaDB にロードされたボルダリング知識ベースから関連情報を検索します。
    *   検索結果と元のプロンプトを統合し、最終的な回答生成のためのプロンプトを作成します。
*   **アドバイス生成:** 最終プロンプトを OpenAI API (GPT-4o mini) に送信し、ユーザーへの具体的なアドバイスを生成・表示します。
*   **デバッグ情報表示:** デバッグモードを有効にすると、ChromaDB の接続ステータスや取得された知識のチャンク数を表示します。

## 技術スタック

*   **Webフレームワーク:** [Streamlit](https://streamlit.io/)
*   **言語モデル:** [OpenAI GPT-4o mini](https://openai.com/), [Google Gemini 1.5 Flash](https://ai.google.dev/)
*   **ベクトルデータベース:** [ChromaDB](https://www.trychroma.com/) (リモートサーバーへ HttpClient で接続)
*   **LLM オーケストレーション:** [LangChain](https://www.langchain.com/)
*   **依存関係管理:** `requirements.txt`
*   **設定管理:** 環境変数, `app_climbing/secrets.yaml` (ローカル実行時), Streamlit Secrets (デプロイ時)
*   **その他:** OpenCV (`opencv-python-headless`), PyYAML

## セットアップ (ローカル)

1.  **リポジトリをクローン:**
    ```bash
    git clone https://github.com/Hassan-python/Climbing_app.git
    cd Climbing_app
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
    ```bash
    pip install -r requirements.txt
    ```

4.  **シークレットファイルの設定:**
    *   `app_climbing` ディレクトリに `secrets.yaml` ファイルを作成します。
    *   以下の内容を参考に、ご自身の API キーと ChromaDB URL を入力します:
        ```yaml
        # app_climbing/secrets.yaml
        openai:
          api_key: "YOUR_OPENAI_API_KEY_HERE"
        chromadb:
          url: "YOUR_CHROMA_DB_URL_HERE" # 例: "https://your-chroma-service-url.run.app"
        google_genai:
          api_key: "YOUR_GOOGLE_GEMINI_API_KEY_HERE"
        ```
    *   **注意:** この `secrets.yaml` ファイルは `.gitignore` に登録されているため、Git リポジトリにはコミットされません。
    *   環境変数 (`OPENAI_API_KEY`, `CHROMA_DB_URL`, `GOOGLE_GENAI_API_KEY`) が設定されている場合は、そちらが優先されます。

## ナレッジベースのセットアップ

アプリケーションが参照する知識ベースをリモートの ChromaDB にロードする必要があります。

1.  **知識データの準備:**
    *   `app_climbing/knowledge_base` ディレクトリを作成します。
    *   このディレクトリ内に、ボルダリングに関する情報を含むテキストファイル (`.txt`) を配置します。
    *   ファイルは UTF-8 エンコーディングで保存してください。

2.  **データロードスクリプトの実行:**
    *   `load_knowledge.py` スクリプトを実行して、`knowledge_base` ディレクトリ内のテキストファイルを読み込み、ベクトル化してリモート ChromaDB に格納します。
    *   **初めてロードする場合や、データを完全に置き換えたい場合:**
        ```bash
        python app_climbing/load_knowledge.py --mode replace
        ```
    *   **既存のデータに新しいドキュメントを追加したい場合:**
        ```bash
        python app_climbing/load_knowledge.py --mode append
        ```
    *   このスクリプトは、ローカル実行時に `app_climbing/secrets.yaml` または環境変数から API キーと ChromaDB URL を読み込みます。

## ローカルでの実行

セットアップとナレッジベースのロードが完了したら、以下のコマンドで Streamlit アプリケーションを起動します:

```bash
streamlit run app_climbing/app.py
```

Web ブラウザで表示されるローカル URL (通常 `http://localhost:8501`) にアクセスします。

## デプロイ (Streamlit Cloud)

このアプリケーションは Streamlit Cloud にデプロイできます。

1.  **必要なファイル:**
    *   `requirements.txt`: Python の依存関係を定義します。
    *   `app_climbing/packages.txt`: OS レベルの依存関係 (例: `build-essential`) を定義します。ChromaDB の一部の依存ライブラリのビルドに必要になる場合があります。
    *   `app_climbing/app.py`: Streamlit アプリケーション本体。

2.  **Streamlit Cloud でのリポジトリ接続:** GitHub リポジトリを Streamlit Cloud アプリに接続します。

3.  **シークレットの設定:**
    *   Streamlit Cloud のアプリケーション設定画面で、以下のシークレットキーを設定します:
        *   `OPENAI_API_KEY`
        *   `CHROMA_DB_URL`
        *   `GOOGLE_GENAI_API_KEY`
    *   アプリケーションは、これらのキーを Streamlit Secrets (`st.secrets`) 経由で読み込みます。

4.  **デプロイ:** 設定が完了したら、アプリケーションをデプロイします。

## 設定

アプリケーションの動作に必要な設定は、以下の方法で行います。

*   **API キーと ChromaDB URL:**
    *   **ローカル実行時:**
        *   環境変数 (`OPENAI_API_KEY`, `CHROMA_DB_URL`, `GOOGLE_GENAI_API_KEY`) (最優先)
        *   `app_climbing/secrets.yaml` ファイル (環境変数がない場合)
    *   **Streamlit Cloud デプロイ時:**
        *   Streamlit Cloud アプリ設定内の Secrets (`st.secrets`)
*   **知識ベース:** `app_climbing/knowledge_base` ディレクトリ内の `.txt` ファイル。
*   **ChromaDB コレクション名:** `app_climbing/load_knowledge.py` および `app_climbing/app.py` 内の `CHROMA_COLLECTION_NAME` 定数 (デフォルト: `bouldering_advice`)。
*   **動画分析間隔:** `app_climbing/app.py` 内の `DEFAULT_ANALYSIS_INTERVAL_SECONDS` 定数 (デフォルト: 3 秒)。 