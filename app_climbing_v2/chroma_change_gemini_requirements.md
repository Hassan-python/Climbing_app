# ChromaDB知識ベースローダースクリプト改修要件定義書 (`chroma_change_gemini_requirements.md`)

## 1. 背景・目的

現在の `app_climbing_v2/main.py` (以下、メインAPI) では、RAG (Retrieval Augmented Generation) のためのEmbeddingモデルとしてGoogle Geminiの `models/embedding-001` (768次元) を使用している。
一方、ChromaDBへの知識ベースのデータ投入スクリプト (`app_climbing/load_knowledge.py`、以下、ローダースクリプト) は、OpenAIのEmbeddingモデル (例: `text-embedding-ada-002`、1536次元) を使用している。

これにより、メインAPIがChromaDBに接続した際に「Collection expecting embedding with dimension of 1536, got 768」というエラーが発生し、RAG機能が正常に動作しない。

本改修の目的は、ローダースクリプトを修正し、メインAPIと一貫性のあるGemini Embeddingモデル (`models/embedding-001`) を使用してChromaDBにデータを投入できるようにすることで、次元の不一致エラーを解消し、RAG機能の正常化を図る。

## 2. スコープ

本改修では、`app_climbing/load_knowledge.py` に対して以下の変更を行う。

-   使用するEmbeddingモデルをOpenAI EmbeddingsからGoogle Gemini Embeddings (`models/embedding-001`) に変更する。
-   上記変更に伴い、必要なAPIキーの取得方法や環境変数名を調整する (OpenAI APIキーからGemini APIキーへ)。
-   ChromaDBコレクション作成時に、新しいEmbeddingモデルの情報 (次元数など) が適切に反映されるようにする。
-   既存の機能（ドキュメント読み込み、分割、ChromaDBへの接続、replace/appendモード）は維持する。

以下の項目はスコープ外とする。

-   `main.py` の変更 (既にGemini Embeddingを使用しているため)。
-   ChromaDB自体のマイグレーションやデータ変換 (本改修では、'replace'モードでのコレクション再作成を基本とする)。
-   知識ベースのドキュメント内容の変更。

## 3. 機能要件

### FR-CHROMA-GEMINI-001: Embeddingモデルの変更

-   **要件ID**: FR-CHROMA-GEMINI-001
-   **説明**: ローダースクリプト内で使用するEmbeddingモデルを、OpenAI Embeddingsから `langchain_google_genai.GoogleGenerativeAIEmbeddings` に変更する。
-   **詳細**:
    -   `OpenAIEmbeddings` の初期化コードを `GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)` に置き換える。
    -   必要なインポート文 (`from langchain_google_genai import GoogleGenerativeAIEmbeddings`) を追加する。

### FR-CHROMA-GEMINI-002: APIキーおよび環境変数の変更

-   **要件ID**: FR-CHROMA-GEMINI-002
-   **説明**: OpenAI APIキーの代わりにGoogle Gemini APIキーを使用するように、関連する関数、環境変数名、および `secrets.yaml` の読み込み処理を修正する。
-   **詳細**:
    -   `get_openai_api_key()` 関数を `get_gemini_api_key()` 関数に変更するか、汎用的なキー取得関数内でキー名を変更する。
    -   環境変数 `OPENAI_API_KEY` の参照を `GEMINI_API_KEY` に変更する。
    -   `load_secrets_from_yaml()` 関数内で、`secrets.yaml` から `openai.api_key` を読み込む部分を、`google.gemini_api_key` (または類似のキー名) を読み込むように修正する。`secrets.yaml` の想定される構造も合わせてドキュメント化またはコメントで明記する。
    -   スクリプト実行時のAPIキー存在チェックや表示メッセージもGemini APIキーを反映するように修正する。

### FR-CHROMA-GEMINI-003: コレクションメタデータの更新 (replaceモード時)

-   **要件ID**: FR-CHROMA-GEMINI-003
-   **説明**: `replace` モードで空のコレクションを新規作成する際に、メタデータとしてGemini Embeddingモデルの情報 (例: モデル名 `models/embedding-001` や次元数768など) を記録することを検討する（ChromaDBの仕様とLangchainの挙動に依存）。
-   **詳細**:
    -   `client.get_or_create_collection` の `metadata` 引数で、`{"embedding_function_name": "GoogleGenerativeAIEmbeddings:models/embedding-001"}` や `{"embedding_dimension": 768}` といった情報を付与することを検討する。
    -   ただし、Langchainの `Chroma` クラスが `add_documents` や初期化時に `embedding_function` を渡すことで、コレクションのメタデータ（特に次元数）は暗黙的に正しく設定されることが期待されるため、過度な明示的指定は不要な場合もある。Langchainの挙動を優先し、エラーが発生しない範囲で情報を付加する。

## 4. 非機能要件

### NFR-CHROMA-GEMINI-001: 動作確認

-   **要件ID**: NFR-CHROMA-GEMINI-001
-   **説明**: 修正後のローダースクリプトを実行し、エラーなくChromaDBにデータが投入（replaceおよびappendモード）され、メインAPI (`main.py`) がそのデータを正常に読み込み、RAG機能が次元不一致エラーなしに動作することを確認する。
-   **詳細**:
    -   ローダースクリプトを `replace` モードで実行し、コレクションが768次元で作成されることを確認（例えば、`main.py` の `/chroma-status` がエラーを吐かなくなる、ChromaDBを直接確認するなど）。
    -   （可能であれば）ローダースクリプトを `append` モードで実行し、既存の768次元コレクションに問題なくデータが追加されることを確認。
    -   メインAPIを起動し、`/analyze` エンドポイント経由でRAG処理を行い、期待通りのレスポンスが得られることを確認。

### NFR-CHROMA-GEMINI-002: 設定ファイルの整合性

-   **要件ID**: NFR-CHROMA-GEMINI-002
-   **説明**: `secrets.yaml` のサンプルや推奨フォーマットを更新し、Gemini APIキーの指定方法を明確にする。
-   **詳細**:
    -   ローダースクリプト内のコメントや、プロジェクトのREADME等で `secrets.yaml` の新しい形式（Gemini APIキーを含む）を案内する。

## 5. 修正作業の優先度

-   高 (メインAPIのRAG機能が正常に動作するために必須の修正であるため)

## 6. 備考

-   ChromaDBの既存コレクション (`bouldering_advice`) は、本改修を適用したローダースクリプトの `replace` モード実行時に一度削除され、新しいEmbeddingモデル (768次元) に基づいて再作成されることを前提とする。これにより、既存の1536次元データは失われるため、必要に応じてバックアップまたは移行戦略を別途検討すること。
-   `load_secrets_from_yaml` 関数や `get_env_or_secret` 関数は、Streamlitの `st.secrets` へのフォールバックロジックを含んでいるが、ローダースクリプトは通常コマンドラインから実行されるため、Streamlitコンテキスト外での `st.secrets` アクセスは期待通りに動作しない可能性が高い。環境変数または `secrets.yaml` による設定を優先する現在の実装方針は適切である。 