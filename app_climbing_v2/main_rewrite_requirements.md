# GCP Cloud Runデプロイエラー (ImportError) 修正のための要件定義書 (`main_rewrite_requirements.md`)

## 1. 背景・目的

既存のボルダリング動画分析API (`app_backup_v1.py`) は、動画フレームの分析処理にGoogle Gemini APIを、関連知識の検索と最終的なアドバイス生成にOpenAI APIおよびChromaDBを利用している。
この構成は複数のLLM API呼び出しを伴い、処理フローが分断されている。

本バージョンアップ (`main.py`) の主な目的は以下の通りである。

GCP Cloud Runへのデプロイ時に、`main.py` 内の以下のインポート文で `ImportError` が発生している。

```python
from langchain_google_genai import GoogleGenerativeAiEmbeddings
```

エラーメッセージ: `ImportError: cannot import name 'GoogleGenerativeAiEmbeddings' from 'langchain_google_genai'`

このエラーは、Cloud Runの実行環境において `langchain-google-genai` パッケージが期待通りに機能していないことを示唆している。
本改修の目的は、この `ImportError` を解消し、アプリケーションがGCP Cloud Run上で正常に起動・動作するようにすることである。

## 2. 現状分析

エラーログから以下の点が推測される。

-   Pythonのバージョンは `3.10` が使用されている。
-   `langchain_google_genai` パッケージ自体は `/usr/local/lib/python3.10/site-packages/langchain_google_genai/__init__.py` に存在しているように見える。
-   しかし、そのパッケージ内から `GoogleGenerativeAiEmbeddings` を名前で直接インポートできていない。

考えられる主な原因：

1.  **依存関係の不足・不整合**: `requirements.txt` (または他の依存関係管理ファイル) に `langchain-google-genai` が記載されていない、またはバージョンが古い・不適切である。
2.  **パッケージ構造の変更**: `langchain-google-genai` パッケージの最近のバージョンで、`GoogleGenerativeAiEmbeddings` のインポートパスやクラス名が変更された可能性がある。
3.  **インストール不備**: Cloud Runのビルドプロセス中に、何らかの理由で `langchain-google-genai` が正しくインストールされなかったか、一部ファイルが欠損した。
4.  **Python環境の差異**: ローカル開発環境とCloud Run環境でのPythonやpipの挙動に微妙な差異があり、ローカルでは問題なくてもCloud Runでは問題が発生している。

## 3. 機能要件 (修正方針)

### FR-IMPERR-001: 依存関係の確認と修正

-   **要件ID**: FR-IMPERR-001
-   **説明**: アプリケーションの依存関係定義ファイル (`requirements.txt` など) を確認し、`langchain-google-genai` パッケージが適切なバージョンで含まれていることを保証する。
-   **詳細**:
    -   `requirements.txt` に `langchain-google-genai` が記載されているか確認する。
    -   記載がない場合、適切なバージョン (例: 最新安定版、またはローカルで動作確認が取れているバージョン) を追加する。
    -   記載がある場合、バージョン指定が古くないか、あるいは逆に新しすぎて互換性の問題がないか確認する。ローカル開発環境で動作しているバージョンと一致させることを推奨。
    -   `langchain-google-genai` が依存する他のパッケージ (例: `google-generativeai`, `langchain-core` など) も `requirements.txt` に適切に記載されているか、または `langchain-google-genai` のインストール時に自動的に解決されるか確認する。
    -   `Dockerfile` (またはCloud Runのビルドに使用する設定ファイル) 内で `pip install -r requirements.txt` が正しく実行され、エラーなく完了していることを確認する。

### FR-IMPERR-002: インポートパスの検証と修正

-   **要件ID**: FR-IMPERR-002
-   **説明**: `langchain-google-genai` パッケージの公式ドキュメントやリリースノートを参照し、`GoogleGenerativeAiEmbeddings` クラスの正しいインポート方法を確認する。必要であれば `main.py` のインポート文を修正する。
-   **詳細**:
    -   `langchain-google-genai` の最新バージョン、または `requirements.txt` で指定しているバージョンにおける `GoogleGenerativeAiEmbeddings` のインポートパスを特定する。
        -   例: `from langchain_google_genai import GoogleGenerativeAIEmbeddings` (クラス名の大文字・小文字が正確か確認)
        -   例: `from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings` (サブモジュールからのインポートが必要か確認)
    -   現在の `main.py` のインポート文 (`from langchain_google_genai import GoogleGenerativeAiEmbeddings`) が、検証した正しいインポートパスと一致しているか確認する。
    -   一致していない場合、`main.py` のインポート文を正しい形式に修正する。

## 4. 非機能要件

### NFR-IMPERR-001: 動作確認

-   **要件ID**: NFR-IMPERR-001
-   **説明**: 修正後、アプリケーションがGCP Cloud Run上でエラーなく起動し、`/analyze` エンドポイントおよび `/chroma-status` エンドポイントが期待通りに動作することを確認する。
-   **詳細**:
    -   Cloud Runへの再デプロイ。
    -   起動ログで `ImportError` が再発しないことを確認。
    -   `/chroma-status` エンドポイントを呼び出し、ChromaDBへの接続が成功することを確認。
    -   （可能であれば）`/upload` および `/analyze` エンドポイントを実際に使用し、RAG処理を含む一連の動作が正常に行われることを確認。

### NFR-IMPERR-002: ローカル環境との一貫性

-   **要件ID**: NFR-IMPERR-002
-   **説明**: 修正内容は、ローカル開発環境でも同様に動作することを確認し、環境間の差異を最小限に抑える。
-   **詳細**:
    -   `requirements.txt` の変更はローカル環境にも反映する。
    -   インポート文の修正はローカル環境でも適用する。
    -   ローカルで `uvicorn main:app --reload` などを実行し、アプリケーションが正常に起動し、主要機能が動作することを確認する。

## 5. 調査・検証ステップ

1.  **ローカル環境での `langchain-google-genai` のバージョンとインポート確認**:
    *   ローカルで `pip show langchain-google-genai` を実行し、インストールされているバージョンを確認する。
    *   ローカルのPythonインタプリタで `from langchain_google_genai import GoogleGenerativeAiEmbeddings` が成功するか試す。
2.  **`requirements.txt` の確認**:
    *   `langchain-google-genai` の記載とバージョンを確認する。
3.  **`langchain-google-genai` のドキュメント確認**:
    *   PyPIやLangchainの公式ドキュメントで、`GoogleGenerativeAiEmbeddings` の推奨されるインポート方法を調べる。特に最近のバージョン変更に注意する。
4.  **Dockerfile/ビルドプロセスの確認**:
    *   Cloud Runのデプロイに使用している `Dockerfile` やビルドスクリプトを確認し、依存関係のインストールが正しく行われているか、ビルドログにエラーがないか確認する。

## 6. 修正作業の優先度

-   高 (アプリケーションが起動しない致命的なエラーのため) 