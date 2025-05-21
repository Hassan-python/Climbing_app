import os
# import shutil # ディレクトリ削除のためにインポート ← 不要
import argparse
import yaml # YAML読み込みのために追加
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import chromadb.config
from urllib.parse import urlparse
import streamlit as st # Secrets読み込みのため (環境変数フォールバックあり)
import sys
import time

# --- 定数 (app.pyと合わせる) ---
# スクリプト自身のディレクトリを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(SCRIPT_DIR, "knowledge_base") # スクリプトからの相対パスに変更

# CHROMA_DB_PATH = "./chroma_db" # ローカルパスは使わない
CHROMA_COLLECTION_NAME = "bouldering_advice"
SECRETS_FILE_PATH = os.path.join(SCRIPT_DIR, "secrets.yaml") # こちらもスクリプトからの相対パスに

# --- YAMLファイルから設定を読み込み環境変数に設定する関数 ---
def load_secrets_from_yaml(file_path=SECRETS_FILE_PATH):
    """YAMLファイルから設定を読み込み、環境変数に設定する"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            secrets = yaml.safe_load(f)
            if not secrets:
                print(f"警告: {file_path} が空か、無効なYAML形式です。", file=sys.stderr)
                return

            # OpenAI APIキーを設定 (secrets['openai']['api_key'] が存在すれば)
            openai_key = secrets.get('openai', {}).get('api_key')
            if openai_key:
                if "OPENAI_API_KEY" not in os.environ:
                    os.environ["OPENAI_API_KEY"] = openai_key
                    print(f"{file_path} から OpenAI APIキーを環境変数に設定しました。")
                else:
                    print(f"環境変数 OPENAI_API_KEY は既に設定されています。{file_path} の値は使用しません。")
            else:
                print(f"警告: {file_path} に OpenAI API キー ('openai.api_key') が見つかりません。", file=sys.stderr)

            # ChromaDB URLを設定 (secrets['chromadb']['url'] が存在すれば)
            chromadb_url = secrets.get('chromadb', {}).get('url')
            if chromadb_url:
                if "CHROMA_DB_URL" not in os.environ:
                    os.environ["CHROMA_DB_URL"] = chromadb_url
                    print(f"{file_path} から ChromaDB URL を環境変数に設定しました。")
                else:
                    print(f"環境変数 CHROMA_DB_URL は既に設定されています。{file_path} の値は使用しません。")
            else:
                 print(f"警告: {file_path} に ChromaDB URL ('chromadb.url') が見つかりません。", file=sys.stderr)

    except FileNotFoundError:
        print(f"警告: {file_path} が見つかりません。環境変数またはStreamlit Secretsを使用します。", file=sys.stderr)
    except yaml.YAMLError as e:
        print(f"エラー: {file_path} の読み込み中にYAMLエラーが発生しました: {e}", file=sys.stderr)
    except Exception as e:
        print(f"エラー: {file_path} の読み込み中に予期せぬエラーが発生しました: {e}", file=sys.stderr)

# --- APIキー/URL取得関数 (環境変数優先、Streamlit Secretsフォールバック) ---
def get_env_or_secret(key_name, secret_section, secret_key):
    """環境変数またはStreamlit Secretsから値を取得"""
    value = os.environ.get(key_name)
    if value:
        # print(f"環境変数 {key_name} から値を取得しました。")
        return value
    else:
        print(f"環境変数 {key_name} が未設定です。Streamlit Secrets ({secret_section}.{secret_key}) を試します。", file=sys.stderr)
        try:
            # Streamlitサーバー外での st.secrets アクセスは通常失敗するため、環境変数を推奨
            if secret_section not in st.secrets or secret_key not in st.secrets[secret_section]:
                print(f"エラー: Streamlit Secrets に {secret_section}.{secret_key} が設定されていません。環境変数 {key_name} を確認してください。", file=sys.stderr)
                return None
            # print(f"Streamlit Secrets {secret_section}.{secret_key} から値を取得しました。")
            return st.secrets[secret_section][secret_key]
        except Exception as e:
            print(f"警告: Streamlit Secrets の読み込みに失敗しました ({e})。環境変数 {key_name} を設定してください。", file=sys.stderr)
            return None

def get_openai_api_key():
    return get_env_or_secret("OPENAI_API_KEY", "openai", "api_key")

def get_chromadb_url():
    return get_env_or_secret("CHROMA_DB_URL", "chromadb", "url")

# --- ドキュメント読み込み関数 (変更なし) ---
def load_documents():
    """knowledge_baseフォルダからドキュメントを読み込む"""
    print(f"'{KNOWLEDGE_BASE_DIR}' からドキュメントを読み込み中...")
    try:
        loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documents = loader.load()
        if not documents:
            print(f"警告: '{KNOWLEDGE_BASE_DIR}' にドキュメントが見つかりませんでした。", file=sys.stderr)
            return []
        print(f"{len(documents)} 個のドキュメントを読み込みました。")
        return documents
    except Exception as e:
        print(f"ドキュメント読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return None

# --- ドキュメント分割関数 (変更なし) ---
def split_documents(documents):
    """ドキュメントをチャンクに分割する"""
    if not documents:
        return []
    print("ドキュメントをチャンクに分割中...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} 個のチャンクに分割しました。")
    return texts

# --- データロード関数 (HttpClientを使うように変更) ---
def load_and_store_knowledge_http(mode='replace'):
    openai_api_key = get_openai_api_key()
    chromadb_url = get_chromadb_url()

    if not openai_api_key or not chromadb_url:
        print("エラー: OpenAI APIキー または ChromaDB URL が取得できませんでした。環境変数または secrets.toml を確認してください。", file=sys.stderr)
        return False

    # --- 1. ドキュメントの読み込みと分割 ---
    documents = load_documents()
    if documents is None: return False # エラー
    texts = split_documents(documents)
    # texts が空でも、replaceモードならコレクション削除の可能性があるので続行

    # --- 2. Embeddingモデルの初期化 ---
    print("Embeddingモデルを初期化中...")
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    except Exception as e:
        print(f"Embeddingモデルの初期化中にエラーが発生しました: {e}", file=sys.stderr)
        return False

    # --- 3. ChromaDB HttpClient 初期化 ---
    print(f"ChromaDBサーバー ({chromadb_url}) に接続中...")
    try:
        parsed_url = urlparse(chromadb_url)
        host = parsed_url.hostname
        port = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)
        ssl_enabled = parsed_url.scheme == 'https'
        settings = chromadb.config.Settings(
            chroma_api_impl="rest",
            # requests_timeout=120 # このバージョンの chromadb ではサポートされていないため削除
        )
        client = chromadb.HttpClient(host=host, port=port, ssl=ssl_enabled, settings=settings)
        # 接続確認
        client.heartbeat()
        print("ChromaDBサーバーへの接続成功。")
    except Exception as e:
        print(f"ChromaDBサーバーへの接続/初期化中にエラーが発生しました: {e}", file=sys.stderr)
        return False

    # --- 4. モードに応じたDB操作 ---
    if mode == 'replace':
        print(f"\n--- モード: replace ---")
        # --- 既存コレクションの削除 --- (エラーは無視する)
        try:
            print(f"既存のコレクション '{CHROMA_COLLECTION_NAME}' を削除しようとしています...")
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
            print(f"コレクション '{CHROMA_COLLECTION_NAME}' を削除しました。(存在しなかった可能性もあります)")
            time.sleep(2) # 削除が反映されるまで少し待つ
        except Exception as e:
            print(f"既存コレクションの削除中にエラーまたは警告: {e} (無視して続行します)")

        # --- 新規コレクション作成 & データ追加 ---
        if not texts:
             print("knowledge_base にドキュメントがないため、空のコレクションを作成します。")
             try:
                 # メタデータで embedding 関数名を指定する (OpenAIEmbeddingsのデフォルトモデル名)
                 collection = client.get_or_create_collection(
                     name=CHROMA_COLLECTION_NAME,
                     metadata={"embedding_function": "text-embedding-ada-002"} # OpenAIのデフォルトを指定
                 )
                 print(f"空のコレクション '{CHROMA_COLLECTION_NAME}' を作成しました。")
                 return True
             except Exception as e:
                 print(f"空のコレクション作成中にエラーが発生しました: {e}", file=sys.stderr)
                 return False
        else:
            print(f"コレクション '{CHROMA_COLLECTION_NAME}' を(再)作成し、データを格納中...")
            try:
                 # LangChainのChromaクラス経由でデータを追加するのが簡単
                 vectorstore = Chroma(
                     client=client,
                     collection_name=CHROMA_COLLECTION_NAME,
                     embedding_function=embeddings,
                     # create_collection_if_not_exists=True # 暗黙的にTrueのはず
                 )
                 vectorstore.add_documents(texts)
                 print(f"{len(texts)} 個のチャンクをコレクションに追加しました。")
                 return True
            except Exception as e:
                print(f"コレクションへのデータ格納中にエラーが発生しました: {e}", file=sys.stderr)
                return False

    elif mode == 'append':
        print(f"\n--- モード: append ---")
        if not texts:
            print("追加する新しいドキュメントがないため、処理をスキップします。")
            return True

        print(f"既存のコレクション '{CHROMA_COLLECTION_NAME}' に接続し、データを追加中...")
        try:
            # LangChainのChromaクラス経由で接続・追加
            vectorstore = Chroma(
                client=client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings
            )
            vectorstore.add_documents(texts)
            print(f"{len(texts)} 個のチャンクをコレクションに追加しました。")
            return True
        except Exception as e:
             print(f"コレクションへのデータ追加中にエラーが発生しました: {e}", file=sys.stderr)
             print(f"コレクション '{CHROMA_COLLECTION_NAME}' が存在しない可能性があります。先に 'replace' モードで作成してください。")
             return False
    else:
        print(f"エラー: 無効なモード '{mode}'", file=sys.stderr)
        return False

if __name__ == "__main__":
    # --- YAMLファイルから設定を読み込む --- (スクリプト実行時にのみ)
    load_secrets_from_yaml()

    parser = argparse.ArgumentParser(description="知識ベースを読み込み、リモートChromaDBに格納/追加します。")
    parser.add_argument(
        "-m", "--mode",
        choices=['replace', 'append'],
        default='replace',
        help="DB更新モード ('replace': 全置き換え, 'append': 既存に追加)"
    )
    args = parser.parse_args()

    print(f"--- 知識ベース読み込み・格納スクリプト開始 (モード: {args.mode}, 接続先: リモート ChromaDB) ---")

    # APIキー/URLの存在チェック
    openai_key = get_openai_api_key()
    chroma_url = get_chromadb_url()
    if not openai_key:
         print("終了: OpenAI APIキーが必要です。環境変数 OPENAI_API_KEY または secrets.toml を設定してください。")
         sys.exit(1)
    if not chroma_url:
         print("終了: ChromaDB URLが必要です。環境変数 CHROMA_DB_URL または secrets.toml を設定してください。")
         sys.exit(1)

    # Streamlit Secrets に依存しないように環境変数名を明記
    print("使用する設定:")
    print(f"  OpenAI API Key: {'設定済み' if openai_key else '未設定'}")
    print(f"  ChromaDB URL: {chroma_url if chroma_url else '未設定'}")
    print(f"  コレクション名: {CHROMA_COLLECTION_NAME}")
    print(f"  知識ベースディレクトリ: {KNOWLEDGE_BASE_DIR}")
    print("-" * 20)

    success = load_and_store_knowledge_http(mode=args.mode)

    if success:
        print(f"--- 処理完了 (モード: {args.mode}) --- ")
    else:
        print(f"--- エラーが発生したため終了しました (モード: {args.mode}) --- ")
        sys.exit(1) 