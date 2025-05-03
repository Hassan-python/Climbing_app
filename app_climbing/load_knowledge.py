import os
import shutil # ディレクトリ削除のためにインポート
import argparse # コマンドライン引数処理のため追加
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st # Secretsを使うためにインポート (ローカル実行用に調整も可)
import sys

# --- 定数 (app.pyと合わせる) ---
KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "bouldering_advice"

# --- OpenAI APIキー取得 (app.pyから流用) ---
def get_openai_api_key():
    """Streamlit SecretsからOpenAI APIキーを取得"""
    # このスクリプトをStreamlit外で実行する場合、環境変数などから読み込む方法も考慮
    # 例: return os.environ.get("OPENAI_API_KEY")
    # ここではStreamlit Secretsを使う前提で実装
    try:
        if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
            print("エラー: OpenAI APIキーがsecrets.tomlに設定されていません。", file=sys.stderr)
            return None
        return st.secrets["openai"]["api_key"]
    except Exception as e:
        # Streamlitサーバー外で実行された場合などのためのフォールバック
        print(f"警告: Streamlit Secretsの読み込みに失敗しました ({e})。環境変数 OPENAI_API_KEY を試します。", file=sys.stderr)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("エラー: 環境変数 OPENAI_API_KEY も設定されていません。", file=sys.stderr)
            return None
        return api_key

def load_documents():
    """knowledge_baseフォルダからドキュメントを読み込む"""
    print(f"'{KNOWLEDGE_BASE_DIR}' からドキュメントを読み込み中...")
    try:
        loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documents = loader.load()
        if not documents:
            print(f"警告: '{KNOWLEDGE_BASE_DIR}' にドキュメントが見つかりませんでした。", file=sys.stderr) # エラーから警告へ変更
            return [] # 空リストを返す
        print(f"{len(documents)} 個のドキュメントを読み込みました。")
        return documents
    except Exception as e:
        print(f"ドキュメント読み込み中にエラーが発生しました: {e}", file=sys.stderr)
        return None # エラー時はNoneを返す

def split_documents(documents):
    """ドキュメントをチャンクに分割する"""
    if not documents:
        return []
    print("ドキュメントをチャンクに分割中...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} 個のチャンクに分割しました。")
    return texts

def load_and_store_knowledge(mode='replace'): # mode引数を追加 (デフォルトはreplace)
    """knowledge_baseフォルダからドキュメントを読み込みChromaDBに格納する"""
    openai_api_key = get_openai_api_key()
    if not openai_api_key:
        return False

    # --- 1. ドキュメントの読み込みと分割 ---
    documents = load_documents()
    if documents is None: # 読み込みエラー
        return False
    if not documents and mode == 'replace':
         print("エラー: 'replace'モードで、knowledge_baseにドキュメントがありません。データベースは空になります。", file=sys.stderr)
         # 既存DB削除だけ行う
         if os.path.exists(CHROMA_DB_PATH):
             print(f"既存のデータベースディレクトリ '{CHROMA_DB_PATH}' を削除しています...")
             try:
                 shutil.rmtree(CHROMA_DB_PATH)
                 print("削除しました。")
             except Exception as e:
                 print(f"エラー: 既存データベースの削除に失敗しました: {e}", file=sys.stderr)
                 return False
         return True # ドキュメントがないが、処理としては成功（空DB作成）
    elif not documents and mode == 'append':
         print("警告: 'append'モードで、knowledge_baseに新しいドキュメントがありません。データベースは変更されません。")
         return True # 何もせず正常終了


    texts = split_documents(documents)
    if not texts and documents: # 分割に失敗した場合など (通常は起こりにくい)
        print("エラー: ドキュメントの分割に失敗しました。", file=sys.stderr)
        return False

    # --- 2. Embeddingモデルの初期化 ---
    print("Embeddingモデルを初期化中...")
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    except Exception as e:
        print(f"Embeddingモデルの初期化中にエラーが発生しました: {e}", file=sys.stderr)
        return False

    # --- 3. モードに応じたDB操作 ---
    if mode == 'replace':
        print(f"\n--- モード: replace ---")
        # --- 既存のChromaDBディレクトリを削除 ---
        if os.path.exists(CHROMA_DB_PATH):
            print(f"既存のデータベースディレクトリ '{CHROMA_DB_PATH}' を削除しています...")
            try:
                shutil.rmtree(CHROMA_DB_PATH)
                print("削除しました。")
            except Exception as e:
                print(f"エラー: 既存データベースの削除に失敗しました: {e}", file=sys.stderr)
                # ここで return False する前に、ロックの問題がないか再度確認を促すメッセージを出すことも検討
                print("他のプロセスがデータベースファイルを使用している可能性があります。Streamlitアプリなどが実行中でないか確認してください。")
                return False
        # --- 新規DB作成 ---
        if not texts: # ドキュメントがない場合は空のDBを作る (上記で処理済みだが念のため)
             print("knowledge_base にドキュメントがないため、空のデータベースを作成します。")
             # 空のDBを作成する場合、Chromaの初期化だけ行う
             try:
                 _ = Chroma(
                     persist_directory=CHROMA_DB_PATH,
                     embedding_function=embeddings, # embedding関数は指定が必要
                     collection_name=CHROMA_COLLECTION_NAME
                 )
                 print("空のデータベースを作成しました。")
                 return True
             except Exception as e:
                  print(f"空のデータベース作成中にエラーが発生しました: {e}", file=sys.stderr)
                  return False
        else:
            print(f"ChromaDB ('{CHROMA_DB_PATH}') にデータを格納中 (新規作成)...")
            try:
                vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_PATH,
                    collection_name=CHROMA_COLLECTION_NAME
                )
                vectorstore.persist()
                print("データの格納が完了しました。")
                return True
            except Exception as e:
                print(f"ChromaDBへのデータ格納中にエラーが発生しました: {e}", file=sys.stderr)
                return False

    elif mode == 'append':
        print(f"\n--- モード: append ---")
        if not texts: # 追加するドキュメントがない場合
            print("追加する新しいドキュメントがないため、処理をスキップします。")
            return True # 何もせず正常終了

        # --- 既存DBへの接続と追加 ---
        if not os.path.exists(CHROMA_DB_PATH):
             print(f"エラー: 'append'モードですが、データベースディレクトリ '{CHROMA_DB_PATH}' が存在しません。", file=sys.stderr)
             print("先に 'replace' モードでデータベースを作成してください。")
             return False

        print(f"既存のChromaDB ('{CHROMA_DB_PATH}') に接続し、データを追加中...")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME
            )
            # ドキュメントIDの重複を避けるための考慮 (Chromaのデフォルト挙動に依存)
            # 必要であれば、追加前に既存IDと重複しないかチェックするロジックを追加
            vectorstore.add_documents(texts)
            vectorstore.persist()
            print("データの追加が完了しました。")
            return True
        except Exception as e:
            print(f"ChromaDBへのデータ追加中にエラーが発生しました: {e}", file=sys.stderr)
            return False
    else:
        print(f"エラー: 無効なモード '{mode}' が指定されました。'replace' または 'append' を使用してください。", file=sys.stderr)
        return False

if __name__ == "__main__":
    # --- コマンドライン引数の設定 ---
    parser = argparse.ArgumentParser(description="知識ベースのドキュメントを読み込み、ChromaDBに格納または追加します。")
    parser.add_argument(
        "-m", "--mode",
        choices=['replace', 'append'],
        default='replace', # デフォルトは 'replace'
        help="データベースの更新モードを選択します ('replace': 全置き換え, 'append': 既存に追加)"
    )
    args = parser.parse_args()

    print(f"--- 知識ベース読み込み・格納スクリプト開始 (モード: {args.mode}) ---")

    # OpenAI APIキーが設定されているか環境変数で確認 (Streamlit外実行のため)
    if "OPENAI_API_KEY" not in os.environ:
        print("警告: 環境変数 'OPENAI_API_KEY' が設定されていません。Streamlit Secretsからの取得を試みますが、コマンドライン実行では失敗する可能性があります。")
        # ここで処理を中断するか、get_openai_api_key内のエラーに任せるか選択
        # ここでは処理を続行し、get_openai_api_keyのエラーハンドリングに任せる

    success = load_and_store_knowledge(mode=args.mode)

    if success:
        print(f"--- 処理完了 (モード: {args.mode}) --- ")
    else:
        print(f"--- エラーが発生したため終了しました (モード: {args.mode}) --- ")
        sys.exit(1) 