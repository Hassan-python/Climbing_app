import chromadb
import sys

# --- 定数 (app.py/load_knowledge.py と合わせる) ---
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "bouldering_advice"

def check_chromadb_contents():
    print(f"--- ChromaDB ('{CHROMA_DB_PATH}') の内容確認開始 ---")
    try:
        # 永続化クライアントを作成
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print(f"クライアント接続成功。")

        # コレクションを取得
        print(f"コレクション '{CHROMA_COLLECTION_NAME}' を取得中...")
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"コレクション取得成功。")

        # 格納されているアイテム数を表示
        count = collection.count()
        print(f"コレクション内のアイテム数: {count}")

        if count > 0:
            # 最初の数件のデータを表示 (ID, Embeddingの一部, メタデータ, ドキュメント)
            print("\n最初の最大5件のデータ:")
            # peek は最新バージョンでは非推奨の可能性 -> get を使う
            # peek_result = collection.peek(limit=5)
            get_result = collection.get(limit=5, include=['embeddings', 'metadatas', 'documents'])

            ids = get_result.get('ids', [])
            embeddings = get_result.get('embeddings', [])
            metadatas = get_result.get('metadatas', [])
            documents = get_result.get('documents', [])

            for i in range(len(ids)):
                print(f"\n--- Item {i+1} ---")
                print(f"  ID: {ids[i]}")
                # Embeddingは長いので一部のみ表示
                emb_preview = embeddings[i][:5] if embeddings and i < len(embeddings) else "N/A"
                print(f"  Embedding (最初の5要素): {emb_preview} ...")
                print(f"  Metadata: {metadatas[i] if metadatas and i < len(metadatas) else 'N/A'}")
                print(f"  Document: {documents[i] if documents and i < len(documents) else 'N/A'}")
        else:
            print("コレクションは空です。")

    except Exception as e:
        print(f"\nChromaDBの確認中にエラーが発生しました: {e}", file=sys.stderr)
        print("考えられる原因:")
        print("- 指定されたパスにDBが存在しない (load_knowledge.py が未実行または失敗)")
        print("- 指定されたコレクション名が存在しない")
        return False
    return True

if __name__ == "__main__":
    success = check_chromadb_contents()
    if success:
        print("\n--- 確認スクリプト終了 ---")
    else:
        print("\n--- 確認スクリプト終了 (エラーあり) ---")
        sys.exit(1) 