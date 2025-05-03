import chromadb
from chromadb.config import Settings

# HTTPクライアントの初期化（設定を単純化）
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings()  # API実装の指定を省略
)

# テスト: コレクションの作成とデータの追加
collection = client.get_or_create_collection("test_collection")
collection.add(documents=["Test document"], ids=["id1"])
results = collection.query(query_texts=["Test"], n_results=1)

# 結果を読みやすく表示
print("Query Results:")
print(f"Documents: {results['documents']}")
print(f"IDs: {results['ids']}")
print(f"Distances: {results['distances']}")

print("\n接続先: http://localhost:8000")
print("注意: このコードを実行する前に、ChromaDBサーバーが起動していることを確認してください。")
print("サーバー起動方法: chroma run --path C:\\Users\\Hassan\\Desktop\\cursor\\chroma_data --port 8000")