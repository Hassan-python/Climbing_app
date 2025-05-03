import subprocess
import os
import sys

def start_chroma_server():
    # 保存ディレクトリの設定
    chroma_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_data")
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(chroma_data_dir):
        os.makedirs(chroma_data_dir)
        print(f"Created directory: {chroma_data_dir}")
    
    # ChromaDBサーバーの起動コマンド
    cmd = ["chroma", "run", "--path", chroma_data_dir, "--port", "8000"]
    
    print(f"Starting ChromaDB server with command: {' '.join(cmd)}")
    
    try:
        # サブプロセスとしてChromaDBサーバーを起動
        process = subprocess.Popen(cmd)
        print(f"ChromaDB server started with PID: {process.pid}")
        print("Server is running at http://localhost:8000")
        print("Press Ctrl+C to stop the server")
        
        # メインプロセスが終了するまで待機
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping ChromaDB server...")
        process.terminate()
        process.wait()
        print("ChromaDB server stopped")
    except Exception as e:
        print(f"Error starting ChromaDB server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_chroma_server() 