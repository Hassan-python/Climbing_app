    # Pythonの公式イメージをベースイメージとして使用
    FROM python:3.10-slim

    # 環境変数を設定
    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1
    ENV PORT 8000 # Cloud Runがリッスンするポート

    # 作業ディレクトリを設定
    WORKDIR /app

    # 依存関係ファイルとパッケージリストをコピー
    # これらはビルドコンテキストのルートにあると想定
    COPY requirements_gcp.txt .
    COPY packages.txt .

    # packages.txt に記載されたシステムライブラリをインストール
    RUN apt-get update && \
        apt-get install -y --no-install-recommends $(cat packages.txt) && \
        rm -rf /var/lib/apt/lists/*

    # Pythonの依存関係をインストール
    RUN pip install --no-cache-dir -r requirements_gcp.txt

    # ★★★ デバッグ用: ビルドコンテキストのルートとWORKDIRの内容を表示 ★★★
    RUN echo "--- Content of /workspace (build context root) ---" && ls -la /workspace
    RUN echo "--- Content of /app (WORKDIR) ---" && ls -la .

    # アプリケーションコードをコピー
    # これらもビルドコンテキストのルートにあると想定
    COPY main.py .
    COPY load_knowledge.py . # もしこれもCloud Runで利用する場合
    # 他に必要な .py ファイルがあれば同様にコピー

    # アプリケーションを実行
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]