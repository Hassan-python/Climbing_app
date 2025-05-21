# Pythonの公式イメージをベースイメージとして使用
FROM python:3.10-slim

# 環境変数を設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8000 # Cloud Runがリッスンするポート

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルとパッケージリストをコピー
COPY requirements_gcp.txt .
COPY packages.txt .

# ★★★ デバッグ: requirements_gcp.txt の内容を確認 ★★★
RUN echo "--- Content of requirements_gcp.txt in Docker build ---" && cat requirements_gcp.txt

# packages.txt に記載されたシステムライブラリをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends $(cat packages.txt | grep -v '^#' | xargs) && \
    rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をインストール
# まず requirements_gcp.txt からインストールし、その後 python-multipart を明示的にインストール
RUN pip install --no-cache-dir -r requirements_gcp.txt && \
    pip install --no-cache-dir python-multipart

# アプリケーションコードをコピー
COPY main.py .
COPY load_knowledge.py .

# アプリケーションを実行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]