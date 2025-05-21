# 言語設定要件定義

## 機能概要
- ボルダリング動画分析サービスの回答結果において、ユーザーの指定した言語（日本語/英語）で結果を生成する機能

## 詳細要件

### FR-001: 言語指定機能
- ユーザーがAPIリクエスト時に言語を指定できること
- 指定方法: HTTPヘッダー `X-Language` を使用
- 対応言語:
  - 日本語: `ja` または `ja-*` で始まるヘッダー値（例: `ja-JP`）
  - 英語: `en` または `en-*` で始まるヘッダー値（例: `en-US`）
  - デフォルト: 指定がない場合は英語

### FR-002: 言語設定の内部処理
- `/analyze` エンドポイントで `X-Language` ヘッダーを受け取り、適切な言語設定値（"日本語"または"英語"）に変換
- 変換した言語設定値を `analyze_and_generate_advice` 関数に引数として渡す
- `analyze_and_generate_advice` 関数内では、受け取った言語設定値をそのままプロンプト内で使用

### FR-003: Geminiプロンプト内の言語指定
- プロンプト内で以下のように言語指定を行うこと:
```
### 回答生成時ルール (output rules) ###
- 回答は{output_language}で生成してください。
```

### FR-004: 整合性の維持
- 言語設定値の命名を `output_language` で統一し、`/analyze` エンドポイントと `analyze_and_generate_advice` 関数間の引数受け渡しを明確にする
- 関数内で引数の値を上書きせず、受け取った言語設定値をそのまま使用する

## テクニカル要件

### TR-001: エラーハンドリング
- `X-Language` ヘッダーの値が無効な場合や、予期しない値の場合はデフォルト（英語）を使用

### TR-002: パフォーマンスへの影響
- 言語設定処理による顕著なパフォーマンスへの影響がないこと

### TR-003: メンテナンス性
- 将来的に対応言語を追加する場合に容易に拡張できる設計とする

## 実装参考（過去バージョン）
過去バージョンでは以下のように実装していました：
```python
# Determine the language for the response based on x_language header
output_language = "英語" # Default to English
if x_language and x_language.lower().startswith("ja"):
    output_language = "日本語"
elif x_language and x_language.lower().startswith("en"):
    output_language = "英語"

prompt_template = f"""
あなたは経験豊富なプロのボルダリングコーチです。以下の情報をすべて考慮して、クライマーへの次のトライで試せるような具体的で実践的な改善アドバイスを生成してください。

### 回答生成時ルール (output rules) ###
- 回答は{output_language}で生成してください。
```

## 検証基準
- 日本語指定時に日本語の回答が返ってくること
- 英語指定時に英語の回答が返ってくること
- 指定なし/不正な指定の場合にデフォルト言語（英語）の回答が返ってくること 