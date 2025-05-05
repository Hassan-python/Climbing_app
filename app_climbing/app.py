import sys
import importlib # importlib を使う

try:
    # pysqlite3をインポート
    pysqlite3_module = importlib.import_module("pysqlite3")
    # インポートしたモジュールを直接 'sqlite3' として登録
    sys.modules["sqlite3"] = pysqlite3_module
    # Streamlitのログで確認できるように標準出力に追加
    print("Successfully swapped sqlite3 with pysqlite3 using importlib")
except ImportError:
    # Streamlitのログで確認できるように標準出力に追加
    print("pysqlite3 not found, using system sqlite3.")
    # pysqlite3が見つからない場合のエラーハンドリングをここに追加することも検討
    pass
except Exception as e: # 念のため他のエラーもキャッチ
    print(f"Error during sqlite3 swap: {e}")
    pass

# 必要なライブラリのインポート (chromadbを含む)
import streamlit as st
import os
from urllib.parse import urlparse # これを追加
# moviepy をインポート (動画の長さを取得するため)
from moviepy.editor import VideoFileClip
import cv2 # OpenCVをインポート
import numpy as np # OpenCVでフレームを扱うのに必要

# LangChain関連のインポート
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma # SnowflakeVectorStore から変更
# from langchain.chains import RetrievalQA # RetrievalQAは使わない
from langchain.prompts import PromptTemplate
# ChromaDBの永続化のために追加
import chromadb
# from langchain_core.runnables import RunnablePassthrough # 使わない
# from langchain_core.output_parsers import StrOutputParser # 使わない
from langchain.schema import Document # Documentオブジェクト操作のため
import chromadb.config # Settingsをインポート
import google.generativeai as genai # Gemini API をインポート
from PIL import Image # PIL をインポート

# --- Streamlit ページ設定 (最初の Streamlit コマンドである必要あり) ---
st.set_page_config(page_title="🧗 ボルダリング動画分析＆アドバイス (Gemini Vision)", layout="wide")
# ---------------------------------------------------------------------

# --- デバッグ用 Secrets 表示 --- (set_page_config の後に移動)
st.sidebar.subheader("Secrets Keys (Debug)")
if hasattr(st.secrets, 'items'): # .items() が使えるか確認
    for section, keys in st.secrets.items():
        st.sidebar.write(f"Section: [{section}]")
        if isinstance(keys, dict): # セクションの値が辞書であることを確認
             st.sidebar.write(f"- Keys: {list(keys.keys())}")
        else:
             # セクション直下に値がある場合 (通常はないはずだが念のため)
             st.sidebar.write(f"- Value type: {type(keys)}")
else:
    st.sidebar.warning("st.secrets object does not have .items()")
st.sidebar.divider()
# ----------------------------

# --- 定数 --- (必要に応じて調整)
ANALYSIS_INTERVAL_SEC = 0.5 # フレーム抽出間隔
TEMP_VIDEO_DIR = "temp_videos" # 一時動画保存フォルダ
CHROMA_COLLECTION_NAME = "bouldering_advice" # Chromaコレクション名
MAX_FRAMES_FOR_GEMINI = 10 # Geminiに渡す最大フレーム数
DEFAULT_ANALYSIS_DURATION = 1.0 # デフォルトの分析時間（秒）

# --- OpenAI APIキー (Secretsから読み込む想定) ---
def get_openai_api_key():
    """Streamlit SecretsからOpenAI APIキーを取得"""
    api_key = st.secrets.get("openai", {}).get("api_key")
    if not api_key:
        st.error("OpenAI APIキーがsecrets.tomlに設定されていません。")
    return api_key

# --- Gemini APIキー (Secretsから読み込む想定) ---
def get_gemini_api_key():
    """Streamlit SecretsからGemini APIキーを取得"""
    # secrets.toml のキー名を "google_genai" に合わせる
    api_key = st.secrets.get("google_genai", {}).get("api_key")
    if not api_key:
        st.error("Gemini APIキー (google_genai.api_key) がsecrets.tomlに設定されていません。")
    return api_key

# --- ChromaDB URL (Secretsから読み込む想定) ---
def get_chromadb_url():
    """Streamlit SecretsからChromaDB Cloud Run URLを取得"""
    url = st.secrets.get("chromadb", {}).get("url")
    if not url:
        st.error("ChromaDBのURL (chromadb.url) がsecrets.tomlに設定されていません。")
    return url

# --- フレーム抽出関数 ---
def extract_frames(video_path, start_sec, end_sec, interval_sec=ANALYSIS_INTERVAL_SEC):
    """指定された動画の区間からフレームを抽出する"""
    frames = []
    cap = None # 初期化
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"動画ファイルを開けませんでした: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) # フレームレートを取得
        if fps is None or fps <= 0: # FPSが取得できない場合や0以下の場合
            st.error(f"動画のFPSを取得できませんでした (値: {fps}): {video_path}")
            return []

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        # end_frame が start_frame より小さくならないように保証
        end_frame = max(start_frame, end_frame)
        interval_frames = max(1, int(interval_sec * fps)) # 少なくとも1フレーム間隔

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # 開始フレームに移動

        current_frame_count = start_frame
        frame_read_count = 0
        while cap.isOpened() and current_frame_count <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # interval_frames ごとにフレームを追加
            if frame_read_count % interval_frames == 0:
                if frame is not None: # フレームが有効かチェック
                    frames.append(frame)
                else:
                    st.warning(f"フレーム {current_frame_count} の読み込み中に無効なデータが見つかりました。")

            current_frame_count += 1
            frame_read_count += 1

    except Exception as e:
        st.error(f"フレーム抽出中に予期せぬエラーが発生しました: {e}")
        # エラーが発生した場合も空のリストを返す
        frames = []
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
    return frames

# --- RAG + Gemini Vision パイプライン関数 ---
def get_advice_from_frames(frames, openai_api_key, gemini_api_key):
    """抽出フレームをGeminiで分析し、その結果とChromaDB検索結果をGPTに渡してアドバイス生成"""
    st.info("Geminiによる画像分析とGPTによるアドバイス生成を開始します...")

    # --- 1. Gemini Vision によるフレーム分析 --- (インデントとロジック修正)
    gemini_analysis_text = "画像分析なし" # デフォルト値
    if not gemini_api_key:
        st.warning("Gemini APIキーが未設定のため、画像分析をスキップします。")
    elif not frames:
        st.warning("分析するフレームがありません。画像分析をスキップします。")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')

            num_frames_to_select = min(len(frames), MAX_FRAMES_FOR_GEMINI)
            selected_frames_cv = []
            if num_frames_to_select > 0:
                indices = np.linspace(0, len(frames) - 1, num_frames_to_select, dtype=int)
                selected_frames_cv = [frames[i] for i in indices]

            selected_frames_pil = []
            for frame_cv in selected_frames_cv:
                if frame_cv is not None:
                    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    selected_frames_pil.append(pil_image)

            if not selected_frames_pil:
                st.warning("Geminiに渡せる有効なフレームがありませんでした。")
            else:
                gemini_prompt_parts = [
                    """あなたはクライミングの動きを分析する専門家です。提供された一連の画像（ボルダリング中のフレーム）を見て、以下の点を**具体的かつ簡潔に**日本語で記述してください。

                    - クライマーの体勢やバランス
                    - 各フレームでの手足の位置と動き
                    - 見受けられる非効率な動きや、落下につながりそうな不安定な要素

                    アドバイスではなく、客観的な観察結果のみを記述してください。
                    """,
                ]
                gemini_prompt_parts.extend(selected_frames_pil)

                with st.spinner(f"Geminiが {len(selected_frames_pil)} フレームを分析中..."):
                    try:
                        response = gemini_vision_model.generate_content(
                            gemini_prompt_parts,
                            request_options={"timeout": 180}
                        )
                        if response.prompt_feedback.block_reason != 0:
                            st.warning(f"Geminiへのリクエストがブロックされました: {response.prompt_feedback.block_reason}")
                            st.warning(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                        elif response.parts:
                            gemini_analysis_text = response.text
                            st.success("Geminiによるフレーム分析が完了しました。")
                        else:
                            st.warning("Geminiからの応答が空でした。")
                    except Exception as genai_e:
                        st.error(f"Gemini API 呼び出し中にエラーが発生しました: {genai_e}")

                if st.session_state.debug_mode and gemini_analysis_text != "画像分析なし":
                    with st.expander("Gemini 分析結果 (デバッグ用)", expanded=False):
                        st.text(gemini_analysis_text)

        except Exception as e:
            st.error(f"Geminiでのフレーム分析準備中にエラーが発生しました: {e}")

    # --- 2. ChromaDBからの知識検索 --- (インデントとロジック修正)
    retrieved_docs_content = "関連知識なし"
    source_docs = []
    if not openai_api_key:
        st.warning("OpenAI APIキーが未設定のため、知識ベースの検索をスキップします。")
    else:
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            chromadb_url = get_chromadb_url()
            if not chromadb_url:
                st.warning("ChromaDB URLが未設定のため、知識ベースの検索をスキップします。")
            else:
                parsed_url = urlparse(chromadb_url)
                host = parsed_url.hostname
                port = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)
                ssl_enabled = parsed_url.scheme == 'https'
                settings = chromadb.config.Settings(chroma_api_impl="rest")
                client = chromadb.HttpClient(host=host, port=port, ssl=ssl_enabled, settings=settings)

                vectorstore = Chroma(
                    client=client,
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=embeddings
                )

                search_query = f"課題の種類: {st.session_state.problem_type if st.session_state.problem_type else '指定なし'}, 難しい点: {st.session_state.crux if st.session_state.crux else '指定なし'}"
                if gemini_analysis_text != "画像分析なし" and gemini_analysis_text:
                    search_query += f"\n画像分析結果の抜粋: {gemini_analysis_text[:300]}"

                with st.spinner("関連知識を検索中..."):
                    source_docs = vectorstore.similarity_search(search_query, k=3)
                    if source_docs:
                        # 文字列結合の修正
                        retrieved_docs_content = "\n\n".join([doc.page_content for doc in source_docs])
                        st.success(f"{len(source_docs)} 件の関連知識を検索しました。")
                    else:
                        st.warning("関連する知識が見つかりませんでした。")

        except Exception as e:
            st.error(f"ChromaDBでの知識検索中にエラーが発生しました: {e}")

    # --- 3. GPT による最終アドバイス生成 --- (インデントとロジック修正)
    final_advice = "アドバイスを生成できませんでした。"
    if not openai_api_key:
        st.error("OpenAI APIキーが未設定のため、最終アドバイスを生成できません。")
    else:
        try:
            openai_model_name = st.secrets.get("openai", {}).get("model_name", "gpt-4o-mini")
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=openai_model_name, temperature=0.7)

            final_prompt_template = """
            あなたは経験豊富なボルダリングコーチです。以下の情報を**すべて考慮**して、クライマーへの**次のトライで試せるような具体的で実践的な改善アドバイス**を日本語で生成してください。
            **重要：絶対に「詳細は不明ですが」「提供された情報だけでは」「もし～なら」といった、推測や情報不足、自信のなさを示す言葉を使わないでください。** コーチとして断定的に、自信を持ってアドバイスしてください。

            ---
            ユーザーが報告した状況:
            - 課題の種類: {user_problem_type}
            - 難しいと感じるポイント: {user_crux}
            ---
            AIによる画像分析結果 (客観的な観察):
            {gemini_analysis}
            ---
            関連するボルダリング知識 (データベースより):
            {retrieved_knowledge}
            ---

            上記情報を踏まえた、コーチとしてのアドバイス (ステップバイステップ形式や箇条書きを推奨):
            """
            PROMPT = PromptTemplate(
                template=final_prompt_template,
                input_variables=["user_problem_type", "user_crux", "gemini_analysis", "retrieved_knowledge"]
            )

            formatted_prompt = PROMPT.format(
                user_problem_type=st.session_state.problem_type if st.session_state.problem_type else "特に指定なし",
                user_crux=st.session_state.crux if st.session_state.crux else "特に指定なし",
                gemini_analysis=gemini_analysis_text,
                retrieved_knowledge=retrieved_docs_content
            )

            with st.spinner(f"GPT ({openai_model_name}) が最終アドバイスを生成中..."):
                final_advice = llm.invoke(formatted_prompt, config={"max_retries": 1, "request_timeout": 120}).content

        except Exception as e:
            st.error(f"最終アドバイスの生成中にエラーが発生しました: {e}")

    return final_advice, source_docs # 関数から抜ける return 文を正しいインデントに戻す

# --- ChromaDB ステータス確認関数 (デバッグ用) --- (インデント修正)
def check_chromadb_status():
    """ChromaDBへの接続と基本的な動作を確認する (デバッグ用)"""
    chromadb_url = get_chromadb_url()
    openai_api_key = get_openai_api_key()

    if not chromadb_url or not openai_api_key:
        return "⚠️ ChromaDB URL または OpenAI API キーが未設定です。"

    try:
        parsed_url = urlparse(chromadb_url)
        host = parsed_url.hostname
        port = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)
        ssl_enabled = parsed_url.scheme == 'https'
        settings = chromadb.config.Settings(chroma_api_impl="rest")
        client = chromadb.HttpClient(host=host, port=port, ssl=ssl_enabled, settings=settings)

        try:
            client.heartbeat()
        except Exception as hb_e:
            return f"❌ ChromaDB サーバー接続失敗 (Heartbeat): {hb_e}"

        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = Chroma(
                client=client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings
            )
            count = vectorstore._collection.count()
            return f"✅ ChromaDB 接続成功 (`{CHROMA_COLLECTION_NAME}`: {count} アイテム)"
        except Exception as coll_e:
             return f"⚠️ ChromaDB コレクション接続/カウント失敗: {coll_e}"

    except Exception as e:
        return f"❌ ChromaDB クライアント初期化失敗: {e}"

# --- Streamlit アプリ本体 ---
st.title("🧗 ボルダリング動画分析＆アドバイス (Gemini Vision)")

# --- セッション状態の初期化 --- (動画データ保持用 state 追加)
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'video_duration' not in st.session_state:
    st.session_state.video_duration = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0.0
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = ""
if 'crux' not in st.session_state:
    st.session_state.crux = ""
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_sources' not in st.session_state:
    st.session_state.analysis_sources = []

# --- UI要素 ---
st.sidebar.header("設定")
st.session_state.debug_mode = st.sidebar.checkbox("デバッグモード (詳細情報表示)", value=st.session_state.debug_mode)

# --- デバッグモード時の ChromaDB ステータス表示 ---
if st.session_state.debug_mode:
    with st.sidebar:
        with st.spinner("ChromaDB ステータス確認中..."):
            chroma_status = check_chromadb_status()
            if "✅" in chroma_status:
                st.info(chroma_status)
            else:
                st.warning(chroma_status)
    st.sidebar.divider()

st.header("1. 動画をアップロード")
uploaded_file = st.file_uploader("分析したいボルダリング動画（MP4, MOVなど）を選択してください", type=['mp4', 'mov', 'avi'])

# --- ファイルアップロード後の処理 --- (session state を使うように変更)
if uploaded_file is not None:
    # 以前のファイルと違うファイルがアップロードされたら状態をリセット
    if uploaded_file.name != st.session_state.get('uploaded_file_name'): # getで安全にアクセス
        st.session_state.video_bytes = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.video_duration = None
        st.session_state.start_time = 0.0
        st.session_state.analysis_result = None
        st.session_state.analysis_sources = []
        st.session_state.problem_type = "" # リセット
        st.session_state.crux = "" # リセット
        # st.success(f"新しい動画 '{uploaded_file.name}' を認識しました。") # メッセージは任意

    # --- 一時ファイルの準備 (バイトデータが存在する場合) ---
    temp_file_path = None
    if st.session_state.video_bytes:
        if not os.path.exists(TEMP_VIDEO_DIR):
            try:
                os.makedirs(TEMP_VIDEO_DIR)
            except OSError as e:
                st.error(f"一時ディレクトリの作成に失敗しました: {e}")
                st.stop()

        temp_file_path = os.path.join(TEMP_VIDEO_DIR, st.session_state.uploaded_file_name)
        try:
            with open(temp_file_path, "wb") as f:
                f.write(st.session_state.video_bytes)
        except Exception as e:
            st.error(f"一時ファイルへの書き込みに失敗しました: {e}")
            st.stop()

        # --- 動画長の取得 (初回のみ) ---
        if st.session_state.video_duration is None:
            try:
                if os.path.exists(temp_file_path):
                    with VideoFileClip(temp_file_path) as clip:
                        st.session_state.video_duration = clip.duration
                else:
                     st.error("一時動画ファイルが見つかりません。(Duration取得時)")
                     st.session_state.video_duration = 0
            except Exception as e:
                st.error(f"動画情報の取得に失敗しました: {e}")
                st.session_state.video_duration = 0

        video_duration = st.session_state.video_duration

        # --- UI表示 (動画長が取得でき、一時ファイルパスが有効なら) ---
        if video_duration > 0 and temp_file_path and os.path.exists(temp_file_path):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("動画プレビュー")
                # st.video の start_time 引数は削除 (シークバー操作の邪魔になるため)
                st.video(temp_file_path)

            with col2:
                st.subheader("2. 分析設定")
                st.success(f"動画 '{st.session_state.uploaded_file_name}' ({video_duration:.2f} 秒) をロードしました。")

                # --- ユーザー入力欄 ---
                st.text_input(
                    "課題の種類 (例: スラブ、強傾斜)",
                    key="problem_type"
                )
                st.text_area(
                    "難しいと感じるポイント (例: 〇〇へのデッド)",
                    key="crux",
                    height=100
                )

                # --- 分析開始時間の設定 ---
                current_start_time = st.number_input(
                    "分析開始時間 (秒)",
                    min_value=0.0,
                    max_value=video_duration,
                    value=st.session_state.start_time, # 初期値はstateから
                    step=0.1,
                    format="%.1f",
                    help="動画のどの時点から分析を開始するかを指定します。",
                    key="start_time_widget"
                )
                # 値が変わったら state を更新
                if current_start_time != st.session_state.start_time:
                    st.session_state.start_time = current_start_time
                    # プレビューに即時反映させたい場合は rerun するが、一旦不要
                    # st.rerun()

                # 分析終了時間を計算 (1秒固定)
                end_time = min(st.session_state.start_time + DEFAULT_ANALYSIS_DURATION, video_duration)
                st.info(f"分析範囲: **{st.session_state.start_time:.1f} 秒 〜 {end_time:.1f} 秒**")

                # --- 分析実行ボタン ---
                if st.button("分析を開始", type="primary", use_container_width=True):
                    st.session_state.analysis_result = None
                    st.session_state.analysis_sources = []

                    openai_api_key = get_openai_api_key()
                    gemini_api_key = get_gemini_api_key()

                    if not openai_api_key or not gemini_api_key:
                        st.error("OpenAI または Gemini の API キーが設定されていません。Secrets を確認してください。")
                    else:
                        start_time_for_analysis = st.session_state.start_time
                        end_time_for_analysis = min(start_time_for_analysis + DEFAULT_ANALYSIS_DURATION, video_duration)

                        st.info(f"{start_time_for_analysis:.1f}秒から{end_time_for_analysis:.1f}秒までの{DEFAULT_ANALYSIS_DURATION}秒間分析を開始します...")
                        frames = []
                        with st.spinner('フレームを抽出中...'):
                            frames = extract_frames(temp_file_path, start_time_for_analysis, end_time_for_analysis)

                        if frames:
                            st.success(f"{len(frames)} フレームの抽出に成功しました。")
                            advice, sources = get_advice_from_frames(
                                frames,
                                openai_api_key,
                                gemini_api_key
                                # problem_type と crux は関数内で state から取得
                            )
                            st.session_state.analysis_result = advice
                            st.session_state.analysis_sources = sources
                        else:
                            st.error("フレームの抽出に失敗しました。")
        else:
            st.warning("動画情報が正しく読み込めていないため、分析設定を表示できません。")

# --- 分析結果の表示 ---
if st.session_state.analysis_result:
    st.divider()
    st.subheader("💡 AIからのアドバイス")
    st.markdown(st.session_state.analysis_result)

    if st.session_state.debug_mode and st.session_state.analysis_sources:
        st.subheader("📚 参照した知識ソース (デバッグ用)")
        for i, doc in enumerate(st.session_state.analysis_sources):
            source_name = "不明"
            if doc.metadata and 'source' in doc.metadata:
                try:
                    source_name = os.path.basename(doc.metadata.get('source', '不明'))
                except Exception:
                    source_name = str(doc.metadata.get('source', '不明'))
            with st.expander(f"ソース {i+1}: `{source_name}`"):
                st.text(doc.page_content)

else: # uploaded_file is None or video_bytes is None
    if st.session_state.get('uploaded_file_name') is not None:
        st.session_state.video_bytes = None
        st.session_state.uploaded_file_name = None
        st.session_state.video_duration = None
        st.session_state.start_time = 0.0
        st.session_state.analysis_result = None
        st.session_state.analysis_sources = []
        # st.rerun() # 状態クリア時にリランが必要なら

    st.info("動画ファイルをアップロードしてください。")

# --- 一時ファイルのクリーンアップ検討 ---
# Streamlit のセッション管理の仕組み上、明示的なクリーンアップは難しい場合がある
# 一時ファイルが残り続ける場合、定期的な手動削除やサーバー側での仕組みが必要になる可能性 