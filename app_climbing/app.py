import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    # Streamlitのログで確認できるように標準出力に追加
    print("Successfully swapped sqlite3 with pysqlite3")
except ImportError:
    # Streamlitのログで確認できるように標準出力に追加
    print("pysqlite3 not found, using system sqlite3.")
    # pysqlite3が見つからない場合のエラーハンドリングをここに追加することも検討
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


# --- 定数 --- (必要に応じて調整)
ANALYSIS_INTERVAL_SEC = 0.5 # フレーム抽出間隔
TEMP_VIDEO_DIR = "temp_videos" # 一時動画保存フォルダ
CHROMA_COLLECTION_NAME = "bouldering_advice" # Chromaコレクション名
MAX_FRAMES_FOR_GEMINI = 3 # Geminiに渡す最大フレーム数

# --- OpenAI APIキー (Secretsから読み込む想定) ---
def get_openai_api_key():
    """Streamlit SecretsからOpenAI APIキーを取得"""
    if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
        st.error("OpenAI APIキーがsecrets.tomlに設定されていません。")
        return None
    return st.secrets["openai"]["api_key"]

# --- Gemini APIキー (Secretsから読み込む想定) ---
def get_gemini_api_key():
    """Streamlit SecretsからGemini APIキーを取得"""
    if "gemini" not in st.secrets or "api_key" not in st.secrets["gemini"]:
        st.error("Gemini APIキーがsecrets.tomlに設定されていません。")
        return None
    return st.secrets["gemini"]["api_key"]

# --- ChromaDB URL (Secretsから読み込む想定) ---
def get_chromadb_url():
    """Streamlit SecretsからChromaDB Cloud Run URLを取得"""
    if "chromadb" not in st.secrets or "url" not in st.secrets["chromadb"]:
        st.error("ChromaDBのURLがsecrets.tomlに設定されていません。")
        return None
    return st.secrets["chromadb"]["url"]

# --- フレーム抽出関数 ---
def extract_frames(video_path, start_sec, end_sec, interval_sec=ANALYSIS_INTERVAL_SEC):
    """指定された動画の区間からフレームを抽出する"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) # フレームレートを取得
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    interval_frames = max(1, int(interval_sec * fps)) # 少なくとも1フレーム間隔

    frames = []
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

    cap.release()
    return frames

# --- RAG + Gemini Vision パイプライン関数 ---
def get_advice_from_frames(frames, openai_api_key, gemini_api_key, problem_type, crux):
    """抽出フレームをGeminiで分析し、その結果とChromaDB検索結果をGPT-4.1 Nanoに渡してアドバイス生成"""
    st.info("Geminiによる画像分析とGPTによるアドバイス生成を開始します...")

    # --- 1. Gemini Vision によるフレーム分析 ---
    gemini_analysis_text = "画像分析なし" # デフォルト値
    if not gemini_api_key:
        st.warning("Gemini APIキーが未設定のため、画像分析をスキップします。")
    elif not frames:
        st.warning("分析するフレームがありません。画像分析をスキップします。")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')

            # 分析するフレームを選択 (例: 最初、中間、最後の最大 MAX_FRAMES_FOR_GEMINI 枚)
            num_frames_to_select = min(len(frames), MAX_FRAMES_FOR_GEMINI)
            indices = np.linspace(0, len(frames) - 1, num_frames_to_select, dtype=int)
            selected_frames_cv = [frames[i] for i in indices]

            # OpenCVフレーム(BGR)をPIL Image(RGB)に変換
            selected_frames_pil = []
            for frame_cv in selected_frames_cv:
                 if frame_cv is not None: # 再度Noneチェック
                    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    selected_frames_pil.append(pil_image)

            if not selected_frames_pil:
                 st.warning("Geminiに渡せる有効なフレームがありませんでした。")
            else:
                # Geminiへのプロンプト作成 (テキスト指示 + 画像リスト)
                gemini_prompt_parts = [
                    "あなたはクライミングの動きを分析する専門家です。提供された一連の画像（ボルダリング中のフレーム）を見て、以下の点を**具体的かつ簡潔に**日本語で記述してください。\n"
                    "- クライマーの体勢やバランス\n"
                    "- 各フレームでの手足の位置と動き\n"
                    "- 見受けられる非効率な動きや、落下につながりそうな不安定な要素\n"
                    "アドバイスではなく、客観的な観察結果のみを記述してください。\n\n",
                ]
                gemini_prompt_parts.extend(selected_frames_pil) # 画像を追加

                with st.spinner(f"Geminiが {len(selected_frames_pil)} フレームを分析中..."):
                    # TODO: response = gemini_vision_model.generate_content(gemini_prompt_parts, request_options={"timeout": 120}) のようにタイムアウト設定を検討
                    response = gemini_vision_model.generate_content(gemini_prompt_parts)
                    # response.prompt_feedback をチェックしてブロックされたか確認するのも良い
                    if response.parts:
                        gemini_analysis_text = response.text
                        st.success("Geminiによるフレーム分析が完了しました。")
                    else:
                        st.warning("Geminiからの応答が空でした。コンテンツフィルターにブロックされた可能性があります。")
                        # prompt_feedback の内容をログやデバッグ表示する
                        st.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")


                    if st.session_state.debug_mode:
                        with st.expander("Gemini 分析結果 (デバッグ用)", expanded=False):
                            st.text(gemini_analysis_text)

        except Exception as e:
            st.error(f"Geminiでのフレーム分析中にエラーが発生しました: {e}")
            # エラーが発生しても処理を続行する

    # --- 2. ChromaDBからの知識検索 ---
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
                # st.info(f"Connecting to ChromaDB at {chromadb_url}") # 接続先URL確認用
                parsed_url = urlparse(chromadb_url)
                host = parsed_url.hostname
                port = parsed_url.port if parsed_url.port else (443 if parsed_url.scheme == 'https' else 80)
                ssl_enabled = parsed_url.scheme == 'https'
                settings = chromadb.config.Settings(chroma_api_impl="rest") # persist_directory=None を削除
                client = chromadb.HttpClient(host=host, port=port, ssl=ssl_enabled, settings=settings)

                # 接続確認 (オプションだがデバッグに役立つ)
                # try:
                #     client.heartbeat() # サーバーへの疎通確認
                #     st.info("ChromaDB server heartbeat successful.")
                # except Exception as hb_e:
                #     st.error(f"ChromaDB server heartbeat failed: {hb_e}")
                #     raise # 接続失敗時はここでエラーにする

                vectorstore = Chroma(
                    client=client,
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=embeddings
                )

                # 検索クエリを作成 (ユーザー入力 + Gemini分析結果の最初の部分)
                search_query = f"課題の種類: {problem_type if problem_type else '指定なし'}, 難しい点: {crux if crux else '指定なし'}"
                if gemini_analysis_text != "画像分析なし" and gemini_analysis_text:
                    search_query += f"\n画像分析結果の抜粋: {gemini_analysis_text[:300]}" # Gemini結果もクエリに含める (長すぎると検索精度が落ちる可能性)

                with st.spinner("関連知識を検索中..."):
                    source_docs = vectorstore.similarity_search(search_query, k=3) # k=3を指定
                    if source_docs:
                        retrieved_docs_content = "\n\n".join([doc.page_content for doc in source_docs]) # 正しい結合方法
                        st.success(f"{len(source_docs)} 件の関連知識を検索しました。")
                    else:
                         st.warning("関連する知識が見つかりませんでした。")

        except Exception as e:
            st.error(f"ChromaDBでの知識検索中にエラーが発生しました: {e}")


    # --- 3. GPT-4.1 Nano による最終アドバイス生成 ---
    final_advice = "アドバイスを生成できませんでした。"
    if not openai_api_key:
        st.error("OpenAI APIキーが未設定のため、最終アドバイスを生成できません。")
    else:
        try:
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4.1-nano-2025-04-14", temperature=0.7) # モデル名は仮

            # 最終プロンプトテンプレート
            final_prompt_template = """
            あなたは経験豊富なボルダリングコーチです。以下の情報を**すべて考慮**して、クライマーへの**具体的で実践的な改善アドバイス**を日本語で生成してください。
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

            # プロンプトに情報を埋め込む
            formatted_prompt = PROMPT.format(
                user_problem_type=problem_type if problem_type else "特に指定なし",
                user_crux=crux if crux else "特に指定なし",
                gemini_analysis=gemini_analysis_text, # デフォルト値 "画像分析なし" が入る場合もある
                retrieved_knowledge=retrieved_docs_content
            )

            with st.spinner("GPTが最終アドバイスを生成中..."):
                # TODO: timeout の設定を検討
                final_advice = llm.invoke(formatted_prompt).content

        except Exception as e:
            st.error(f"最終アドバイスの生成中にエラーが発生しました: {e}")

    # アドバイスとソースドキュメントを返す
    return final_advice, source_docs

# --- Streamlit アプリ本体 ---
st.set_page_config(page_title="🧗 ボルダリング動画分析＆アドバイス (Gemini Vision)", layout="wide") # タイトルとレイアウト設定
st.title("🧗 ボルダリング動画分析＆アドバイス (Gemini Vision)")

# --- セッション状態の初期化 --- (変更なし)
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
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
st.session_state.debug_mode = st.sidebar.checkbox("デバッグモード (参照ソース/Gemini結果表示)", value=st.session_state.debug_mode)

st.header("1. 動画をアップロード")
uploaded_file = st.file_uploader("分析したいボルダリング動画（MP4, MOVなど）を選択してください", type=['mp4', 'mov', 'avi'])


if uploaded_file is not None:
    # 一時フォルダを作成 (存在しない場合)
    if not os.path.exists(TEMP_VIDEO_DIR):
        try:
            os.makedirs(TEMP_VIDEO_DIR)
        except OSError as e:
            st.error(f"一時ディレクトリの作成に失敗しました: {e}")
            st.stop() # ディレクトリが作れない場合は続行不可

    # アップロードされたファイルを一時フォルダに保存
    temp_file_path = os.path.join(TEMP_VIDEO_DIR, uploaded_file.name)
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"一時ファイルへの書き込みに失敗しました: {e}")
        st.stop()


    # st.success(f"動画 '{uploaded_file.name}' をアップロードしました。") # ボタン押下後に移動

    # 動画の長さを取得
    try:
        # 存在確認を追加
        if not os.path.exists(temp_file_path):
            st.error("一時動画ファイルが見つかりません。")
            st.stop()

        with VideoFileClip(temp_file_path) as clip: # with を使って自動クローズ
            video_duration = clip.duration
        # st.info(f"動画の長さ: {video_duration:.2f} 秒") # ボタン押下後に移動
    except Exception as e:
        st.error(f"動画情報の取得に失敗しました: {e}")
        video_duration = None # エラー時はdurationをNoneに

    if video_duration is not None:
        col1, col2 = st.columns([2, 1]) # レイアウト調整

        with col1:
            st.subheader("動画プレビュー")
             # 動画を表示 (スライダーと連動)
            st.video(temp_file_path, start_time=int(st.session_state.start_time))

        with col2:
            st.subheader("2. 分析設定")
            st.success(f"動画 '{uploaded_file.name}' ({video_duration:.2f} 秒) をロードしました。")

            # スライダーの最大値は動画の最後とする
            max_slider_value = video_duration

            # スライダーの表示範囲が有効かチェック (video_durationが0に近い場合)
            if max_slider_value > 0.0:
                start_time = st.slider(
                    "分析を開始する秒数を選択してください (3秒間分析)",
                    min_value=0.0,
                    max_value=max_slider_value,
                    # valueがmax_valueを超えないように調整
                    value=min(st.session_state.start_time, max_slider_value),
                    step=0.1,
                    format="%.1f"
                )
                # スライダーの値をセッション状態に保存
                st.session_state.start_time = start_time

                # 分析終了時間を計算 (開始時間+3秒 or 動画の最後)
                end_time = min(start_time + 3.0, video_duration)

                st.info(f"分析範囲: **{start_time:.1f} 秒 〜 {end_time:.1f} 秒**")

                # --- ユーザー入力欄を追加 ---
                st.text_input(
                    "課題の種類 (例: スラブ、強傾斜)",
                    key="problem_type" # セッションステートに直接紐付け
                )
                st.text_area(
                    "難しいと感じるポイント (例: 〇〇へのデッド)",
                    key="crux", # セッションステートに直接紐付け
                    height=100
                )
                # --------------------------

                if st.button("分析を開始", type="primary", use_container_width=True):
                    st.session_state.analysis_result = None # 結果をリセット
                    st.session_state.analysis_sources = [] # ソースをリセット

                    # APIキーのチェック
                    openai_api_key = get_openai_api_key()
                    gemini_api_key = get_gemini_api_key()

                    if not openai_api_key or not gemini_api_key:
                        st.error("OpenAI または Gemini の API キーが設定されていません。Secrets を確認してください。")
                    else:
                        st.info(f"{start_time:.1f}秒から{end_time:.1f}秒までの分析を開始します...")
                        frames = []
                        with st.spinner('フレームを抽出中...'):
                            frames = extract_frames(temp_file_path, start_time, end_time)

                        if frames:
                            st.success(f"{len(frames)} フレームの抽出に成功しました。")
                            # フレーム表示はデバッグモード時のみ、または削除
                            # if st.session_state.debug_mode:
                            #     st.subheader("抽出されたフレーム (デバッグ用)")
                            #     # ... (フレーム表示ループ) ...

                            # --- 分析の実行 (Gemini + GPT) ---
                            advice, sources = get_advice_from_frames(
                                frames,
                                openai_api_key,
                                gemini_api_key,
                                st.session_state.problem_type, # セッションステートから取得
                                st.session_state.crux        # セッションステートから取得
                            )
                            st.session_state.analysis_result = advice
                            st.session_state.analysis_sources = sources
                        else:
                            st.error("フレームの抽出に失敗しました。")
            else:
                # 動画長がほぼ0の場合
                st.warning("動画が短すぎるため、分析範囲を選択できません。")
                # start_timeをリセットしておく
                st.session_state.start_time = 0.0

    # --- 分析結果の表示 ---
    if st.session_state.analysis_result:
        st.divider()
        st.subheader("💡 AIからのアドバイス")
        st.markdown(st.session_state.analysis_result) # markdownとして表示

        # --- 参照ソースの表示 (デバッグモード時のみ) ---
        if st.session_state.debug_mode and st.session_state.analysis_sources:
            st.subheader("📚 参照した知識ソース (デバッグ用)")
            # 各ソースドキュメントを展開表示
            for i, doc in enumerate(st.session_state.analysis_sources):
                source_name = os.path.basename(doc.metadata.get('source', '不明なソース')) if doc.metadata else '不明なソース'
                with st.expander(f"ソース {i+1}: `{source_name}`"):
                    st.text(doc.page_content)
                    # メタデータ全体も表示（デバッグ用）
                    # st.json(doc.metadata)

else:
    st.info("動画ファイルをアップロードしてください。")

# --- 一時ファイルのクリーンアップ (オプション) ---
# Streamlit Cloud では自動でクリーンアップされることが多いが、ローカル実行用に考慮
# アプリ終了時などに TEMP_VIDEO_DIR 内を削除する処理を追加することも可能 