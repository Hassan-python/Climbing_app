import streamlit as st
import os
# moviepy をインポート (動画の長さを取得するため)
from moviepy.editor import VideoFileClip
import cv2 # OpenCVをインポート
import numpy as np # OpenCVでフレームを扱うのに必要

# LangChain関連のインポート
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma # SnowflakeVectorStore から変更
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# ChromaDBの永続化のために追加
import chromadb
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document # Documentオブジェクト操作のため

# --- 定数 --- (必要に応じて調整)
ANALYSIS_INTERVAL_SEC = 0.5 # フレーム抽出間隔
TEMP_VIDEO_DIR = "temp_videos" # 一時動画保存フォルダ
CHROMA_DB_PATH = "./chroma_db" # ChromaDBのデータ保存先
CHROMA_COLLECTION_NAME = "bouldering_advice" # Chromaコレクション名

# --- OpenAI APIキー (Secretsから読み込む想定) ---
def get_openai_api_key():
    """Streamlit SecretsからOpenAI APIキーを取得"""
    # st.secretsに以下のキーで設定されていると仮定:
    # [openai]
    # api_key = "sk-..."
    if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
        st.error("OpenAI APIキーがsecrets.tomlに設定されていません。")
        return None
    return st.secrets["openai"]["api_key"]

# フレーム抽出関数
def extract_frames(video_path, start_sec, end_sec, interval_sec=ANALYSIS_INTERVAL_SEC):
    """指定された動画の区間からフレームを抽出する"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"動画ファイルを開けませんでした: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) # フレームレートを取得
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    interval_frames = int(interval_sec * fps)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # 開始フレームに移動

    current_frame_count = start_frame
    while cap.isOpened() and current_frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # interval_frames ごとにフレームを追加
        if (current_frame_count - start_frame) % interval_frames == 0:
            # OpenCVはBGR形式で読み込むので、必要ならRGBに変換
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ここではBGRのままNumPy配列として返す
            frames.append(frame)

        current_frame_count += 1

    cap.release()
    return frames

# --- RAGパイプライン関数 (プロンプトと情報注入方法を修正) ---
def get_advice_from_frames(frames, openai_api_key, problem_type, crux):
    """抽出されたフレームからRAG (ChromaDB) でアドバイスを生成する"""
    # st.warning("RAGパイプラインは現在実装中です。") # 実装したのでコメントアウト

    # 1. フレーム情報の前処理 (必須だが、まだ)
    # TODO: フレーム情報を活用したクエリ生成

    # 2. Embeddingモデルの初期化
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    except Exception as e:
        st.error(f"Embeddingモデルの初期化に失敗しました: {e}")
        return None, []

    # 3. ChromaDB Vector Storeの初期化/接続
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        base_retriever = vectorstore.as_retriever()
        st.info("ChromaDBに接続しました。")
    except Exception as e:
        st.error(f"ChromaDBへの接続/初期化に失敗しました: {e}")
        st.error("知識ベースがまだ準備されていない可能性があります。")
        return None, [] # エラー時はNoneと空リストを返す

    # 5. LLM (ChatOpenAI) の初期化
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4.1-nano-2025-04-14")
    except Exception as e:
        st.error(f"LLMの初期化に失敗しました: {e}")
        return None, []

    # 6. プロンプトテンプレートの作成 (指示をさらに強化、変数はcontext, questionのみ)
    prompt_template_text = """
    あなたは経験豊富なボルダリングコーチです。壁から落下してしまった原因を分析し、アドバイスすることが得意です。
    提供されたコンテキスト情報（ボルダリングの一般的知識）と質問（ユーザーが説明する状況を含む）を**唯一の根拠**として、**可能な限り**具体的で実践的なアドバイスを日本語でしてください。
    **重要：絶対に「フレームの詳細は不明ですが」「提供された情報だけでは」「もし～なら」といった、推測や情報不足、自信のなさを示す言葉を使わないでください。** コーチとして断定的に、自信を持ってアドバイスしてください。

    コンテキスト:
    {context}

    質問:
    {question}

    コーチとしてのアドバイス:
    """
    PROMPT = PromptTemplate(
        template=prompt_template_text,
        input_variables=["context", "question"] # context と question のみ
    )

    # 7. Retriever と QAチェーンの準備
    try:
        # QA チェーン (stuffのデフォルトプロンプトを使う)
        # デフォルトのStuffプロンプトを変更するために combine_docs_chain を指定
        from langchain.chains.question_answering import load_qa_chain
        combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
        
        qa_chain = RetrievalQA(
            retriever=base_retriever,
            combine_documents_chain=combine_docs_chain, # カスタムプロンプトを持つチェーンを指定
            return_source_documents=True
        )

    except Exception as e:
        st.error(f"RAGチェーンの準備中にエラーが発生しました: {e}")
        return None, []

    # フレーム情報とユーザー入力を全て質問文に含める
    num_frames = len(frames)
    # 三重引用符を使って複数行f-stringを定義
    user_info_text = f"""状況:
    - 分析対象: 動画から抽出された {num_frames} フレームのシーケンス。
    - ユーザーが認識する課題の種類: {problem_type if problem_type else '特に指定なし'}
    - ユーザーが認識する難しいポイント: {crux if crux else '特に指定なし'}"""
    main_question = "この状況とコンテキスト情報を踏まえ、観察されるであろうクライマーの動きについて、改善点を指摘し、具体的な改善方法を提案してください。"
    question = f"{user_info_text}\n\n質問: {main_question}"

    advice = None
    source_docs = []
    try:
        with st.spinner("AIが考えています..."):
            # RetrievalQA の場合、query のみに情報を詰めて渡す
            result = qa_chain.invoke({"query": question})
            advice = result.get("result")
            source_docs = result.get("source_documents", [])
    except Exception as e:
        st.error(f"アドバイス生成中にエラーが発生しました: {e}")
        # エラーでもNoneと空リストを返す

    # ソースファイル名を抽出 (重複除去)
    # source_files = list(set([doc.metadata['source'] for doc in source_docs if 'source' in doc.metadata]))

    # アドバイスとソースドキュメントのリストを返すように変更
    return advice, source_docs

# --- Streamlit アプリ本体 ---
st.title("🧗 ボルダリング動画分析＆アドバイス")

# デバッグモードの状態管理とトグルスイッチ
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
st.session_state.debug_mode = st.checkbox("デバッグモード (参照ソースを表示)", value=st.session_state.debug_mode)

st.header("1. 動画をアップロード")
uploaded_file = st.file_uploader("分析したいボルダリング動画（MP4, MOVなど）を選択してください", type=['mp4', 'mov', 'avi'])

# --- セッション状態の初期化 ---
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0.0
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = ""
if 'crux' not in st.session_state:
    st.session_state.crux = ""
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_sources' not in st.session_state:
    # analysis_sources には Document オブジェクトのリストを保存する
    st.session_state.analysis_sources = []

if uploaded_file is not None:
    # 一時フォルダを作成 (存在しない場合)
    if not os.path.exists(TEMP_VIDEO_DIR):
        os.makedirs(TEMP_VIDEO_DIR)

    # アップロードされたファイルを一時フォルダに保存
    temp_file_path = os.path.join(TEMP_VIDEO_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"動画 '{uploaded_file.name}' をアップロードしました。")

    # 動画の長さを取得
    try:
        clip = VideoFileClip(temp_file_path)
        video_duration = clip.duration
        clip.close() # リソースを解放
        st.info(f"動画の長さ: {video_duration:.2f} 秒")
    except Exception as e:
        st.error(f"動画情報の取得に失敗しました: {e}")
        video_duration = None # エラー時はdurationをNoneに

    # 動画を表示
    st.video(temp_file_path, start_time=int(st.session_state.start_time)) # スライダーと連動して開始位置を設定

    if video_duration is not None:
        st.header("2. 分析開始位置を指定")

        # スライダーの最大値は動画の最後とする
        # (ただし、ごく短い動画の場合に min_value == max_value を避けるため、わずかに小さくする)
        # epsilon = 1e-6 # 微小量
        # max_slider_value = max(epsilon, video_duration - epsilon)
        # → シンプルに video_duration でよさそう。Streamlitがよしなにするはず。
        max_slider_value = video_duration

        # スライダーの表示範囲が有効かチェック (video_durationが0に近い場合)
        if max_slider_value > 0.0:
            start_time = st.slider(
                "分析を開始する秒数を選択してください",
                min_value=0.0,
                max_value=max_slider_value,
                # valueがmax_valueを超えないように調整
                value=min(st.session_state.start_time, max_slider_value),
                step=0.1,
                format="%.1f"
            )
            # スライダーの値をセッション状態に保存
            st.session_state.start_time = start_time

            # 分析終了時間を計算 (開始時間+5秒 or 動画の最後)
            end_time = min(start_time + 5.0, video_duration)

            st.write(f"分析範囲: {start_time:.1f} 秒 〜 {end_time:.1f} 秒")

            # --- ユーザー入力欄を追加 ---
            st.subheader("課題情報の入力 (任意)")
            problem_type_input = st.text_input(
                "課題の種類 (例: スラブ、強傾斜、バランス系、パワー系)",
                value=st.session_state.problem_type
            )
            crux_input = st.text_area(
                "難しいと感じるポイント (例: 〇〇のホールドへのデッドポイント、ヒールフックが抜ける、最後のランジ)",
                value=st.session_state.crux,
                height=100
            )
            # 入力値をセッション状態に保存 (ボタン押下時に利用)
            st.session_state.problem_type = problem_type_input
            st.session_state.crux = crux_input
            # --------------------------

            if st.button("この範囲で分析を開始"):
                st.info(f"{start_time:.1f}秒から{end_time:.1f}秒までの分析を開始します...")
                st.session_state.analysis_result = None
                st.session_state.analysis_sources = []
                frames = []
                with st.spinner('フレームを抽出中...'):
                    frames = extract_frames(temp_file_path, start_time, end_time)

                if frames:
                    st.success(f"{len(frames)} フレームの抽出に成功しました。")
                    st.subheader("抽出されたフレーム")
                    
                    # --- フレームを横に並べて表示 --- 
                    cols_per_row = 4 # 1行あたりに表示するフレーム数
                    # フレームリストをcols_per_row個ずつのチャンクに分割
                    frame_chunks = [frames[i:i + cols_per_row] for i in range(0, len(frames), cols_per_row)]
                    
                    # enumerate を使ってチャンクのインデックスを取得
                    for chunk_index, chunk in enumerate(frame_chunks):
                        # 各チャンクに対して列を作成
                        cols = st.columns(len(chunk)) # チャンクの要素数で列を作成
                        for i, frame in enumerate(chunk):
                            with cols[i]: # 対応する列に画像を表示
                                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                                # より安全なインデックス取得 (enumerateの結果を利用)
                                global_index = chunk_index * cols_per_row + i
                                st.image(
                                    rotated_frame, 
                                    caption=f"フレーム {global_index + 1} (回転後)", 
                                    channels="BGR", 
                                    use_container_width=True
                                )
                    # -----------------------------------

                    # --- RAG分析の実行 (ChromaDB使用) ---
                    st.subheader("3. アドバイス生成")
                    openai_api_key = get_openai_api_key()
                    if openai_api_key:
                        with st.spinner('AIがアドバイスを生成中...'):
                            # advice と source_docs を受け取る
                            advice, sources = get_advice_from_frames(
                                frames,
                                openai_api_key,
                                st.session_state.problem_type,
                                st.session_state.crux
                            )
                            st.session_state.analysis_result = advice
                            st.session_state.analysis_sources = sources # ソースドキュメントリストを保存
                    else:
                        st.error("OpenAI APIキーが設定されていないため、分析を実行できません。")
                else:
                    st.error("フレームの抽出に失敗しました。")
        else:
            # 動画長がほぼ0の場合
            st.warning("動画が短すぎるため、分析範囲を選択できません。")
            # start_timeをリセットしておく
            st.session_state.start_time = 0.0

    # --- 分析結果の表示 ---
    if st.session_state.analysis_result:
        st.subheader("💡 AIからのアドバイス")
        st.markdown(st.session_state.analysis_result)

        # --- 参照ソースの表示 (デバッグモード時のみ) ---
        if st.session_state.debug_mode and st.session_state.analysis_sources:
            st.subheader("📚 参照した知識ソース (デバッグ用)")
            # 各ソースドキュメントを展開表示
            for i, doc in enumerate(st.session_state.analysis_sources):
                source_name = os.path.basename(doc.metadata.get('source', '不明なソース'))
                with st.expander(f"ソース {i+1}: `{source_name}`"):
                    st.text(doc.page_content)
                    # メタデータ全体も表示（デバッグ用）
                    # st.json(doc.metadata)

else:
    st.info("動画ファイルをアップロードしてください。") 