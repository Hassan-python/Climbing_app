from fastapi import FastAPI, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
from typing import Optional, List, Dict, Any, Tuple
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from PIL import Image
from google.cloud import storage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from functools import lru_cache

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
ANALYSIS_INTERVAL_SEC = 0.5
MAX_FRAMES_FOR_GEMINI = 10
CHROMA_COLLECTION_NAME = "bouldering_advice"
DEFAULT_RETRIEVAL_K = 3

class AnalysisSettings(BaseModel):
    problemType: str
    crux: str
    startTime: float
    gcsBlobName: str

class Source(BaseModel):
    name: str
    content: str

class AnalysisResponse(BaseModel):
    advice: str
    sources: list[Source]
    geminiAnalysis: Optional[str] = None

def extract_frames(video_path: str, start_sec: float, end_sec: float, interval_sec: float = ANALYSIS_INTERVAL_SEC) -> list:
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    interval_frames = max(1, int(interval_sec * fps))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if (current_frame - start_frame) % interval_frames == 0:
            frames.append(frame)
            
        current_frame += 1
        
    cap.release()
    return frames

@lru_cache(maxsize=1)
def get_chroma_client():
    chromadb_url = os.getenv("CHROMA_DB_URL")
    if not chromadb_url:
        raise HTTPException(status_code=500, detail="ChromaDB URL not configured")
        
    try:
        settings = Settings(chroma_api_impl="rest")
        client = chromadb.HttpClient(host=chromadb_url, settings=settings)
        return client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB connection failed: {str(e)}")

@lru_cache(maxsize=1)
def get_langchain_chroma_vectorstore() -> Chroma:
    """Langchain経由でChromaベクターストアを取得する"""
    try:
        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        chroma_client = get_chroma_client() 
        
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=gemini_embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error creating Langchain Chroma vectorstore: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize vectorstore: {str(e)}")

def retrieve_from_chroma_langchain(query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[dict]:
    """Langchainラッパーを使用してChromaDBから関連ドキュメントを取得する"""
    try:
        vectorstore = get_langchain_chroma_vectorstore()
        source_docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        documents = []
        print(f"[DEBUG Langchain Chroma] Executing query: {query} with k={k}")
        for i, (doc, score) in enumerate(source_docs_with_scores):
            doc_name = doc.metadata.get("name", f"doc_{i+1}") if doc.metadata else f"doc_{i+1}"
            documents.append({
                "name": doc_name,
                "content": doc.page_content,
                "score": score
            })
            print(f"[DEBUG Langchain Chroma] Retrieved doc: {doc_name}, Score: {score:.4f}, Content (first 50 chars): {doc.page_content[:50]}...")
        
        return documents
    except Exception as e:
        print(f"Langchain Chroma retrieval error: {e}")
        return []

def analyze_and_generate_advice(
    frames: list, 
    problem_type: str, 
    crux: str, 
    output_language: str
) -> Tuple[str, str, List[Source]]:
    """1回のGemini呼び出しで動画分析とアドバイス生成を行う"""
    if not frames:
        return "No frames available for analysis", "アドバイスを生成できません", []
        
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Select frames for analysis
        num_frames = min(len(frames), MAX_FRAMES_FOR_GEMINI)
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]
        
        # Convert frames to PIL images
        pil_images = []
        for frame in selected_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_images.append(pil_image)
            
        # ChromaDBから関連情報を検索 (ユーザーのテキスト入力のみを使用)
        rag_query = f"課題の種類: {problem_type}, 難しい点: {crux}"
        retrieved_docs_for_gemini = retrieve_from_chroma_langchain(rag_query)
        
        # Format retrieved_knowledge for prompt as per FR-001 and FR-002
        formatted_knowledge_parts = []
        if retrieved_docs_for_gemini:
            for i, doc in enumerate(retrieved_docs_for_gemini):
                # Using the name from metadata if available, otherwise a generic one
                source_name = doc.get("name", f"知識{i+1}") 
                formatted_knowledge_parts.append(f"[知識{i+1}: {source_name}]\n{doc['content']}")
            retrieved_knowledge_for_prompt = "\n\n".join(formatted_knowledge_parts)
        else:
            retrieved_knowledge_for_prompt = "関連する知識は見つかりませんでした。"
            
        # output_language の値に基づいてプロンプトを構築
        if output_language == "English":
            prompt = f"""**
        ### Generation Rules (Must Follow) ###
        Your response MUST be written in **English**. Do not use any other languages.

        ### Role ###
        You are an expert climbing movement analyst and an experienced professional bouldering coach.

        ### Instructions ###
        1. Analyze the provided series of images (frames from a bouldering attempt) and identify the climber's posture, balance, hand/foot positions and movements, inefficiencies, and any unstable elements that could lead to a fall.
        2. Based on the image analysis, the user's reported situation, and the provided "Related Bouldering Knowledge", generate specific and practical improvement advice for the climber.
        3. Present the advice in a step-by-step format (around 3 steps) so the climber can make incremental improvements.
        4. **Absolutely DO NOT include the source of the referenced knowledge (e.g., [Knowledge 1: doc_1]) in the generated advice.** Utilize the knowledge only to inform the advice content; do not cite the sources.

        ### Language and Format for Response Generation ###
        - Your response MUST be in English.
        - Even if no relevant bouldering knowledge is found, provide the best possible advice based on the image analysis and the user's situation.
        - Clearly divide your response into the following two sections:
          - `# Image Analysis`
          - `# Advice`

        ---
        User's Reported Situation:
        - Type of Problem: {problem_type or "Not specified"}
        - Points of Difficulty: {crux or "Not specified"}
        ---
        ### Related Bouldering Knowledge (Reference information from the database. Use this to inform your advice. Do not cite sources.)
        {retrieved_knowledge_for_prompt}
        ---
        
        Example Response Format:
        
        # Image Analysis
        [Describe objective analysis of the images here]
        
        # Advice
        [Provide coaching advice in about 3 steps here. **Do not cite sources.**]
        """
        elif output_language == "日本語":
            prompt = f"""**
        ###生成時ルール（must rule）###
        あなたの応答は必ず**日本語**で記述してください。他の言語は一切使用しないでください。

        ### 役割 ###
        あなたはクライミングの動きを分析する専門家であり、経験豊富なプロのボルダリングコーチです。

        ### 指示 ###
        1.  提供された一連の画像（ボルダリング中のフレーム）を分析し、クライマーの体勢、バランス、手足の位置と動き、非効率な動きや不安定な要素を特定してください。
        2.  上記の画像分析結果、ユーザーが報告した状況、および提供される「関連するボルダリング知識」を**総合的に考慮**して、具体的で実践的な改善アドバイスを生成してください。
        3.  アドバイスは、クライマーが段階的に改善できるよう、3ステップ程度のステップ形式で提示してください。
        4.  **生成するアドバイスには、参照した知識の出典元（例：[知識1: doc_1]など）を絶対に含めないでください。** 知識はアドバイス内容に活かすのみとし、出典情報は記載しないでください。

        ### 回答生成時の言語およびフォーマット ###
        - 回答は必ず日本語で生成してください。
        - 関連するボルダリング知識が見つからない場合でも、画像分析結果とユーザーの状況に基づいて最適なアドバイスを提供してください。
        - 回答は、以下の2つのセクションに明確に分けて出力してください。
          - `# 画像分析`
          - `# アドバイス`

        ---
        ユーザーが報告した状況:
        - 課題の種類: {problem_type or "特に指定なし"}
        - 難しいと感じるポイント: {crux or "特に指定なし"}
        ---
        ### 関連するボルダリング知識 (データベースより参考情報。アドバイスに活かすこと。出典は記載しないこと。)
        {retrieved_knowledge_for_prompt}
        ---
        
        回答形式の例:
        
        # 画像分析
        [ここに画像の客観的な分析結果を記述]
        
        # アドバイス
        [ここにコーチとしてのアドバイスを3ステップ程度で記述。**出典元は記載しない。**]
        """
        else: # デフォルトは英語プロンプト (念のため)
            prompt = f"""**
        ### Generation Rules (Must Follow) ###
        Your response MUST be written in **English**. Do not use any other languages.

        ### Role ###
        You are an expert climbing movement analyst and an experienced professional bouldering coach.

        ### Instructions ###
        1. Analyze the provided series of images (frames from a bouldering attempt) and identify the climber's posture, balance, hand/foot positions and movements, inefficiencies, and any unstable elements that could lead to a fall.
        2. Based on the image analysis, the user's reported situation, and the provided "Related Bouldering Knowledge", generate specific and practical improvement advice for the climber.
        3. Present the advice in a step-by-step format (around 3 steps) so the climber can make incremental improvements.
        4. **Absolutely DO NOT include the source of the referenced knowledge (e.g., [Knowledge 1: doc_1]) in the generated advice.** Utilize the knowledge only to inform the advice content; do not cite the sources.

        ### Language and Format for Response Generation ###
        - Your response MUST be in English.
        - Even if no relevant bouldering knowledge is found, provide the best possible advice based on the image analysis and the user's situation.
        - Clearly divide your response into the following two sections:
          - `# Image Analysis`
          - `# Advice`

        ---
        User's Reported Situation:
        - Type of Problem: {problem_type or "Not specified"}
        - Points of Difficulty: {crux or "Not specified"}
        ---
        ### Related Bouldering Knowledge (Reference information from the database. Use this to inform your advice. Do not cite sources.)
        {retrieved_knowledge_for_prompt}
        ---
        
        Example Response Format:
        
        # Image Analysis
        [Describe objective analysis of the images here]
        
        # Advice
        [Provide coaching advice in about 3 steps here. **Do not cite sources.**]
        """ # ここに三重引用符が抜けていたので補完しました
        
        print(f"[DEBUG] Prompt to Gemini: {prompt}") 
        response = model.generate_content([prompt, *pil_images])
        full_response = response.text
        print(f"[DEBUG] Full response from Gemini: {full_response}") # Geminiからの生のレスポンスをログ出力
        
        # レスポンスを分析とアドバイスに分割
        try:
            analysis_part = ""
            advice_part = ""
            
            if "# 画像分析" in full_response and "# アドバイス" in full_response:
                parts = full_response.split("# アドバイス")
                analysis_part = parts[0].replace("# 画像分析", "").strip()
                advice_part = parts[1].strip()
            else:
                # セクションが明確に分かれていない場合
                analysis_part = "分析結果を抽出できませんでした"
                advice_part = full_response
            
            return analysis_part, advice_part, [Source(name=doc["name"], content=doc["content"]) for doc in retrieved_docs_for_gemini]
        except Exception as e:
            print(f"Response parsing error: {e}")
            # エラー時は分析結果、アドバイス、空のソースリストを返す
            if "\n\n" in full_response:
                 analysis_part, advice_part = full_response.split("\n\n", 1) # 最初の区切りで分割
            else:
                 analysis_part = full_response
                 advice_part = "アドバイスの抽出に失敗しました"
            return analysis_part, advice_part, []
        
    except Exception as e:
        print(f"Gemini analysis and advice generation error: {e}")
        return "画像分析中にエラーが発生しました", "アドバイス生成中にエラーが発生しました", []

@app.post("/upload")
async def upload_video(video: UploadFile):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")
    if not GCS_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME not configured")

    # Generate unique filename for GCS
    video_id = str(uuid.uuid4())
    file_extension = os.path.splitext(video.filename)[1]
    gcs_blob_name = f"videos/{video_id}{file_extension}"

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_blob_name)

        # Save uploaded file to GCS
        content = await video.read()
        blob.upload_from_string(content, content_type=video.content_type)

        # Verify it's a valid video and check duration
        temp_local_path = f"/tmp/{video_id}{file_extension}"
        blob.download_to_filename(temp_local_path)

        with VideoFileClip(temp_local_path) as clip:
            if clip.duration > 5.0:
                os.remove(temp_local_path)
                blob.delete()
                raise HTTPException(status_code=400, detail="Video must be 5 seconds or shorter")
        
        os.remove(temp_local_path)

        return {"gcsBlobName": gcs_blob_name, "videoId": video_id}

    except Exception as e:
        if 'blob' in locals() and blob.exists():
            try:
                blob.delete()
            except Exception as delete_e:
                print(f"Error deleting blob during cleanup: {delete_e}")

        if 'temp_local_path' in locals() and os.path.exists(temp_local_path):
            os.remove(temp_local_path)
            
        print(f"Upload error: {e}") 
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(settings: AnalysisSettings, x_language: Optional[str] = Header(None, alias="X-Language")):
    if not settings.gcsBlobName:
        raise HTTPException(status_code=400, detail="gcsBlobName must be provided in settings")
    if not GCS_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="GCS_BUCKET_NAME not configured")

    temp_local_path = f"/tmp/{os.path.basename(settings.gcsBlobName)}" 

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(settings.gcsBlobName)

        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"Video blob {settings.gcsBlobName} not found in GCS")

        blob.download_to_filename(temp_local_path)

        with VideoFileClip(temp_local_path) as clip:
            end_time = min(settings.startTime + 1.0, clip.duration)

        frames = extract_frames(temp_local_path, settings.startTime, end_time)
        
        # 言語設定の取得 (FR-001, FR-002, TR-001)
        output_language = "English" # Default to English
        print(f"[DEBUG] Received X-Language header: {x_language}")
        if x_language:
            if x_language.lower().startswith("ja"):
                output_language = "日本語"
            elif x_language.lower().startswith("en"):
                output_language = "English"
            # 上記以外の場合はデフォルトの「英語」のまま
        print(f"[DEBUG] Determined output_language: {output_language}")
        
        # 1回のGemini呼び出しで分析とアドバイス生成、RAG結果取得を行う
        gemini_analysis, final_advice, retrieved_sources = analyze_and_generate_advice(
            frames,
            settings.problemType,
            settings.crux,
            output_language
        )
                
        if os.path.exists(temp_local_path):
            os.remove(temp_local_path)
            
        return AnalysisResponse(
            advice=final_advice,
            sources=retrieved_sources,
            geminiAnalysis=gemini_analysis
        )
        
    except Exception as e:
        if os.path.exists(temp_local_path):
            os.remove(temp_local_path)
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze video: {str(e)}")

@app.get("/chroma-status")
async def check_chroma_status():
    try:
        vectorstore = get_langchain_chroma_vectorstore()
        dummy_search_query = "test query"
        dummy_docs = vectorstore.similarity_search_with_score(dummy_search_query, k=1)
        count = vectorstore._collection.count()
        
        if dummy_docs:
            print(f"[Chroma Status] Dummy search retrieved {len(dummy_docs)} doc(s). First doc score: {dummy_docs[0][1] if dummy_docs else 'N/A'}")
        else:
            print("[Chroma Status] Dummy search retrieved no documents.")
            
        return {"status": f"✅ ChromaDB(Langchain) 接続成功 (`{CHROMA_COLLECTION_NAME}`: {count} アイテム)"}
    except Exception as e:
        print(f"❌ ChromaDB(Langchain) connection failed: {str(e)}")
        return {"status": f"❌ ChromaDB(Langchain) connection failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)