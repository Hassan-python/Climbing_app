from fastapi import FastAPI, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
from typing import Optional
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from PIL import Image
from google.cloud import storage

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
# TEMP_VIDEO_DIR = "temp_videos"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
ANALYSIS_INTERVAL_SEC = 0.5
MAX_FRAMES_FOR_GEMINI = 10
CHROMA_COLLECTION_NAME = "bouldering_advice"

# Create temp directory if it doesn't exist
# os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

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

def analyze_with_gemini(frames: list) -> str:
    if not frames:
        return "No frames available for analysis"
        
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
            
        prompt = """
        あなたはクライミングの動きを分析する専門家です。提供された一連の画像（ボルダリング中のフレーム）を見て、以下の点を具体的かつ簡潔に記述してください。

        - クライマーの体勢やバランス
        - 各フレームでの手足の位置と動き
        - 見受けられる非効率な動きや、落下につながりそうな不安定な要素

        アドバイスではなく、客観的な観察結果のみを記述してください。
        """
        
        response = model.generate_content([prompt, *pil_images])
        return response.text
        
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return "画像分析中にエラーが発生しました"

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
        # Option 1: Download temporarily for validation
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
        
        gemini_analysis = analyze_with_gemini(frames)
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = Chroma(
            client=get_chroma_client(),
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings
        )
        
        search_query = f"課題の種類: {settings.problemType}, 難しい点: {settings.crux}\n画像分析結果: {gemini_analysis[:300]}"
        source_docs = vectorstore.similarity_search(search_query, k=3)
        
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano"),
            temperature=0.5
        )
        
        # Determine the language for the response based on x_language header
        output_language = "英語" # Default to English
        if x_language and x_language.lower().startswith("ja"):
            output_language = "日本語"
        elif x_language and x_language.lower().startswith("en"):
            output_language = "英語"
        # Add more language conditions here if needed

        prompt_template = f"""
        あなたは経験豊富なプロのボルダリングコーチです。以下の情報をすべて考慮して、クライマーへの次のトライで試せるような具体的で実践的な改善アドバイスを生成してください。

        ### 回答生成時ルール (output rules) ###
        - 回答は{output_language}で生成してください。
        
        
        ---
        ユーザーが報告した状況:
        - 課題の種類: {{user_problem_type}}
        - 難しいと感じるポイント: {{user_crux}}
        ---
        AIによる画像分析結果 (客観的な観察):
        {{gemini_analysis}}
        ---
        関連するボルダリング知識 (データベースより):
        {{retrieved_knowledge}}
        ---
        
        上記情報を踏まえた、コーチとしてのアドバイス (回答生成時間短縮のため，3ステップを推奨):
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["user_problem_type", "user_crux", "gemini_analysis", "retrieved_knowledge"]
        )
        
        retrieved_docs_content = "\n\n".join([doc.page_content for doc in source_docs])
        
        formatted_prompt = PROMPT.format(
            user_problem_type=settings.problemType or "特に指定なし",
            user_crux=settings.crux or "特に指定なし",
            gemini_analysis=gemini_analysis,
            retrieved_knowledge=retrieved_docs_content
        )
        
        final_advice = llm.invoke(formatted_prompt).content
        
        if os.path.exists(temp_local_path):
            os.remove(temp_local_path)
            
        return AnalysisResponse(
            advice=final_advice,
            sources=[Source(name=str(i), content=doc.page_content) for i, doc in enumerate(source_docs, 1)],
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
        client = get_chroma_client()
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
        count = collection.count()
        return {"status": f"✅ ChromaDB 接続成功 (`{CHROMA_COLLECTION_NAME}`: {count} アイテム)"}
    except Exception as e:
        return {"status": f"❌ ChromaDB connection failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)