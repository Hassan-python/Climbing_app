import axios, { AxiosProgressEvent } from 'axios';
import { AnalysisSettings, AnalysisResponse, VideoInfo, AnalysisProgress, StreamAnalysisResponse, Source } from '../types';
import i18next from 'i18next';

const API_BASE_URL = 'https://my-fastapi-service-932280363930.asia-northeast1.run.app';

const getHeaders = () => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Language': i18next.language || 'ja',
  };

  const geminiApiKey = localStorage.getItem('geminiApiKey');
  if (geminiApiKey) headers['X-Gemini-Key'] = geminiApiKey;

  return headers;
};

export const uploadVideo = async (
  file: File,
  onProgress?: (progress: number) => void
): Promise<VideoInfo> => {
  const formData = new FormData();
  formData.append('video', file);

  try {
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        ...getHeaders(),
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent: AxiosProgressEvent) => {
        if (progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress?.(percentCompleted);
        }
      },
    });

    const url = URL.createObjectURL(file);
    const duration = await getVideoDuration(url);

    return {
      file,
      url,
      duration,
      name: file.name,
      gcsBlobName: response.data.gcsBlobName
    };
  } catch (error) {
    let status, responseData, errorMessage, requestConfig;
    if (axios.isAxiosError(error)) {
      status = error.response?.status;
      responseData = error.response?.data;
      errorMessage = error.message;
      requestConfig = {
        url: error.config?.url,
        method: error.config?.method,
        headers: error.config?.headers,
      };
      console.error('Upload error details:', { status, data: responseData, message: errorMessage, config: requestConfig });

      const errorDetail = error.response?.data?.detail;
      if (errorDetail) {
        const detailString = typeof errorDetail === 'string' ? errorDetail : JSON.stringify(errorDetail);
        if (detailString.includes('5 seconds or shorter')) {
          throw new Error('動画は5秒以下である必要があります');
        } else if (detailString.includes('No video file provided')) {
          throw new Error('動画ファイルを選択してください');
        }
        throw new Error(detailString);
      }
      throw new Error('動画のアップロードに失敗しました');
    } else if (error instanceof Error) {
      console.error('Non-Axios upload error:', error.message);
      throw new Error(`動画のアップロード中に予期せぬエラーが発生しました: ${error.message}`);
    } else {
      console.error('Unknown upload error:', error);
      throw new Error('動画のアップロード中に不明なエラーが発生しました');
    }
  }
};

const getVideoDuration = (url: string): Promise<number> => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    
    video.onloadedmetadata = () => {
      resolve(video.duration);
      video.remove();
    };
    
    video.onerror = () => {
      reject('動画のメタデータの読み込みに失敗しました');
      video.remove();
    };
    
    video.src = url;
  });
};

export const analyzeVideo = async (
  settings: AnalysisSettings,
  onProgress?: (progress: AnalysisProgress) => void,
  onAdviceChunk?: (chunk: string) => void
): Promise<AnalysisResponse> => {
  try {
    console.log('Sending analysis request:', settings);
    
    // Simulating initial progress for better UX, can be removed or adjusted
    for (let i = 0; i <= 20; i += 5) { // Reduced initial simulated progress
      onProgress?.({ stage: 'preparation', progress: i });
      await new Promise(resolve => setTimeout(resolve, 50)); 
    }

    const eventSource = new EventSource(`${API_BASE_URL}/analyze/stream?${new URLSearchParams({
      gcsBlobName: settings.gcsBlobName,
      problemType: settings.problemType,
      crux: settings.crux,
      startTime: settings.startTime.toString()
    })}`);

    return new Promise((resolve, reject) => {
      let fullAdvice = '';
      let sources: Source[] = [];
      let geminiAnalysis: string | null = null;
      let retrievedKnowledge: string | undefined;
      // let typingDelay = 50; // Not directly used in char-by-char
      let lastChunkTime = Date.now(); // Still useful for managing potential minimum delays if needed elsewhere
      // const minChunkDelay = 30; // Not directly used in char-by-char

      eventSource.onmessage = async (event) => {
        try {
          const data: StreamAnalysisResponse = JSON.parse(event.data);
          
          if (data.type === 'advice' && data.content) {
            if (data.content.trim()) {
              fullAdvice += data.content; // Maintain full advice for final resolution
              
              for (const char of data.content) {
                onAdviceChunk?.(char); // Send character to UI
                // A minimal delay to allow the event loop to process and UI to update.
                await new Promise(resolve_inner => setTimeout(resolve_inner, 5)); // Small delay (e.g., 5ms)
              }
            }
            // Optional: Update progress more meaningfully if possible
            // onProgress?.({ stage: 'analysis', progress: 50 }); // Example: Update less frequently
          } else if (data.type === 'complete') {
            onProgress?.({ stage: 'analysis', progress: 100 });
            eventSource.close();
            resolve({
              advice: fullAdvice, 
              sources: data.sources || [], 
              geminiAnalysis: data.geminiAnalysis !== undefined ? data.geminiAnalysis : null, 
              retrievedKnowledge: data.retrievedKnowledge, 
              isComplete: data.isComplete !== undefined ? data.isComplete : true 
            });
          } else if (data.type === 'error' || data.type === 'warning') {
            console.warn(`SSE ${data.type} event:`, data.message);
            if (data.type === 'error') {
              eventSource.close();
              reject(new Error(data.message || '分析中にエラーが発生しました'));
            }
          }
        } catch (error) {
          console.error('Error parsing SSE data:', error, "Raw data:", event.data);
          eventSource.close();
          reject(new Error('分析データの解析に失敗しました'));
        }
      };

      eventSource.onerror = (event) => {
        console.error('SSE Error (event object):', event);
        eventSource.close();
        
        const errorEvent = event as ErrorEvent;
        const errorMessage = errorEvent.message || 
                           (errorEvent.error && (errorEvent.error as Error).message) || // Type assertion for errorEvent.error
                           '接続エラー: サーバーとの通信に失敗しました';
        
        if (!navigator.onLine) {
          reject(new Error('インターネット接続が切断されています'));
        } else {
          reject(new Error(errorMessage));
        }
      };
    });
  } catch (error) {
    let status, responseData, errorMessageText, requestConfig; // Renamed errorMessage to errorMessageText
    if (axios.isAxiosError(error)) {
      status = error.response?.status;
      responseData = error.response?.data;
      errorMessageText = error.message; // Use errorMessageText
      requestConfig = {
        url: error.config?.url,
        method: error.config?.method,
        data: error.config?.data,
        headers: error.config?.headers,
      };
      console.error('Analysis error details:', { status, data: responseData, message: errorMessageText, config: requestConfig });

      const errorDetail = error.response?.data?.detail;
      if (errorDetail) {
        const detailString = typeof errorDetail === 'string' ? errorDetail : JSON.stringify(errorDetail);
        if (detailString.includes('ChromaDB')) {
          throw new Error('知識ベースへの接続に失敗しました');
        } else if (detailString.includes('Video blob') && detailString.includes('not found')) {
          throw new Error('動画ファイルが見つかりませんでした');
        }
        throw new Error(`分析エラー: ${detailString}`);
      }
      throw new Error('動画の分析に失敗しました');
    } else if (error instanceof Error) {
      console.error('Non-Axios analysis error:', error.message);
      throw new Error(error.message);
    } else {
      console.error('Unknown analysis error:', error);
      throw new Error('分析中に不明なエラーが発生しました');
    }
  }
};

export const uploadAndAnalyze = async (
  videoInfo: VideoInfo,
  settings: AnalysisSettings,
  onProgress?: (progress: AnalysisProgress) => void,
  onAdviceChunk?: (chunk: string) => void
): Promise<AnalysisResponse> => {
  try {
    console.log('Starting upload and analysis process:', {
      fileName: videoInfo.file.name,
      fileSize: videoInfo.file.size,
      settings
    });
    
    let currentGcsBlobName = videoInfo.gcsBlobName;

    if (!currentGcsBlobName) {
      onProgress?.({ stage: 'upload', progress: 0 });
      const uploadedInfo = await uploadVideo(videoInfo.file, (progress) => {
        onProgress?.({ stage: 'upload', progress });
      });
      currentGcsBlobName = uploadedInfo.gcsBlobName; 
    }
    
    if (!currentGcsBlobName) {
      throw new Error('GCS Blob Name could not be obtained and is required for analysis.');
    }

    onProgress?.({stage: 'analysis', progress: 0 }); // Indicate start of analysis
    const result = await analyzeVideo({
      ...settings,
      gcsBlobName: currentGcsBlobName
    }, onProgress, onAdviceChunk);
    
    console.log('Analysis completed successfully');
    return result;
  } catch (error) {
    console.error('Upload and analyze process failed:', error);
    if (error instanceof Error) {
      throw error; // Re-throw the original error to preserve its type and message
    }
    // Fallback for non-Error objects (though less common in modern JS)
    throw new Error('動画の処理中に予期せぬエラーが発生しました');
  }
}; 