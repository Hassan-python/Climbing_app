import axios, { AxiosProgressEvent } from 'axios';
import { AnalysisSettings, AnalysisResponse, VideoInfo, AnalysisProgress, StreamAnalysisResponse } from '../types';
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
    let errorMessage = 'Failed to upload video';
    if (axios.isAxiosError(error) && error.response?.data?.detail) {
      const detail = error.response.data.detail;
      if (typeof detail === 'string' && detail.includes('5 seconds or shorter')) {
        errorMessage = 'Video must be 5 seconds or shorter';
      }
    }
    throw new Error(errorMessage);
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
      reject('Failed to load video metadata');
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
    console.log('Starting analysis with settings:', settings);
    
    // Simulate compression progress
    for (let i = 0; i <= 100; i += 10) {
      onProgress?.({ stage: 'compression', progress: i });
      await new Promise(resolve => setTimeout(resolve, 200));
    }

    const eventSource = new EventSource(`${API_BASE_URL}/analyze/stream?${new URLSearchParams({
      gcsBlobName: settings.gcsBlobName,
      problemType: settings.problemType,
      crux: settings.crux,
      startTime: settings.startTime.toString()
    })}`);

    return new Promise((resolve, reject) => {
      let fullAdvice = '';
      let sources = [];
      let geminiAnalysis = null;
      let retrievedKnowledge;

      eventSource.onmessage = (event) => {
        try {
          const data: StreamAnalysisResponse = JSON.parse(event.data);
          
          if (data.type === 'advice' && data.content) {
            fullAdvice += data.content;
            onAdviceChunk?.(data.content);
            onProgress?.({ stage: 'analysis', progress: 50 });
          } else if (data.type === 'complete') {
            onProgress?.({ stage: 'analysis', progress: 100 });
            eventSource.close();
            resolve({
              advice: fullAdvice,
              sources: data.sources || [],
              geminiAnalysis: data.geminiAnalysis || null,
              retrievedKnowledge: data.retrievedKnowledge,
              isComplete: true
            });
          } else if (data.type === 'error' || data.type === 'warning') {
            if (data.type === 'warning') {
              console.warn('Analysis warning:', data.message);
              return;
            }
            eventSource.close();
            reject(new Error(data.message || 'Analysis failed'));
          }
        } catch (error) {
          console.error('Error parsing SSE data:', error);
          eventSource.close();
          reject(new Error('Failed to parse analysis data'));
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        eventSource.close();
        reject(new Error('Connection error: Failed to communicate with server'));
      };
    });
  } catch (error) {
    console.error('Analysis error:', error);
    throw error instanceof Error ? error : new Error('An unexpected error occurred during analysis');
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
      throw new Error('Failed to get blob name for analysis');
    }

    const result = await analyzeVideo({
      ...settings,
      gcsBlobName: currentGcsBlobName
    }, onProgress, onAdviceChunk);
    
    console.log('Analysis completed successfully');
    return result;
  } catch (error) {
    console.error('Upload and analyze process failed:', error);
    throw error instanceof Error ? error : new Error('Failed to process video');
  }
};