import axios from 'axios';
import { AnalysisSettings, AnalysisResponse, VideoInfo, AnalysisProgress } from '../types';
import i18next from 'i18next';

const API_BASE_URL = 'https://my-fastapi-service-932280363930.asia-northeast1.run.app';

const getHeaders = () => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Language': i18next.language || 'ja',
  };

  const geminiApiKey = localStorage.getItem('geminiApiKey');
  const openaiApiKey = localStorage.getItem('openaiApiKey');
  const openaiModel = localStorage.getItem('openaiModel');

  if (geminiApiKey) headers['X-Gemini-Key'] = geminiApiKey;
  if (openaiApiKey) headers['X-OpenAI-Key'] = openaiApiKey;
  if (openaiModel) headers['X-OpenAI-Model'] = openaiModel;

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
      onUploadProgress: (progressEvent) => {
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
    if (axios.isAxiosError(error)) {
      console.error('Upload error details:', {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      });
      throw new Error(error.response?.data?.detail || error.response?.data?.message || 'Failed to upload video');
    }
    throw new Error('Failed to upload video');
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
  onProgress?: (progress: AnalysisProgress) => void
): Promise<AnalysisResponse> => {
  try {
    console.log('Sending analysis request:', settings);
    
    // Simulate compression progress
    for (let i = 0; i <= 100; i += 10) {
      onProgress?.({ stage: 'compression', progress: i });
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    // Simulate analysis progress
    const response = await axios.post(`${API_BASE_URL}/analyze`, settings, {
      headers: getHeaders(),
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress?.({ stage: 'analysis', progress });
        }
      },
    });
    
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Analysis error details:', {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message,
        config: {
          url: error.config?.url,
          method: error.config?.method,
          data: error.config?.data
        }
      });
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message;
      throw new Error(`Analysis failed: ${errorMessage}`);
    }
    console.error('Non-Axios error during analysis:', error);
    throw new Error('An unexpected error occurred during analysis');
  }
};

export const uploadAndAnalyze = async (
  videoInfo: VideoInfo,
  settings: AnalysisSettings,
  onProgress?: (progress: AnalysisProgress) => void
): Promise<AnalysisResponse> => {
  try {
    console.log('Starting upload and analysis process:', {
      fileName: videoInfo.file.name,
      fileSize: videoInfo.file.size,
      settings
    });
    
    if (!videoInfo.gcsBlobName) {
      onProgress?.({ stage: 'upload', progress: 0 });
      const uploadedInfo = await uploadVideo(videoInfo.file, (progress) => {
        onProgress?.({ stage: 'upload', progress });
      });
      videoInfo = { ...videoInfo, gcsBlobName: uploadedInfo.gcsBlobName };
    }
    
    const result = await analyzeVideo({
      ...settings,
      gcsBlobName: videoInfo.gcsBlobName
    }, onProgress);
    
    console.log('Analysis completed successfully');
    return result;
  } catch (error) {
    console.error('Upload and analyze process failed:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to process video');
  }
};