export interface VideoInfo {
  file: File;
  url: string;
  duration: number;
  name: string;
  gcsBlobName?: string;
}

export interface AnalysisSettings {
  problemType: string;
  crux: string;
  startTime: number;
  gcsBlobName: string;
}

export interface Source {
  name: string;
  content: string;
}

export interface AnalysisResponse {
  advice: string;
  sources: Source[];
  geminiAnalysis: string | null;
}

export interface AnalysisProgress {
  stage: 'upload' | 'compression' | 'analysis';
  progress: number;
}