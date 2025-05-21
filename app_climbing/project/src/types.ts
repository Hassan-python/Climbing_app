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
  retrievedKnowledge?: string;
  isComplete?: boolean;
}

export interface AnalysisProgress {
  stage: 'upload' | 'compression' | 'analysis';
  progress: number;
}

export interface StreamAnalysisResponse {
  type: 'advice' | 'complete' | 'error' | 'warning';
  content?: string;

  // Fields for 'complete' type event
  advice?: string;
  sources?: Source[];
  geminiAnalysis?: string | null;
  retrievedKnowledge?: string;
  isComplete?: boolean;

  // Fields for 'error' and 'warning' types
  message?: string;
}