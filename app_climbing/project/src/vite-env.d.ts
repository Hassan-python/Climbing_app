export interface StreamAnalysisResponse {
    type: 'advice' | 'complete' | 'error' | 'warning';
    content?: string; // For 'advice' type
    advice?: string; // For 'complete' type (full advice text)
    sources?: Source[]; // For 'complete' type
    geminiAnalysis?: string | null; // For 'complete' type
    retrievedKnowledge?: string; // For 'complete' type
    isComplete?: boolean; // For 'complete' type
    message?: string; // For 'error' and 'warning' types
  }