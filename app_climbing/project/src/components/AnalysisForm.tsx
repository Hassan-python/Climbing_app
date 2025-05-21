import { useState, useEffect, useCallback, memo } from 'react';
import { Play } from 'lucide-react';
import { VideoInfo, AnalysisSettings, AnalysisProgress } from '../types';
import { useTranslation } from 'react-i18next';

interface AnalysisFormProps {
  videoInfo: VideoInfo | null;
  onSubmit: (settings: AnalysisSettings) => void;
  isAnalyzing: boolean;
  progress: AnalysisProgress;
}

const AnalysisForm = memo(({ videoInfo, onSubmit, isAnalyzing, progress }: AnalysisFormProps) => {
  const { t } = useTranslation();
  const [problemType, setProblemType] = useState('');
  const [crux, setCrux] = useState('');
  const [startTime, setStartTime] = useState(0);
  
  useEffect(() => {
    if (videoInfo) {
      setStartTime(0);
    }
  }, [videoInfo]);
  
  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    
    if (!videoInfo?.gcsBlobName) {
      return;
    }
    
    onSubmit({
      problemType,
      crux,
      startTime,
      gcsBlobName: videoInfo.gcsBlobName
    });
  }, [problemType, crux, startTime, onSubmit, videoInfo]);

  const getProgressMessage = () => {
    switch (progress.stage) {
      case 'upload':
        return t('uploader.processing');
      case 'compression':
        return t('uploader.compressing');
      case 'analysis':
        return t('analysis.button.analyzing');
      default:
        return '';
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-sm p-6">
      <div className="space-y-4">
        <div>
          <label htmlFor="problem-type" className="block text-sm font-medium text-gray-700">
            {t('analysis.problemType.label')}
          </label>
          <input
            type="text"
            id="problem-type"
            value={problemType}
            onChange={(e) => setProblemType(e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500 sm:text-sm"
            placeholder={t('analysis.problemType.placeholder')}
          />
        </div>
        
        <div>
          <label htmlFor="crux" className="block text-sm font-medium text-gray-700">
            {t('analysis.crux.label')}
            <span className="block mt-1 text-xs text-gray-500">
              {t('analysis.crux.description')}
            </span>
          </label>
          <textarea
            id="crux"
            value={crux}
            onChange={(e) => setCrux(e.target.value)}
            rows={3}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500 sm:text-sm"
            placeholder={t('analysis.crux.placeholder')}
          />
        </div>
        
        <div>
          <label htmlFor="start-time" className="block text-sm font-medium text-gray-700">
            {t('analysis.startTime.label')}
          </label>
          <div className="mt-1 flex items-center">
            <input
              type="range"
              id="start-time"
              min={0}
              max={videoInfo?.duration || 0}
              step={0.1}
              value={startTime}
              onChange={(e) => setStartTime(parseFloat(e.target.value))}
              className={`block w-full mr-3 ${!videoInfo ? 'opacity-50' : ''}`}
              disabled={!videoInfo}
            />
            <span className="text-sm text-gray-500 w-16 text-right">
              {startTime.toFixed(1)}{t('analysis.startTime.seconds')}
            </span>
          </div>
        </div>
        
        <div className="mt-6">
          <button
            type="submit"
            disabled={isAnalyzing || !videoInfo?.gcsBlobName}
            className={`w-full flex items-center justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white 
              ${(isAnalyzing || !videoInfo?.gcsBlobName)
                ? 'bg-emerald-600' 
                : 'bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500'}
            `}
          >
            {isAnalyzing ? (
              <div className="w-full space-y-2">
                <div className="flex items-center justify-center">
                  <span className="text-sm font-medium">{getProgressMessage()}</span>
                </div>
                <div className="w-full bg-emerald-500/30 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-white h-2 rounded-full transition-all duration-300 ease-in-out animate-pulse"
                    style={{ width: `${progress.progress}%` }}
                  ></div>
                </div>
              </div>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                {t('analysis.button.start')}
              </>
            )}
          </button>
        </div>
      </div>
    </form>
  );
});

AnalysisForm.displayName = 'AnalysisForm';

export default AnalysisForm;