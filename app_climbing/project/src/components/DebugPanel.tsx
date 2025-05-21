import { Bug, Book, Activity, Database } from 'lucide-react';
import { useState } from 'react';
import { Source } from '../types';
import { useTranslation } from 'react-i18next';

interface DebugPanelProps {
  geminiAnalysis: string | null;
  sources: Source[];
  retrievedKnowledge?: string;
}

const DebugPanel = ({ geminiAnalysis, sources, retrievedKnowledge }: DebugPanelProps) => {
  const { t } = useTranslation();
  const [expandedSource, setExpandedSource] = useState<number | null>(null);
  
  const toggleSource = (index: number) => {
    if (expandedSource === index) {
      setExpandedSource(null);
    } else {
      setExpandedSource(index);
    }
  };
  
  return (
    <div className="mt-8 space-y-6">
      {/* Progress Section */}
      <div>
        <div className="flex items-center mb-3">
          <Activity className="mr-2 h-5 w-5 text-emerald-500" />
          <h3 className="text-lg font-medium">{t('debug.progress.title')}</h3>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
          <div className="space-y-2">
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">{t('debug.progress.videoUpload')}</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">{t('debug.progress.frameExtraction')}</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">{t('debug.progress.geminiAnalysis')}</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">{t('debug.progress.adviceGeneration')}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Retrieved Knowledge */}
      {retrievedKnowledge && (
        <div>
          <div className="flex items-center mb-3">
            <Database className="mr-2 h-5 w-5 text-indigo-500" />
            <h3 className="text-lg font-medium">{t('debug.retrievedKnowledge.title')}</h3>
          </div>
          <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
            <pre className="whitespace-pre-wrap text-sm text-gray-800">{retrievedKnowledge}</pre>
          </div>
        </div>
      )}

      {/* Gemini Analysis */}
      {geminiAnalysis && (
        <div>
          <div className="flex items-center mb-3">
            <Bug className="mr-2 h-5 w-5 text-purple-500" />
            <h3 className="text-lg font-medium">{t('debug.geminiAnalysis.title')}</h3>
          </div>
          <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
            <pre className="whitespace-pre-wrap text-sm text-gray-800">{geminiAnalysis}</pre>
          </div>
        </div>
      )}
      
      {/* Knowledge Sources */}
      {sources.length > 0 && (
        <div>
          <div className="flex items-center mb-3">
            <Book className="mr-2 h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-medium">{t('debug.sources.title')}</h3>
          </div>
          
          <div className="space-y-2">
            {sources.map((source, index) => (
              <div key={index} className="bg-gray-50 border border-gray-200 rounded-md">
                <button
                  onClick={() => toggleSource(index)}
                  className="flex items-center justify-between w-full px-4 py-3 text-left"
                >
                  <span className="font-medium text-sm">{t('debug.sources.source')} {index + 1}: {source.name}</span>
                  <span className="text-gray-500">
                    {expandedSource === index ? '▲' : '▼'}
                  </span>
                </button>
                
                {expandedSource === index && (
                  <div className="px-4 py-3 border-t border-gray-200">
                    <pre className="whitespace-pre-wrap text-sm text-gray-700">{source.content}</pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DebugPanel;