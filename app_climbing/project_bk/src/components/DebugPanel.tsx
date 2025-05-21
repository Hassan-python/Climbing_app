import { Bug, Book, Activity } from 'lucide-react';
import { useState } from 'react';
import { Source } from '../types';

interface DebugPanelProps {
  geminiAnalysis: string;
  sources: Source[];
}

const DebugPanel = ({ geminiAnalysis, sources }: DebugPanelProps) => {
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
          <h3 className="text-lg font-medium">処理状況</h3>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
          <div className="space-y-2">
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">動画アップロード完了</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">フレーム抽出完了</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">Gemini 分析完了</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 rounded-full bg-emerald-500 mr-2"></div>
              <span className="text-sm">アドバイス生成完了</span>
            </div>
          </div>
        </div>
      </div>

      {/* Gemini Analysis */}
      <div>
        <div className="flex items-center mb-3">
          <Bug className="mr-2 h-5 w-5 text-purple-500" />
          <h3 className="text-lg font-medium">Gemini 分析結果 (デバッグ用)</h3>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
          <pre className="whitespace-pre-wrap text-sm text-gray-800">{geminiAnalysis}</pre>
        </div>
      </div>
      
      {/* Knowledge Sources */}
      {sources.length > 0 && (
        <div>
          <div className="flex items-center mb-3">
            <Book className="mr-2 h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-medium">参照した知識ソース (デバッグ用)</h3>
          </div>
          
          <div className="space-y-2">
            {sources.map((source, index) => (
              <div key={index} className="bg-gray-50 border border-gray-200 rounded-md">
                <button
                  onClick={() => toggleSource(index)}
                  className="flex items-center justify-between w-full px-4 py-3 text-left"
                >
                  <span className="font-medium text-sm">ソース {index + 1}: {source.name}</span>
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