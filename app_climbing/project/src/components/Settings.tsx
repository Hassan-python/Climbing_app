import { useState } from 'react';
import { Settings as SettingsIcon, Key } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface SettingsProps {
  debugMode: boolean;
  setDebugMode: (value: boolean) => void;
}

export const Settings = ({ 
  debugMode, 
  setDebugMode, 
}: SettingsProps) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [geminiApiKey, setGeminiApiKey] = useState(localStorage.getItem('geminiApiKey') || '');
  
  const handleGeminiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value.trim();
    setGeminiApiKey(newValue);
    
    if (newValue) {
      localStorage.setItem('geminiApiKey', newValue);
    } else {
      localStorage.removeItem('geminiApiKey');
    }
  };
  
  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-full hover:bg-gray-100"
      >
        <SettingsIcon className="h-5 w-5 text-gray-500" />
      </button>
      
      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-white rounded-md shadow-lg z-20">
          <div className="p-4">
            <h3 className="font-medium text-gray-900 mb-3">{t('settings.title')}</h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700">{t('settings.debugMode')}</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    className="sr-only peer"
                    checked={debugMode}
                    onChange={() => setDebugMode(!debugMode)}
                  />
                  <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-emerald-300 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-600"></div>
                </label>
              </div>

              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-700 flex items-center gap-2">
                  <Key className="h-4 w-4" />
                  {t('settings.apiKeys.title')}
                </h4>

                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    {t('settings.apiKeys.gemini')}
                  </label>
                  <input
                    type="password"
                    value={geminiApiKey}
                    onChange={handleGeminiKeyChange}
                    className="w-full text-sm rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500"
                    placeholder="Gemini API Key"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {isOpen && (
        <div 
          className="fixed inset-0 z-10" 
          onClick={() => setIsOpen(false)}
        ></div>
      )}
    </div>
  );
};