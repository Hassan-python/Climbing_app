import { useState, lazy, Suspense } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Header from './components/Header';
import VideoUploader from './components/VideoUploader';
import AnalysisForm from './components/AnalysisForm';
import LoadingSpinner from './components/LoadingSpinner';
import DebugPanel from './components/DebugPanel';
import { Settings } from './components/Settings';
import { VideoInfo, AnalysisSettings, AnalysisResponse, AnalysisProgress } from './types';
import { uploadAndAnalyze } from './api/analysisService';
import { useTranslation } from 'react-i18next';

const AnalysisResult = lazy(() => import('./components/AnalysisResult'));

function App() {
  const { t } = useTranslation();
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [currentAdvice, setCurrentAdvice] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>({ stage: 'upload', progress: 0 });

  const handleVideoUpload = (info: VideoInfo) => {
    setVideoInfo(info);
    setAnalysisResult(null);
    setCurrentAdvice('');
  };

  const handleStartAnalysis = async (settings: AnalysisSettings) => {
    if (!videoInfo) {
      toast.error(t('errors.uploadVideo'));
      return;
    }
    
    setIsAnalyzing(true);
    setCurrentAdvice('');
    try {
      console.log('Starting analysis with settings:', settings);
      const result = await uploadAndAnalyze(
        videoInfo,
        settings,
        (progress: AnalysisProgress) => {
          console.log('Analysis progress:', progress);
          setAnalysisProgress(progress);
        },
        (chunk: string) => {
          setCurrentAdvice(prev => prev + chunk);
        }
      );
      setAnalysisResult(result);
      toast.success(t('result.complete'));
    } catch (error) {
      console.error('Analysis error:', error);
      const errorMessage = error instanceof Error 
        ? error.message
        : t('errors.unexpectedError');
      
      toast.error(errorMessage, {
        autoClose: 10000,
        position: "top-center",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="lg:w-1/2">
            <h2 className="text-xl font-semibold mb-4">{t('uploader.title')}</h2>
            <VideoUploader onVideoUploaded={handleVideoUpload} />
            
            {videoInfo && (
              <div className="mt-6">
                <div className="aspect-video bg-black rounded-lg overflow-hidden">
                  <video 
                    src={videoInfo.url} 
                    controls 
                    className="w-full h-full"
                  ></video>
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  {t('analysis.duration', { duration: videoInfo.duration.toFixed(2) })}
                </p>
              </div>
            )}
          </div>
          
          <div className="lg:w-1/2">
            <div className="sticky top-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">{t('analysis.title')}</h2>
                <Settings 
                  debugMode={debugMode} 
                  setDebugMode={setDebugMode}
                />
              </div>
              
              <AnalysisForm 
                videoInfo={videoInfo}
                onSubmit={handleStartAnalysis}
                isAnalyzing={isAnalyzing}
                progress={analysisProgress}
              />
            </div>
          </div>
        </div>
        
        {(currentAdvice || analysisResult) && (
          <div className="mt-8">
            <hr className="my-6" />
            <Suspense fallback={<LoadingSpinner />}>
              <AnalysisResult advice={currentAdvice || analysisResult?.advice || ''} />
            </Suspense>
            
            {debugMode && (
              <DebugPanel 
                geminiAnalysis={analysisResult?.geminiAnalysis}
                sources={analysisResult?.sources || []}
                retrievedKnowledge={analysisResult?.retrievedKnowledge}
              />
            )}
          </div>
        )}
      </main>
      
      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
    </div>
  );
}

export default App;