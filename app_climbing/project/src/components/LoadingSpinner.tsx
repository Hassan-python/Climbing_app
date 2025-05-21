import { useTranslation } from 'react-i18next';

const LoadingSpinner = () => {
  const { t } = useTranslation();

  return (
    <div className="flex items-center justify-center p-8">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
      <span className="ml-3 text-gray-600">{t('result.loading')}</span>
    </div>
  );
};

export default LoadingSpinner;