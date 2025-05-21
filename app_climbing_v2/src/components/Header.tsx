import { Mountain } from 'lucide-react';
import LanguageSwitcher from './LanguageSwitcher';

const Header = () => {
  return (
    <header className="bg-gradient-to-r from-emerald-800 to-teal-700 text-white shadow-md">
      <div className="container mx-auto px-4 py-5">
        <div className="flex items-center justify-between">
          <div className="flex flex-col">
            <div className="flex items-center gap-3">
              <Mountain size={28} className="text-white" />
              <h1 className="text-2xl font-bold">AI Climbing Tokyo</h1>
            </div>
            <p className="text-sm text-gray-200 mt-1 ml-10">
              Powered by Japanese climbers. Upload your fall — get Gen-AI feedback inspired by Japan’s top pros.
            </p>
          </div>
          <LanguageSwitcher />
        </div>
      </div>
    </header>
  );
};

export default Header;