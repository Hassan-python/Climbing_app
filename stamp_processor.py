import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any
from stamp_extractor import StampExtractor
from stamp_analyzer import StampAnalyzer
from stamp_name_extractor import StampNameExtractor

class StampProcessor:
    """押印処理を一連の流れで実行するクラス"""
    
    def __init__(self, min_area: int = 5000, blur_kernel: Tuple[int, int] = (15, 15)):
        """
        初期化
        
        Parameters:
        -----------
        min_area : int
            抽出する押印の最小面積
        blur_kernel : tuple
            ぼかし処理のカーネルサイズ
        """
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.extractor = StampExtractor(min_area=min_area, blur_kernel=blur_kernel)
        self.analyzer = StampAnalyzer()
        self.name_extractor = StampNameExtractor()
        
        # 一時ディレクトリの作成
        self.temp_dir = "temp_stamps"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def process_image(self, image_path: str) -> List[Dict[str, str]]:
        """
        画像から押印を抽出し、分析して名前を取得する
        
        Parameters:
        -----------
        image_path : str
            処理する画像のパス
            
        Returns:
        --------
        list
            各押印の情報（名前、日付、押印場所）を含む辞書のリスト
        """
        try:
            # 1. 画像の読み込み
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
            
            # 2. 押印の抽出
            stamps, mask, coords = self.extractor.extract_stamps(img)
            
            # 3. 抽出した押印の保存
            stamp_paths = []
            for i, stamp in enumerate(stamps):
                output_path = os.path.join(self.temp_dir, f"detected_stamp_{i+1}.png")
                cv2.imwrite(output_path, stamp)
                stamp_paths.append(output_path)
            
            # 4. 押印の分析
            analysis_results = self.analyzer.analyze_stamps(stamp_paths)
            
            # 5. 名前の抽出
            name_results = self.name_extractor.extract_names(stamp_paths)
            
            # 6. 一時ファイルの削除
            for path in stamp_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            return name_results
            
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            # エラー発生時も一時ファイルを削除
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
            raise

if __name__ == "__main__":
    # テスト用の画像を読み込む
    test_image_path = os.path.join("test_input", "sample_img.jpg")
    if not os.path.exists(test_image_path):
        print(f"エラー: テスト画像が見つかりません: {test_image_path}")
        exit(1)
    
    # 処理を実行
    processor = StampProcessor()
    try:
        results = processor.process_image(test_image_path)
        print("\n===== 処理結果 =====")
        for i, result in enumerate(results, 1):
            print(f"\n押印 {i}:")
            print(f"  - 名前: {result['名前']}")
            print(f"  - 日付: {result['日付']}")
            print(f"  - 押印場所: {result['押印場所']}")
    except Exception as e:
        print(f"エラー: {str(e)}")
        exit(1) 