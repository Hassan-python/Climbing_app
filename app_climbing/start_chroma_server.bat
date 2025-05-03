@echo off
echo ChromaDBサーバーを起動しています...
echo 保存先: C:\Users\Hassan\Desktop\cursor\chroma_data
echo ポート: 8000

cd %~dp0
python -c "import subprocess; subprocess.run(['chroma', 'run', '--path', 'C:\\Users\\Hassan\\Desktop\\cursor\\chroma_data', '--port', '8000'])"

pause 