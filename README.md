# heartsound2022

cài đặt python3.10

cài môi trường ảo
 $ python -m venv venv
 $ source venv/bin/activate (macOS, linux)
        or 
    .\venv\Scripts\activate  (Windows)

 cài thư viện
 $ pip install -r requirements.txt

deactivate

uvicorn main:app --reload --port 8001
Các bước:
1. Tải data từ link dưới đây: 'https://physionet.org/content/circor-heart-sound/1.0.3/#files-panel' kéo xuống dưới cùng
2. Giải nén các tệp vào thư mục heartsound2022
3. Chạy file "slpit_wav_each_folder.ipynb" để chia các file âm thanh ra từng folder train, test
4. Chạy file "heart-sound-v2-new.ipynb" để train model(Đã bao gồm xử lý dữ liệu và phần train).

Sau khi chạy sẽ tạo ra 2 tập dữ liệu "data_spectrogram.csv" và "data_spectrogram_K5-2.csv" nên đưa dữ liệu lên Kaggle để train, nếu máy không có GPU.
