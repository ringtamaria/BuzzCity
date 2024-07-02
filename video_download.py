
import os
import cv2
import subprocess

# TikTokのURL
url = 'https://vt.tiktok.com/ZSRmjy3pM/'

# yt-dlpで動画をダウンロード
subprocess.run(['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', url])

# ダウンロードした動画のファイル名を取得
video_filename = subprocess.check_output(['yt-dlp', '--get-filename', url]).decode().strip()

# 動画を読み込む
cap = cv2.VideoCapture(video_filename)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームを表示
    cv2.imshow('TikTok Video', frame)

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
