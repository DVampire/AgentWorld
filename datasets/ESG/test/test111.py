import cv2
from mutagen.mp4 import MP4

try:
    audio = MP4("output.mp4")
    print("音频标签：", audio.tags)
except Exception as e:
    print("无法读取音轨信息：", e)

cap = cv2.VideoCapture("output.mp4")
if not cap.isOpened():
    print("视频无法打开，文件可能损坏")
else:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，可能损坏或没有视频轨")
    else:
        print("视频轨正常")
cap.release()
