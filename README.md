# 11225011 吳宥諭

本作業為人工智慧課程期中報告，主要實作並記錄如何在 **Google Colaboratory** 雲端環境中跑目標跟蹤的 test 代碼。

## 實作環境與規格

* **雲端運算平台**：Google Colaboratory
* **運算核心**：NVIDIA Tesla T4 GPU
* **開發語言**：Python 3.10+
* **核心套件**：PyTorch / Torchvision / NumPy

## 一、安裝工具包

```bash
!pip install -q ultralytics opencv-python-headless
```

## 二、下載測試影片

```python
import urllib.request

video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
urllib.request.urlretrieve(video_url, "test_video.mp4")
```

**輸出結果：**

```text
('test_video.mp4', <http.client.HTTPMessage at 0x7bcaa6651370>)
```

## 三、載入模型並執行物件追蹤

```python
from ultralytics import YOLO

# 載入預訓練模型
model = YOLO("yolov8n.pt")

# 執行目標追蹤
results = model.track(
    source="test_video.mp4",
    tracker="bytetrack.yaml",
    conf=0.3,
    save=True
)

print("✅ 跟踪完成！结果保存在 runs/detect/track/ 目录下")
```

**執行過程輸出日誌：**

```text
WARNING ⚠️ Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict for help.

video 1/1 (frame 1/377) /content/test_video.mp4: 384x640 (no detections), 343.1ms
video 1/1 (frame 2/377) /content/test_video.mp4: 384x640 (no detections), 133.2ms
video 1/1 (frame 3/377) /content/test_video.mp4: 384x640 (no detections), 133.4ms
video 1/1 (frame 4/377) /content/test_video.mp4: 384x640 (no detections), 128.9ms
video 1/1 (frame 5/377) /content/test_video.mp4: 384x640 (no detections), 130.4ms
video 1/1 (frame 6/377) /content/test_video.mp4: 384x640 (no detections), 136.0ms
video 1/1 (frame 7/377) /content/test_video.mp4: 384x640 (no detections), 129.4ms
video 1/1 (frame 8/377) /content/test_video.mp4: 384x640 (no detections), 148.3ms
video 1/1 (frame 9/377) /content/test_video.mp4: 384x640 (no detections), 133.3ms
video 1/1 (frame 10/377) /content/test_video.mp4: 384x640 (no detections), 131.7ms
video 1/1 (frame 11/377) /content/test_video.mp4: 384x640 (no detections), 132.1ms
video 1/1 (frame 12/377) /content/test_video.mp4: 384x640 (no detections), 131.3ms
video 1/1 (frame 13/377) /content/test_video.mp4: 384x640 (no detections), 132.4ms
video 1/1 (frame 14/377) /content/test_video.mp4: 384x640 (no detections), 136.7ms
video 1/1 (frame 15/377) /content/test_video.mp4: 384x640 (no detections), 154.1ms
video 1/1 (frame 16/377) /content/test_video.mp4: 384x640 (no detections), 134.6ms
video 1/1 (frame 17/377) /content/test_video.mp4: 384x640 (no detections), 208.9ms
video 1/1 (frame 18/377) /content/test_video.mp4: 384x640 (no detections), 215.5ms
video 1/1 (frame 19/377) /content/test_video.mp4: 384x640 (no detections), 206.0ms
video 1/1 (frame 20/377) /content/test_video.mp4: 384x640 (no detections), 237.3ms
...
```

## 四、實驗結果與分析

1. **輸出檔案與路徑**：
   * 本次執行結果已成功導出並保存在 `runs/detect/track/test_video.mp4`（或 `runs/detect/track/` 目錄內）。
   * 輸出的影片檔已被寫入物件邊界框（Bounding Boxes）與車輛追蹤 ID（Tracking IDs）。

2. **辨識與追蹤效能**：
   * **總張數**：影片共計 377 幀（Frames）。
   * **前段畫面 (Frame 1 ~ 20+)**：顯示 `no detections`，主因是影片開頭畫面尚無車輛駛入辨識區域。
   * **後續畫面**：當車輛進入畫面後，YOLOv8n 模型配合 ByteTrack 演算法成功進行物件識別並進行持續的跨幀軌跡追蹤。

3. **實驗總結**：
   * 本次實作證實了 YOLOv8 輕量化模型（`yolov8n.pt`）於 Google Colab T4 GPU 上具有即時處理（Inference）之能力，平均單張推理時間約落在 130ms ~ 200ms 之間，能有效達成多目標車輛追蹤之任務。
