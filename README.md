# 足球目标检测 —— YOLOv5 作业

用 YOLOv5 训练的一个两类检测模型，识别画面中的 **球员（player）** 与 **足球（ball）**。

## 1. 视频来源

本地自拍的 3 段足球视频（不入库，仅本地使用）：

| 组 | 内容 | 分辨率 / 帧率 | 时长 |
| --- | --- | --- | --- |
| A | 白天，航拍全景 + 地面近景 | 1918x1080 / 30fps | 178 s |
| B | 夜场，看台垂直俯视 | 1920x1080 / 24fps | 171 s |
| C | 职业转播战术机位 | 1920x1080 / 25fps | 345 s |

抽帧方式：每 50 帧取一帧，模糊帧丢弃，相邻帧去重，统一缩放到宽 1280。共得 **319 帧**。

## 2. 类别与标注

| 类别 id | 名称 | 说明 |
| --- | --- | --- |
| 0 | player | 场上球员、门将、裁判 |
| 1 | ball | 足球（小目标，训练重点） |

标注为 YOLO 格式（`class cx cy w h`，归一化）。先用 COCO 预训练的 YOLOv5x（imgsz 1280）生成初稿，再人工复核：A 组逐帧核对（补漏检、修框、加球），B/C 组以初稿为主。

## 3. 数据划分

按**帧序号时间顺序**切分（前 80% 训练、后 20% 验证），而不是随机切分 —— 相邻帧几乎相同，随机切会让验证指标虚高。

| 集合 | 图片 | A | B | C | 标注框（player / ball） | 含球图片 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 255 | 86 | 58 | 111 | 5115 / 170 | 130 |
| val | 64 | 21 | 15 | 28 | 1229 / 34 | 28 |

合计 6548 个框，球框占比 3.1%。val 中 `B_003951`、`B_004001` 两张无目标（背景图）。

配置文件 `data/mydataset.yaml`：

```yaml
path: datasets/mydataset   # 相对仓库根目录
train: images/train
val: images/val
nc: 2
names: ['player', 'ball']
```

## 4. 训练环境与命令

- GPU：NVIDIA GeForce RTX 5060 Laptop，8 GB
- Python 3.12.14（venv），PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128（RTX 50 系需 cu128 及以上）
- 代码：ultralytics/yolov5 v7.0（commit 915bbf2）

```bash
python train.py --weights yolov5s.pt --data data/mydataset.yaml \
  --epochs 100 --imgsz 1280 --batch 8 --device 0 --workers 2 --name football-1280
python detect.py --weights best.pt --source <video> --imgsz 1280 --conf 0.25 --device 0
```

100 epochs 用时约 21 分钟，显存峰值约 7 GB。

## 5. 验证集指标（imgsz 1280）

| 类别 | P | R | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| all | 0.788 | 0.791 | **0.795** | **0.521** |
| player | 0.865 | 0.936 | 0.964 | 0.738 |
| ball | 0.712 | 0.647 | 0.625 | 0.304 |

曲线与验证集可视化见 `metrics/`。

## 6. 推理结果

`result.mp4`：15 秒、1280x720、约 24 MB（< 50 MB）。片段取自视频 A 的 161.7–176.7 秒（**未参与训练**的后段），用 `best.pt` 以 imgsz 1280 推理。渲染时按类别分别设阈值，避免非球员区域的误检：

- player：conf ≥ 0.90（红框）
- ball：conf ≥ 0.30（绿框，球是小目标，阈值适当放低）

450 帧中 163 帧检出足球（36%）；整段 842 帧以 conf 0.25 推理时 283 帧检出足球（33.6%）。

`frame_01.png` / `frame_02.png` / `frame_03.png` 为从 `result.mp4` 截取的 3 帧（均为球被检出的帧）。

## 7. 遇到的问题

1. **球识别不出来**：首版 640 训练的模型球 mAP 为 0（球样本仅 2.4%，640 下小球被缩没）。改为 imgsz 1280、加大球标注量后，球的 mAP@0.5 达到 0.625。
2. **内存不足**：1280 训练时 `--workers 4` 在第 13 轮抛 OpenCV `OutOfMemoryError`（机器仅 16 GB 内存），改 `--workers 2` 后稳定。
3. **labelImg 崩溃**：labelImg 1.8.6 与 PyQt5 5.15.11 不兼容（`drawLine` 传 float 报 TypeError），对 `libs/canvas.py`、`libs/shape.py` 4 处补 int() 后可用。
4. **已知不足**：观众席、广告牌偶有误检成 player；航拍远景的小球员存在漏检；B/C 组标注未逐帧人工复核。

## 8. 文件清单

```
result.mp4         推理视频（15s / 1280x720 / 24MB，player conf 0.9、ball conf 0.3）
frame_01~03.png    从 result.mp4 截取的 3 帧
best.pt            YOLOv5s 权重（100 epochs，imgsz 1280）
README.md          本文件
metrics/           PR 曲线、F1 曲线、混淆矩阵、验证集标签/预测可视化
```

数据集本身不上传：图片与标注留在本地，按第 2、3 节说明可重新整理出同样的 train/val。
