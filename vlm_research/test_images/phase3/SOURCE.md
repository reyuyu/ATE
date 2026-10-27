# Phase 3 小样本图片

- `street_mirrored.png`：由第一阶段 `../example.jpg` 使用 Pillow `FLIP_LEFT_RIGHT` 生成的无损 PNG；原图未改动。
- 原始来源：https://www.ilankelman.org/stopsigns/australia.jpg 。
- 用于同问题、左右答案相反的配对空间样本；是镜像派生图，不是独立自然图，文字也会镜像。
- 新增 QA 由用户授权 AI 目视复核，`human_verified=false`；不是人工核验数据集。
