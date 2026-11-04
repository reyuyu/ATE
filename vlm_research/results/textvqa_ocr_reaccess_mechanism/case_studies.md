# OCR与视觉回看：冻结规则选择的案例

每类按Question ID升序取第一个满足预定条件的案例；分类已先冻结。软分数不是二值正确率。完整损害记录见negative_cases.csv。

## OCR_Direct_A_full_B_zero / 34650

what word does the license plate say?

类别：OCR-Direct。依据：Explicitly asks to read the license-plate word; word visible on plate.

![原图缩略图](case_images/34650.jpg)

原图：`/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/images/textvqa/ef00f12063be7b07.jpg`；SHA256 `955382abbf4c2ab87473a00d4c4aae495c2cd64241bcc215523f16bdc67e5a7a`。

官方10个参考答案：`["french", "french", "french", "french", "french", "french", "french", "french", "french", "french"]`

| 条件 | 实际回答 | Soft Score | 相对B首处分叉（1-based） |
|---|---|---:|---:|
| A | French | 1.0 | 1 |
| B | California | 0.0 | 相同 |
| F | French | 1.0 | 1 |
| G | California | 0.0 | 相同 |
| E | California | 0.0 | 相同 |
| Both-4 | California | 0.0 | 相同 |
| Both-8 | California | 0.0 | 相同 |
| Both-12 | California | 0.0 | 相同 |
| Both-16 | California | 0.0 | 相同 |
| Both-24 | California | 0.0 | 相同 |

## OCR_Direct_B_full / 34970

what number of the cashier on the far right?

类别：OCR-Direct。依据：Read the checkout number above the far-right cashier.

![原图缩略图](case_images/34970.jpg)

原图：`/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/images/textvqa/d8800354e8ab991e.jpg`；SHA256 `5337f78eab723dbb1765b466f8bd0ade9cefc407baf94cf277c496e04e9da253`。

官方10个参考答案：`["1", "1", "2", "11  march 2011", "1", "1", "1", "1", "1", "1"]`

| 条件 | 实际回答 | Soft Score | 相对B首处分叉（1-based） |
|---|---|---:|---:|
| A | 12 | 0.0 | 3 |
| B | 1 | 1.0 | 相同 |
| F | 1 | 1.0 | 相同 |
| G | 12 | 0.0 | 3 |
| E | 1 | 1.0 | 相同 |
| Both-4 | 1 | 1.0 | 相同 |
| Both-8 | 2 | 0.3 | 2 |
| Both-12 | 2 | 0.3 | 2 |
| Both-16 | 2 | 0.3 | 2 |
| Both-24 | 12 | 0.0 | 3 |

## OCR_Direct_F_recovers_G_not / 34650

what word does the license plate say?

类别：OCR-Direct。依据：Explicitly asks to read the license-plate word; word visible on plate.

![原图缩略图](case_images/34650.jpg)

原图：`/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/images/textvqa/ef00f12063be7b07.jpg`；SHA256 `955382abbf4c2ab87473a00d4c4aae495c2cd64241bcc215523f16bdc67e5a7a`。

官方10个参考答案：`["french", "french", "french", "french", "french", "french", "french", "french", "french", "french"]`

| 条件 | 实际回答 | Soft Score | 相对B首处分叉（1-based） |
|---|---|---:|---:|
| A | French | 1.0 | 1 |
| B | California | 0.0 | 相同 |
| F | French | 1.0 | 1 |
| G | California | 0.0 | 相同 |
| E | California | 0.0 | 相同 |
| Both-4 | California | 0.0 | 相同 |
| Both-8 | California | 0.0 | 相同 |
| Both-12 | California | 0.0 | 相同 |
| Both-16 | California | 0.0 | 相同 |
| Both-24 | California | 0.0 | 相同 |

## OCR_Direct_G_recovers_F_not / 34832

what is the team name on this jersey?

类别：OCR-Direct。依据：Team name is printed across the jersey front.

![原图缩略图](case_images/34832.jpg)

原图：`/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/images/textvqa/443c0b31a360e242.jpg`；SHA256 `30152a651a4adf91e58973e164f295d97dc9ab427a65bbdcb18abc8ed3135233`。

官方10个参考答案：`["seals", "demarco", "seals", "seals", "seals", "seals", "seals", "danizd krone", "demarco", "seals"]`

| 条件 | 实际回答 | Soft Score | 相对B首处分叉（1-based） |
|---|---|---:|---:|
| A | Seals | 1.0 | 2 |
| B | Seawolves | 0.0 | 相同 |
| F | Seawolves | 0.0 | 相同 |
| G | Seals | 1.0 | 2 |
| E | Seahawks | 0.0 | 2 |
| Both-4 | Seahawks | 0.0 | 2 |
| Both-8 | Seahawks | 0.0 | 2 |
| Both-12 | Seahawks | 0.0 | 2 |
| Both-16 | Seahawks | 0.0 | 2 |
| Both-24 | Seals | 1.0 | 2 |

## NonOCR_reaccess_gain / 37621

what color is the rooster?

类别：Non-OCR Visual。依据：Rooster is a visible black silhouette; color does not require reading the adjacent brand name.

![原图缩略图](case_images/37621.jpg)

原图：`/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/images/textvqa/116a6eae4f120bac.jpg`；SHA256 `72bced2e7cd10f4d955a2cf2bc2504ec55cd609c0419dc8c354f1287054980e6`。

官方10个参考答案：`["black", "black", "black", "black", "black", "black", "black", "black", "red", "black"]`

| 条件 | 实际回答 | Soft Score | 相对B首处分叉（1-based） |
|---|---|---:|---:|
| A | Red | 0.3 | 1 |
| B | Brown | 0.0 | 相同 |
| F | Red | 0.3 | 1 |
| G | Brown | 0.0 | 相同 |
| E | Brown | 0.0 | 相同 |
| Both-4 | Brown | 0.0 | 相同 |
| Both-8 | Brown | 0.0 | 相同 |
| Both-12 | Black | 1.0 | 1 |
| Both-16 | Red | 0.3 | 1 |
| Both-24 | Red | 0.3 | 1 |

## Higher_budget_harms / 34809

what does that guy's shirt say?

类别：OCR-Direct。依据：Explicit shirt-text transcription.

![原图缩略图](case_images/34809.jpg)

原图：`/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/images/textvqa/a59bb7f78afc83a5.jpg`；SHA256 `784af818ca2899ef6d38835fdba9dcad0478f541141f8b59cc6698f2feb73602`。

官方10个参考答案：`["same same ", "same same", "same same", "same same", "same", "same same", "same same", "same", "same same", "same same"]`

| 条件 | 实际回答 | Soft Score | 相对B首处分叉（1-based） |
|---|---|---:|---:|
| A | Same | 0.6 | 1 |
| B | Nothing | 0.0 | 相同 |
| F | Same | 0.6 | 1 |
| G | Nothing | 0.0 | 相同 |
| E | Same | 0.6 | 1 |
| Both-4 | Same | 0.6 | 1 |
| Both-8 | Says | 0.0 | 1 |
| Both-12 | Games | 0.0 | 1 |
| Both-16 | Same | 0.6 | 1 |
| Both-24 | Same | 0.6 | 1 |

## OCR_Reasoning_Both24_below_A

无符合冻结条件的案例。
