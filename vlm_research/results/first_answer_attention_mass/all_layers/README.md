# Full L0-L31 Case Gallery

12 frozen candidates x all32 layers =384 native maps; no additional GPU inference. Original probabilities are unchanged.

**Shared-log overlay / pure heatmap:** visual-conditional probability on one log scale across all32 layers of a case. Raw Image mass is labelled M.

**Local contrast:** each layer uses its own 2nd-98th percentile display limits. This is for inspecting spatial patterns ONLY; colors are not probability and their intensity cannot be compared across layers.

[Enhanced four-case PNG](figure_y_enhanced.png) / [PDF](figure_y_enhanced.pdf). Original linear/raw figures remain unchanged in the parent directory.

Each individual L00.png-L31.png shows original image, shared-log overlay and local-contrast pure heatmap. Each all32_views.pdf has three pages: shared-log overlay, local-contrast overlay, and pure shared-log heatmaps.

| Task | QID | All32 shared log | Local contrast | Pure heatmap | Three-page PDF |
|---|---|---|---|---|---|
| ChooseAttr | 09152646 | [PNG](09152646/all32_shared_log_overlay.png) | [PNG](09152646/all32_local_contrast.png) | [PNG](09152646/all32_pure_heatmap.png) | [PDF](09152646/all32_views.pdf) |
| ChooseAttr | 13363937 | [PNG](13363937/all32_shared_log_overlay.png) | [PNG](13363937/all32_local_contrast.png) | [PNG](13363937/all32_pure_heatmap.png) | [PDF](13363937/all32_views.pdf) |
| ChooseCat | 03268289 | [PNG](03268289/all32_shared_log_overlay.png) | [PNG](03268289/all32_local_contrast.png) | [PNG](03268289/all32_pure_heatmap.png) | [PDF](03268289/all32_views.pdf) |
| ChooseCat | 00902652 | [PNG](00902652/all32_shared_log_overlay.png) | [PNG](00902652/all32_local_contrast.png) | [PNG](00902652/all32_pure_heatmap.png) | [PDF](00902652/all32_views.pdf) |
| ChooseRel | 061046745 | [PNG](061046745/all32_shared_log_overlay.png) | [PNG](061046745/all32_local_contrast.png) | [PNG](061046745/all32_pure_heatmap.png) | [PDF](061046745/all32_views.pdf) |
| ChooseRel | 06926418 | [PNG](06926418/all32_shared_log_overlay.png) | [PNG](06926418/all32_local_contrast.png) | [PNG](06926418/all32_pure_heatmap.png) | [PDF](06926418/all32_views.pdf) |
| CompareAttr | 06952284 | [PNG](06952284/all32_shared_log_overlay.png) | [PNG](06952284/all32_local_contrast.png) | [PNG](06952284/all32_pure_heatmap.png) | [PDF](06952284/all32_views.pdf) |
| CompareAttr | 1925371 | [PNG](1925371/all32_shared_log_overlay.png) | [PNG](1925371/all32_local_contrast.png) | [PNG](1925371/all32_pure_heatmap.png) | [PDF](1925371/all32_views.pdf) |
| LogicalObj | 07547208 | [PNG](07547208/all32_shared_log_overlay.png) | [PNG](07547208/all32_local_contrast.png) | [PNG](07547208/all32_pure_heatmap.png) | [PDF](07547208/all32_views.pdf) |
| LogicalObj | 13297809 | [PNG](13297809/all32_shared_log_overlay.png) | [PNG](13297809/all32_local_contrast.png) | [PNG](13297809/all32_pure_heatmap.png) | [PDF](13297809/all32_views.pdf) |
| QueryAttr | 08822814 | [PNG](08822814/all32_shared_log_overlay.png) | [PNG](08822814/all32_local_contrast.png) | [PNG](08822814/all32_pure_heatmap.png) | [PDF](08822814/all32_views.pdf) |
| QueryAttr | 04819723 | [PNG](04819723/all32_shared_log_overlay.png) | [PNG](04819723/all32_local_contrast.png) | [PNG](04819723/all32_pure_heatmap.png) | [PDF](04819723/all32_views.pdf) |

[Full32-layer raw and conditional patch data](patch_attention_all32.csv). The display_manifest.json records exact display limits and all original checkpoint hashes.
