# 交付图表复核

- 六组图均由本轮实测CSV生成，提供PNG/PDF；没有新增、插值或平滑未测条件。
- Figure1–3图例移到坐标区外，避免遮挡高层的置信区间。原版保存在analysis_versions/initial_figure_legends。
- Figure5最初的箱线图FixedFormatter继承导致稀疏横轴刻度文字错误，已显式绑定真实层号0、2、…、30。原图保存在analysis_versions/initial_figure5_ticklabels，并新增CPU回归测试。
- 修正仅涉及显示；原始生成、官方分数、配对统计表及阈值未改。前后SHA256检查见quality_tables_before_plot_refinement.sha256及quality_tables_after_plot_refinement_check.log。
- 图中所有负向极端值保留。Figure6显示真实图片、Full/干预文本、原始首个不同Token及自动分数差异；语义核查及其边界见case_visual_review.md。
