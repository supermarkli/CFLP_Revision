# 变更日志

## 2026-04-01

### 收敛图表适配黑白印刷
- **修改** `src/utils/draw.py`：
  - 配色改为全黑，用三种线型区分（实线+圆形、虚线+方形、点划线+三角）
  - Y轴标签 Accuracy → 准确率 (%)
  - Y轴刻度改为百分比格式（0%~100%）
  - 标记样式改为空心（markerfacecolor='none'）

## 2026-03-13

### 文档整合
- **删除** `REPRODUCTION_CONFIG.md`，关键内容已迁移至 README.md
- **更新** `README.md`：
  - 新增详细环境依赖版本（Python 3.8+, PyTorch 2.0+）
  - 新增 CIFAR-10 数据归一化参数
  - 新增数据增强详细配置
  - 新增配置参数：momentum=0.9, weight_decay=0.0005
  - 新增"预期效果"章节（准确率 ~95%+, AUC ~0.99+）
  - 新增"常见问题"章节（FAQ）
  - 添加"必须用 SGD 非 Adam"的重要提示

### 可视化调整
- **更新** `src/utils/draw.py`：配色改为复古学术风格，子图标题字体改为 Times New Roman
- **更新** `out/convergence_combined.png`：配合配色调整重新生成
