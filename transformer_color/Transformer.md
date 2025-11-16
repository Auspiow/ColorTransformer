# 基于Transformer的感知色差预测模型

本工作旨在利用深度学习模型拟合旧有色彩心理物理实验中的“人类感知色差”（human perceptual color difference）。传统色差公式（如 ΔE76 / ΔE94 / ΔE2000）均为人工设计，而 AI 模型能够从跨数据集的实验数据中自动学习更符合人类主观视觉一致性的色差度量。

以下为完整的项目说明文档。

## 1. 数据集准备（Datasets）

为了构建统一且覆盖足够多颜色对的训练数据，我们使用 GitHub 上 Coloria 整合仓库：

🔗 [https://github.com/coloria-dev/color-data](https://github.com/coloria-dev/color-data?utm_source=chatgpt.com)

从中选择了 1980–1990 年代最经典、使用最广的心理物理实验数据集：

| 数据集          | 光源 | 说明                    |
| --------------- | ---- | ----------------------- |
| bfd-c.json      | C    | BFD（土耳其）、经典实验 |
| bfd-d65.json    | D65  |                         |
| bfd-m.json      | M    |                         |
| leeds.json      | D65  | Leeds 色差实验          |
| rit-dupont.json |      | RIT–DuPont 汽车涂料实验 |
| witt.json       |      | Witt 实验               |

最终数据目录结构：

```
datasets/
├─ bfd-c.json
├─ bfd-d65.json
├─ bfd-m.json
├─ rit-dupont.json
├─ leeds.json
└─ witt.json
```

### 1.1 数据格式（统一 JSON 结构）

每个 JSON 文件均包含：

```
{
    "reference_white": [],
    "dv": [],      // 人类感知差异评分 (difference values)
    "pairs": [],   // 索引对 (i, j)：表示颜色 xyz[i], xyz[j]
    "xyz": []      // 所有 XYZ 颜色样本
}
```

加载后转换为最终训练格式：

```
L1, a1, b1, L2, a2, b2 → DE_human
```

其中 Lab 使用 `colormath` 以 D65、2° observer 转换。

---

## 2.模型输入 / 输出设计

**输入特征（Feature Design）**

```
(L1, a1, b1, L2, a2, b2)
```

输入为两组 Lab 颜色拼接，并 reshape 为：

```
batch × 2 × 3
```

以便送入 Transformer 作为两个 token。

**输出（Target）**

```
y_pred ∈ ℝ  # 模型预测的人类视觉色差 ΔE_vis
```

此 ΔE_vis 不是任何已有色差公式，而是直接拟合实验中的“主观差异评分”。

### 2.1 重要的数据预处理

为了使训练更加稳定，使用：

- **log(1+ΔE)** 抑制长尾分布
- **标准化（z-score）** 提升训练速度
- **按 ΔE 区间采样（balanced sampling）** 让模型在小差异区域（人类敏感区域）学习更多

---

## 3. 模型结构（Transformer for Color Difference）

本项目采用一个轻量级 Transformer 编码器，用于学习两个颜色 token 的关系：

```python
import torch
import torch.nn as nn

class ColorTransformer(nn.Module):
    def __init__(self, dim=32, depth=4, heads=4):
        super().__init__()

        self.embed = nn.Linear(3, dim)  # Lab → embedding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=128,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.fc = nn.Linear(dim * 2, 1)  # flatten embeddings → scalar output

    def forward(self, x):
        # x: (batch, 2, 3)
        e = self.embed(x)
        out = self.encoder(e)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)
```

---

## 4. 基线色差公式（Benchmark）

为了评估模型是否真正“比 ΔE 更像人类”，我们使用以下基线：

| Baseline | 说明                   |
| -------- | ---------------------- |
| ΔE76     | 欧氏距离               |
| ΔE94     | 工业色差               |
| ΔE2000   | 当前最常用             |
| OKLab ΔE | perceptual color space |

### 4.1 Oklab ΔE

```python
def oklab_de(l1,a1,b1,l2,a2,b2):
    return np.sqrt((l1-l2)**2 + (a1-a2)**2 + (b1-b2)**2)
```

### 4.2 ΔE2000

```python
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def de2000(row):
    c1 = LabColor(row.L1, row.a1, row.b1)
    c2 = LabColor(row.L2, row.a2, row.b2)
    return delta_e_cie2000(c1, c2)
```

---

## 5. 评价指标（Evaluation Metrics）

使用皮尔逊相关系数 Pearson R 衡量模型输出与人类实验的相关性：

```
from scipy.stats import pearsonr

r = pearsonr(true_values, predicted_values)[0]
```

---

## 6. 可视化与实验结果（Visualization）

### 6.1 散点图：AI vs Human

```
plt.figure()
plt.scatter(true_all, preds_un, s=6, alpha=0.4)
plt.xlabel("Human score (ΔE raw)")
plt.ylabel("Model prediction (ΔE raw)")
plt.title(f"Siamese Model vs Human (R={r_model:.4f})")
plt.plot([0, max(true_all.max(), preds_un.max())], [0, max(true_all.max(), preds_un.max())], 'r--', linewidth=1)
plt.savefig("scatter_siamese_pred_vs_human.png", dpi=150)
plt.close()
```

<img src="./images/scatter_siamese_pred_vs_human.png" style="zoom: 50%;" />

### 6.2 误差直方图（error hist）

```
err = (preds_un.ravel() - true_all.ravel())
plt.figure()
sns.histplot(err, bins=80, kde=True)
plt.title("Prediction error (pred - human)")
plt.savefig("hist_error_siamese.png", dpi=150)
plt.close()
```

<img src="./images/hist_error_siamese.png" style="zoom:50%;" />

### 6.3 R值对比图（R comparison bar）

```
plt.figure()
labels = ["Siamese", "ΔE2000"]
vals = [r_model, r_de2000]
sns.barplot(x=labels, y=vals)
plt.ylim(0,1)
plt.title("Pearson R comparison")
plt.savefig("r_comparison_siamese.png", dpi=150)
plt.close()
```

<img src="./images/r_comparison_siamese.png" style="zoom:50%;" />

### 6.4 实验结果

当前模型（经过 log-scaling + balanced sampling + Huber Loss）：

```
R(model)   = 0.9253
R(DE2000)  = 0.7754
```

AI 模型显著超越 DE2000（≈ +0.15），达到接近人类间一致性（inter-observer consistency ≈ 0.90–0.95）。

说明模型确实在学习“人类视觉感知”，而不仅仅是 Lab 几何距离。

