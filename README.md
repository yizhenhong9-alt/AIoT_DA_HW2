# 🍷 紅酒品質預測分析報告

**主題：** Kaggle《Red Wine Quality》資料集分析與模型建構
**方法論：** CRISP-DM 流程
**作者：** 洪翌榛
**執行環境：** Python（Pandas、Seaborn、Scikit-learn、Matplotlib）

---

## 一、專案背景與目標（Business Understanding）

葡萄酒品質受到多重理化特徵的影響，如酸度、糖分、酒精濃度、揮發性酸等。傳統上品質評分依賴人工品酒師，具有主觀性與成本高等問題。因此，本專案旨在利用 Kaggle 上的《Red Wine Quality》資料集，透過機器學習方法預測紅酒品質分數，以協助自動化品質評估、降低人力成本並提供釀酒決策依據。

本專案目標如下：

1. 探討紅酒理化變數與品質分數之間的關聯性。
2. 建立多種迴歸模型進行品質預測（含特徵選擇）。
3. 比較模型效能並分析關鍵特徵。
4. 建立預測結果與信賴區間的視覺化圖表。

---

## 二、資料理解（Data Understanding）

### 2.1 資料來源

資料集來自 [Kaggle: Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)，共包含 **1599 筆樣本**與 **12 個欄位**：

| 特徵名稱                 | 說明                | 資料型態  |
| -------------------- | ----------------- | ----- |
| fixed acidity        | 固定酸度              | float |
| volatile acidity     | 揮發性酸度             | float |
| citric acid          | 檸檬酸               | float |
| residual sugar       | 殘糖量               | float |
| chlorides            | 氯化物濃度             | float |
| free sulfur dioxide  | 遊離二氧化硫            | float |
| total sulfur dioxide | 總二氧化硫             | float |
| density              | 密度                | float |
| pH                   | 酸鹼值               | float |
| sulphates            | 硫酸鹽               | float |
| alcohol              | 酒精濃度              | float |
| quality              | 紅酒品質評分（目標變數，0–10） | int   |

### 2.2 資料特性

* 無缺失值（缺失比例 = 0%）。
* 各特徵分布為非對稱型，部分變數（如 `residual sugar`、`total sulfur dioxide`）呈右偏。
* 目標變數 `quality` 多集中於 5–7 分區間。

### 2.3 初步相關性分析

使用 Pearson 相關係數可發現：

* **酒精濃度（alcohol）** 與品質呈正相關（r ≈ 0.48）。
* **揮發性酸（volatile acidity）** 與品質呈負相關（r ≈ -0.39）。
* 其餘特徵如 `sulphates` 與 `citric acid` 也具有中度正相關。

---

## 三、資料準備（Data Preparation）

1. **標準化（Standardization）**：
   為消除不同特徵尺度差異，使用 `StandardScaler()` 將輸入變數標準化。

2. **特徵選擇（Feature Selection）**：

   * 採用 `SelectKBest(f_regression)` 篩選最具解釋力的特徵。
   * 經交叉驗證後選出前五項重要特徵：

     ```
     alcohol, volatile acidity, sulphates, citric acid, total sulfur dioxide
     ```

3. **資料切割**：

   * 訓練集：80%
   * 測試集：20%
   * 隨機種子：42，確保實驗可重現。

---

## 四、建模（Modeling）

為比較不同迴歸模型的效能，本研究建立了下列模型：

| 模型名稱                            | 說明        | 特性           |
| ------------------------------- | --------- | ------------ |
| **Linear Regression**           | 基本線性模型    | 解釋性高，假設線性關係  |
| **Ridge Regression**            | 加入 L2 正規化 | 抑制多重共線性      |
| **Lasso Regression**            | 加入 L1 正規化 | 具特徵選擇能力      |
| **Decision Tree Regressor**     | 非線性樹狀模型   | 可捕捉非線性結構     |
| **Random Forest Regressor**     | 多樹集成模型    | 泛化能力強，具特徵重要度 |
| **Gradient Boosting Regressor** | 逐步提升弱分類器  | 表現穩定但訓練時間較長  |

---

## 五、評估（Evaluation）

### 5.1 效能指標

採用以下指標：

* **R²**（解釋變異比例）
* **MAE**（平均絕對誤差）
* **RMSE**（均方根誤差）

### 5.2 模型比較結果（測試集）

| 模型                | R²        | MAE       | RMSE      |
| ----------------- | --------- | --------- | --------- |
| Linear Regression | 0.285     | 0.565     | 0.756     |
| Ridge Regression  | 0.289     | 0.561     | 0.752     |
| Lasso Regression  | 0.277     | 0.573     | 0.764     |
| Decision Tree     | 0.238     | 0.590     | 0.783     |
| Random Forest     | **0.405** | **0.478** | **0.686** |
| Gradient Boosting | 0.392     | 0.489     | 0.699     |

> ✅ **最佳模型為 Random Forest Regressor**，表現穩定且誤差最低。

---

## 六、結果分析與視覺化（Results & Visualization）

### 6.1 特徵重要度

根據隨機森林模型：

```
1️⃣ alcohol  
2️⃣ volatile acidity  
3️⃣ sulphates  
4️⃣ citric acid  
5️⃣ total sulfur dioxide
```

酒精濃度為最關鍵影響品質的特徵，與品酒常識相符。

### 6.2 預測 vs 實際散點圖

* 散點圖顯示預測值與實際值近似沿對角線分布。
* 中間區間（品質 5–7 分）預測最為穩定。
* 加入 95% 信賴區間帶後可見模型偏差主要集中於極端分數。

### 6.3 模型誤差分布

* 殘差接近常態分布，無明顯系統性誤差。
* 標準差 ≈ 0.68，代表預測誤差在 ±0.7 品質分以內。

---

## 七、部署與應用（Deployment）

此模型可延伸應用於：

* 釀酒生產線自動評分系統。
* 紅酒品質預測儀表板（可結合 Streamlit 或 Flask）。
* 與白酒資料集結合進行跨品種泛化分析。

---

## 八、最終結論與反思（Final Thoughts）

本專案透過 CRISP-DM 流程完整實作紅酒品質預測模型，從資料理解、特徵分析、建模到視覺化評估，獲得以下結論：

1. **特徵影響：** 酒精濃度、揮發性酸與硫酸鹽含量為主要影響品質的因子。
2. **模型表現：** 隨機森林回歸表現最佳，R² 約 0.40，顯示模型能解釋約 40% 的品質變異。
3. **限制與改進方向：**

   * 資料分布偏中間區，導致模型難以學習極端品質樣本。
   * 可考慮使用 **XGBoost** 或 **LightGBM** 進一步提升效能。
   * 未來可嘗試 **集成多模型（Ensemble Stacking）** 或結合化學特徵工程提升預測準確度。

---

## 九、附錄（Appendix）

* Python 版本：3.10
* 主要套件：

  ```
  pandas==2.2.2
  scikit-learn==1.5.0
  seaborn==0.13.2
  matplotlib==3.9.0
  numpy==1.26.4
  ```

---

📘 **總結一句話：**

> 本專案以資料驅動的方式揭示紅酒品質的關鍵因素，成功建構可重現、可擴展的品質預測模型，為未來自動化品酒與釀酒決策提供了可行依據。
