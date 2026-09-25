# 實驗總覽

這個資料夾是「隨時回來看」的入口：每一個實驗做了什麼、看到什麼、因此選了什麼。
數字的唯一來源仍然是 [docs/benchmarks.md](../benchmarks.md)（只追加、最新的在最後）；
這裡只是索引，加上兩份整理過的讀物。

| 檔案 | 內容 |
|---|---|
| [figures.html](figures.html) | 十四張論文式圖表，分五章，每張圖下面三行：實驗、看到、選擇。用瀏覽器打開就好。 |
| [planner-drift-report.md](planner-drift-report.md) | 規劃器跨天漂移的完整調查報告（八次讀數、四個實驗、排除清單、避免的規矩）。 |
| [../deck/](../deck/) | 專案歷程簡報（pptx、html 自測模式、講義）。 |
| [../architecture.md](../architecture.md) | 層次、門檻、每個門檻從哪裡來。 |

線上版（需要登入，只有作者能開）：
[實驗圖集](https://claude.ai/artifact/M8Yg48EEf7UV4c5k4pRDLR) ·
[漂移調查報告](https://claude.ai/code/artifact/bc3e5a58-2c2d-4e7c-bd1e-289e9b86266c) ·
[五關講義](https://claude.ai/artifact/Q449z635stZkpphaj5Vekf)

## 每個實驗一列

「條目」欄是 benchmarks.md 裡的標題與日期，用搜尋找。「結果檔」在 `models/` 或 `eval/results/`，gitignore，跑腳本會重建。

### 第一章 · 主線：範本相減 + ResNet-18，DeepPCB

| 圖 | 問題 | 腳本 | 條目 | 結果檔 | 選擇 |
|---|---|---|---|---|---|
| — | 相減能不能找到缺陷（S0） | `scripts/gate_check.py` | S0 gate（08-22 起） | `eval/results/gate_check*.json` | 能，在二值化掃描上 |
| 1 | 省多少人工、漏多少 | `scripts/report.py`、`scripts/threshold_cv.py` | Operating points · 2026-08-31 | `models/test_predictions.npz` | 報操作點曲線；門檻五折上界選 → 0.912，55.6% @ 0.66% |
| 2 | 重跑一次會落在哪 | `scripts/seed_variance.py` | Seed variance · 2026-08-31 | `models/seed_variance.json` | 每個數字註明種子，報中位數與範圍 |
| 3 | 漏的是哪一類 | `scripts/class_escape_report.py` | Per-class escape · 2026-09-08 | — | 不做類別否決；電測補 open |
| 4 | 盛行率換了會怎樣 | `scripts/prevalence_report.py` | Prevalence · 2026-08-26 | — | 省人工的數字附盛行率；影子模式 |
| 5 | 對位能修什麼 | `src/aoi_agent/aoi/registration.py` | Registration · 2026-08-24 | — | 只修平移，三種情況拒絕 |
| 6 | INT8 划不划算 | `scripts/quantisation_report.py` | Quantisation · 2026-08-26 | `models/onnx/` | 不部署；省的是記憶體 |
| — | 一個候選要多久 | `scripts/reverifier_latency.py` | Re-verifier latency | — | CPU 2.5 ms，不需要 GPU |
| — | 沒有網路的地板 | `scripts/model_free_baseline.py`、`scripts/feature_baseline.py` | model-free floor、feature baseline · 09-01 | — | 手工特徵 16.7%，ResNet 值得 |

### 第二章 · 換一種影像

| 圖 | 問題 | 腳本 | 條目 | 結果檔 | 選擇 |
|---|---|---|---|---|---|
| 7 | 相減在照片上行不行 | `scripts/gate_check.py --dataset hripcb`、`scripts/transfer_report.py` | S0 gate on HRIPCB、Transfer · 2026-08-26 | `eval/results/gate_check_hripcb_*.json` | 差分只適用二值化影像；換偵測器前端 |
| 8 | 偵測器分得出真假嗎 | `scripts/train_detector.py`、`scripts/detector_report.py` | Detector front end · 08-26、imgsz 1280 · 08-31 | `models/detector_pcbaoi.pt` | 找得到、分不出；640 留預設 |
| 8 | 框上再加重驗 | `scripts/build_detector_patches.py`、`scripts/crop_reverifier_report.py` | Crop re-verifier · 08-28、三種子 · 09-16 | `data/patches_pcbaoi/`、`models/pcbaoi_reverifier*` | 0.3 / 1.2 / 2.8，沒有東西上線 |
| 8 | 訓練框看過答案嗎 | `scripts/detector_fold_report.py` | fold · 09-20 與裁決 | `models/detector_fold.json` | 判定空洞；需要「至少撿掉一點」的地板 |

### 第三章 · 更強的預訓練特徵

| 圖 | 問題 | 腳本 | 條目 | 結果檔 | 選擇 |
|---|---|---|---|---|---|
| 9 | DINOv2 + 線性頭 | `scripts/dinov2_probe.py`、`scripts/dinov2_latency.py` | DINOv2 probe · 09-15 | `models/dinov2_probe.json` | 更差（42%），41 ms vs 2.5 ms |
| 8 | 同樣在照片裁切上 | `scripts/dinov2_crops_report.py` | DINOv2 crops · 09-16 | `models/dinov2_crops.json` | 分不出 |
| 9 | 深層提示微調 | `scripts/dinov2_vpt.py` | VPT · 09-17，重跑 · 09-18 | `models/dinov2_vpt.json` | 追平 ResNet；留 ResNet |
| 8 | 提示微調在照片裁切上 | `scripts/dinov2_vpt_crops.py` | VPT crops · 09-18 | `models/dinov2_vpt_crops.json` | 分不出；這條線結束 |

### 第四章 · 語言層

| 圖 | 問題 | 腳本 | 條目 | 結果檔 | 選擇 |
|---|---|---|---|---|---|
| — | LLM 可以做決定嗎 | `scripts/agent_eval.py` | Agent layer · 08-23 | — | 不行（12 次推翻、1 次對）；它只解釋 |
| 10 | 理由是真的嗎 | `scripts/rationale_eval.py` | rationale · 09-01、09-02、09-22 兩臂 | `models/rationale_eval*.json` | 語言規則列出類別名；詞彙句 9 → 0 |
| 11 | 解釋來不來得及 | `scripts/latency_report.py` | Agent-layer latency · 08-29、08-30、09-17 | — | 提示寫進門檻；60 s 與 10 s 分開 |
| — | 規劃器查對東西嗎 | `scripts/analysis_eval.py` | Analysis planner（多次）、SQL tool 兩臂 · 08-29、listed-set · 08-30 | `models/analysis_eval*.json` | 70 題外部出題；SQL 工具留、加守衛 |
| — | 合成的文字對得上結果嗎 | `scripts/synthesis_eval.py` | synthesis · 08-31 | — | 兩種語言都量才發表 |
| — | 非生成的決策模型（Jev）能不能判「該不該拒答」 | `scripts/jev_refusal_eval.py` | Jev refusal · 09-25 | `models/jev_refusal_*.json` | 不接：F1 0.62 < gpt-oss 0.67–0.71（事先定的 0.5）；判定線整條掃過最好 0.67；快 14 倍但答得少（圖 14） |

### 第五章 · 同一個模型隔天答不一樣

| 圖 | 問題 | 腳本 | 條目 | 結果檔 | 選擇 |
|---|---|---|---|---|---|
| 12 | 是退步還是雜訊 | `scripts/analysis_eval.py --plan-only` | 第三次讀數 · 09-19、五次讀數 · 09-22 | `models/planner_drift_2026-09-22_run*.log` | ±3 基線作廢 |
| 13 | 同一時段內什麼會動 | `scripts/planner_session_probe.py` | The runner as a controlled variable · 09-23 | `models/planner_session_*.json` | batch 是原因；mmap、順序、卸載不是 |
| — | 09-02 那句提示詞 | `scripts/language_note_arms.py` | Correction, and the two sentences · 09-23 | `models/language_note_*.json` | 不是；`AOI_LANGUAGE_NOTE` 開關留著 |
| — | 跨天的推力 | （讀既有資料） | closing adjudication · 09-25 | — | 框到日誌之下，收掉；評測預設釘 batch 512 |

## Jev 的金鑰怎麼申請

Jev 是 TypeSafe AI 的決策模型，只回機率不生成文字；本專案透過 Vercel AI Gateway 呼叫，它在免費層。

1. 到 vercel.com 註冊（Hobby 方案）。**要綁信用卡**才解鎖每月 $5 免費額度；沒綁會回 403 `customer_verification_required`。建議在 Budgets & Spend 設 $1 上限。
2. 儀表板左側 **AI Gateway** → **API Keys** → **Create key**，複製（只顯示一次）。
3. 放進 `~/.zshrc`：`export AI_GATEWAY_API_KEY=vck_...`。不進 repo，`.env` 也在 gitignore。
4. 驗證：`uv run python scripts/jev_refusal_eval.py --limit 2 --repeats 1 --arms full`，每題印出 P(answerable) 與延遲就通了。
5. 原廠金鑰（console.typesafe.ai/keys）也可以：`TYPESAFE_API_KEY`，client 會自動改走 api.typesafe.ai。

## 怎麼重建一張圖

圖裡的每個數字都抄自 benchmarks 的表；要更新，先跑腳本讓它追加新條目，再改 `figures.html` 裡對應的資料陣列（每張圖一段，變數名就是表頭）。不要反過來先改圖。
