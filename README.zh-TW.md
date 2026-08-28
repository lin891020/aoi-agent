# AOI-Agent

PCB AOI 複判系統：視覺模型在複判佇列前面，agent 在後面，每一個主張都有對應的量測。

[![tests](https://github.com/lin891020/aoi-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/lin891020/aoi-agent/actions/workflows/tests.yml)
![python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![PyTorch · LangGraph · MCP · FastAPI](https://img.shields.io/badge/PyTorch%20%C2%B7%20LangGraph%20%C2%B7%20MCP%20%C2%B7%20FastAPI-2d3142)
![licence MIT](https://img.shields.io/badge/licence-MIT-0B6455)
&nbsp; **[English version →](README.md)**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/disposition-flow-dark.zh-TW.svg">
  <img alt="單一標記區域的處置流程：複判模型先分類；高信心的 false call 直接排除、高信心的缺陷直接確認，不經過語言模型；其餘取得生產脈絡與允收標準、由 LLM 產生說明，再依分類器信心決定，或透過可持久化的 interrupt 升級給作業員。" src="docs/diagrams/disposition-flow-light.zh-TW.svg" width="100%">
</picture>

<sub>處置流程。由 `scripts/render_diagrams.py` 讀取 `graph/flow.py` 的 node 名稱與門檻產生。`/ask` 的流程圖見[問產線問題](#問產線問題--ask)。</sub>

## 概要

| | |
|---|---|
| **問題** | 產線 AOI 以 recall 為目標，過度標記。每個標記區域都由人複判，多數為 false call。 |
| **方法** | ResNet-18 複判模型為每個 candidate 評分。LangGraph flow 對不確定者取得脈絡與 LLM 說明，無法定案者透過可持久化的 `interrupt()` 升級給作業員。第二入口 `/ask` 將主管的問題轉成經驗證的型別化查詢計畫與圖表。 |
| **資料** | [DeepPCB](https://github.com/tangsanli5201/DeepPCB) 官方切分：499 片測試板、7,322 個 candidate，41.2% 為真實缺陷。False call 由範本相減產生，非人工編造。 |
| **結果** | **在 ≤0.5% 的 escape budget 下省去 52.8% 的人工複判。** Escape rate 的 95% 區間上緣 0.82%。 |
| **技術** | Python 3.12 · PyTorch（MPS / CPU）· LangGraph · MCP · FastAPI + Jinja · SQLite · Ollama（`gpt-oss:20b`） |
| **驗證** | 1,276 個測試，不需模型或 GPU。每個門檻皆引用出處；每個數字皆註明產生它的腳本。 |
| **限制** | 照片板材上，相減前端無法通過第一道閘門；錫膏影像上，YOLO26n 偵測器可定位 92% 的缺陷，但排序只能省 1.2%。見[遷移](#遷移兩份新資料集)。 |

## Demo

<!-- 影片佔位：錄好後把這一段換成連結。分鏡：docs/demo-script.md -->

示範影片：製作中。流程——跑一片板 → 佇列 → 區域頁 → `0`（無法判斷）→ 資深複判
→ `/boards` → `/ask` 查機台事件 → 對照機台 → 語言切換。

站台畫面（每一頁皆有 English 版）：

| 佇列——agent 無法定案的區域，等候最久者在前 | 區域——範本、待測板、差異圖與交付說明 |
|---|---|
| ![佇列](docs/screenshots/queue-zh.png) | ![區域](docs/screenshots/region-zh.png) |
| **`/boards`——扣住、放行、等待中，以整表計數** | **`/ask`——經驗證的計畫、平行展開、依結果形狀產生的圖表與說明** |
| ![板](docs/screenshots/boards-zh.png) | ![提問](docs/screenshots/ask-zh.png) |

## 快速開始

```bash
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
uv sync                                                  # Python 3.12；torch 使用 MPS 或 CPU
uv run python scripts/build_patches.py --split trainval && uv run python scripts/build_patches.py --split test
uv run python scripts/train.py                           # M5 Air 約 4 分鐘 -> models/reverifier.pt
uv run python scripts/seed_store.py --split test --limit 500
uv run python scripts/add_operator.py mike --role senior # 互動輸入 passphrase
uv run python -m aoi_agent board 20085294 --queue        # 一片板走完整 flow；需要 Ollama + gpt-oss:20b
uv run python -m aoi_agent station                       # http://127.0.0.1:8110
```

`uv run pytest` 執行全部測試，不需模型、GPU 或資料集。量測腳本、容器與 CLI 子命令：[怎麼跑](#怎麼跑)。

## 結果

DeepPCB 官方測試切分：499 片未見過的板、7,322 個 AOI candidate，其中 3,018 個（41.2%）為真實缺陷。

| escape budget | 實際 escape rate | 省去的人工複判 |
|---|---|---|
| ≤0.25% | 0.23% | **40.2%** |
| ≤0.50% | 0.50% | **52.8%** |
| ≤1.00% | 0.99% | **58.3%** |

- **指標。** 漏判（escape）讓缺陷板出貨；false call 只多花作業員幾秒。因此模型以
  escape budget 下的 operating-point 曲線報告，而非 accuracy（參考值 96.5%）。
- **區間。** 0.50% 為此切分上 3,018 個缺陷中的 15 個。95% Wilson 區間上緣 0.82%，
  故僅憑此證據無法確認未見缺陷上的預算達標。
- **分類別。** 出貨門檻下 `short` 的 escape 為 1.55%，為整體的 3.1 倍，且該類別無任何
  作業指導書允收。漏判的 open 其 `P(open)` < 0.006，屬高信心錯誤，模型自身輸出無法
  切分；電性測試可以。超標類別隨 checkpoint 移動（2026-08-24 重訓前為 `open`，1.35%）。
- **盛行率。** 前一版頭條 56.2%（8,143 個 candidate）量測於對位階段之前，2026-08-26
  作廢。對位移除 842 個易判 false call 並多找到 21 個缺陷；複判減量降為 52.8%，
  但送達作業員的區域由 3,568 降為 3,457。複判減量一律附註其假設的盛行率。
- **分流。** 82.2% 的 candidate 由複判模型單獨處置，CPU 上每個 2.5 ms（p50）；
  LLM 僅處理其餘 17.8%。

完整報告：[docs/benchmarks.md](docs/benchmarks.md)，只增不改、新的在後。區間與分類別細節：
[prevalence](docs/benchmarks.md#prevalence--what-survives-a-line-that-is-not-this-dataset-1)
· [per-class escape](docs/benchmarks.md#per-class-escape--one-budget-over-six-classes-that-are-not-alike-1)。

## 量測改變了什麼

六項量測推翻了先前的主張或設計決定，每一項都改了程式。下表每項一列；完整經過與
改前改後的數字在 [docs/findings.zh-TW.md](docs/findings.zh-TW.md)。

| 量了什麼 | 量到什麼 | 改了什麼 |
|---|---|---|
| LLM 在 decision path 上 | 同一批 60 個區域，LLM 43/60、classifier 51/60；LLM 改了 12 次判斷、對 1 次 | 它只解釋、不再決定；`test_the_classifier_class_stands_when_the_llm_disagrees` 守著 —— [完整版](docs/findings.zh-TW.md#llm-本來在-decision-path-上量過之後把它拿掉了) |
| Planner，用作者沒看過的 70 題打分 | 55/70；膽小不亂判（28/28 拒答全對）；之後加了兩個 tool，題庫一字未動、每次重考 | 題庫不改，miss 逐題裁定；2026-08-28 改 prompt 後重跑 —— [完整版](docs/findings.zh-TW.md#planner-是用它作者沒看過的題目打分的) |
| plan 對之後那段文字，對照 payload | 265 句、602 個數字：沒有一個是編的、沒有一個掛錯對象；第一輪 43 件裡 41 件是 checker 自己的錯 | 每個 checker 修正兩邊都有測試 —— [完整版](docs/findings.zh-TW.md#plan-對之後那段文字有沒有照著資料寫也量了) |
| 標準檢索 | 27.8% 的段落來自別的 class 的 work instruction；disposition path 把 pin hole 的規則拿給作業員判 open | 每份文件宣告自己管哪一類、檢索帶 scope：現在 0.0% —— [完整版](docs/findings.zh-TW.md#標準檢索把別的-defect-class-的規定當成這一類的答案) |
| 兩個 threshold 的出處 | `ESCALATE_BELOW` 引一個沒人跑過的 sweep；`CONFIDENT` 引一條沒有數字的條款 | 兩個都 sweep 了；`ESCALATE_BELOW` 在結構上等於 dismiss threshold；29 個測試守每一個引用 —— [完整版](docs/findings.zh-TW.md#兩個-threshold-的出處其實沒寫那件事) |
| 全線 escape rate | 發表 5.4%；實際 0.61%，而且是兩個數字（0.22% 沒標到、0.38% 判掉） | 一個「框得多緊」的統計量被當成漏檢率 —— [完整版](docs/findings.zh-TW.md#全線-escape-rate-被高估了將近一個數量級) |

## 遷移：兩份新資料集

DeepPCB 是二值化、已對位的，等於拿掉了真實產線兩個最大的 false call 來源。所以
2026-08-26 把出貨的 pipeline 原封不動跑在兩份它從來沒被掃過的資料集上，答案比問題
預期的早一層出現。

**HRIPCB —— 照片。** 十片真的板子、每片一張範本、693 張把 defect 畫到範本上的影
像，以及同樣 693 張旋轉 ±10° 以內的版本。在出貨的灰階 threshold 下，相減這一階段
只標到 **16.9%** 的 defect，而且 S0 gate 在*任何*設定下都過不了：recall 最高 92%，
代價是每張 1.8 個 false call，gate 自己的擾動更產生 860 個。DeepPCB 會過，是因為
二值化讓一個 defect 和一條沒對準的邊變成同樣的 255 級差異；在照片上 defect 只差
36 級，而每一條邊都是漸層。把 threshold 設在 recall 最高的地方，複判模型在 0.961
下把 **2,953 個真 defect 判掉 1,387 個——escape rate 47%**，而那條佇列九成是真
defect：一個訓練在 255 級差異上的模型，看到一個很淡的差異就有把握地說它是 false
call。對位那一步拒絕了 693 對旋轉影像裡的 563 對，如設計：它只還原平移，還不了的
時候會說。**相減這個前端的工作範圍是二值化影像**，而這個專案在量出來之前沒有任何
地方這樣寫。
→ [gate](docs/benchmarks.md#s0-gate-on-hripcb--does-template-differencing-produce-a-reviewable-queue-on-photographs)
· [transfer](docs/benchmarks.md#transfer--the-shipped-pipeline-on-hripcb-a-dataset-it-was-never-swept-on)

**PCB-AoI —— 錫膏，沒有範本。** 真實 SMT 檢測影像，元件擺上去本來就有公差，沒有東
西可以相減，所以偵測器是唯一做得出來的前端。一個在筆電上訓了 56 分鐘的 YOLO26n
框得到東西——**覆蓋 91.6% 的 defect**，驗證集 mAP50 0.658，目標中位數 17 px——但排
不了序：把 `P(false call) = 1 − confidence` 拿到 ≤0.5% 的 escape budget 下讀，它只
省得掉 **1.2%** 的佇列，DeepPCB 上相減加複判是 52.8%。兩條線、兩種盛行率，所以不是
排名；形狀才是發現。偵測器的信心是一個掛著分類頭的定位分數，不是校正過的 false
call 機率，所以這個前端目前沒有複判階段。兩份資料集從相反的方向指向同一個下一步：
**在偵測器的框上接一個複判器**——已經有的架構，拿掉範本那個通道。
→ [偵測器](docs/benchmarks.md#detector-front-end--yolo26n-on-pcb-aoi-read-at-the-escape-budget)

**2026-08-28：那個複判器做出來了，它也排不了序。** 同一個 ResNet-18 訓練在偵測器框的
64 px RGB 裁切上（11,928 個訓練 patch，CPU 一次訓練，seed 0），在 ≤0.5% 預算下對同
樣 578 個測試 candidate 省掉 **2.8%**，同一基準上偵測器是 0.3%——而這條佇列 77.7% 是
真 defect，任何排序最多只能省 22.3%。兩種前端現在從相反的方向同意一件事：在這份資料
上，false call 和 defect 光看外觀分不開，這就是為什麼相減前端的範本通道從來不是為了
方便。訓練用的 candidate 來自一個看過訓練影像的偵測器，六十張圖讓每個區間都很寬；兩
點都寫在那一節上。→ [裁切複判器](docs/benchmarks.md#crop-re-verifier--the-resnet-18-over-the-detectors-boxes-against-the-detectors-own-ordering)

兩者都不能證明的：一個從沒訓練過照片的 checkpoint、一個看過結果才選的灰階
threshold、六十張測試圖和很寬的區間、一次訓練一個種子。`scripts/transfer_report.py`、
`scripts/gate_check.py --dataset hripcb` 和 `scripts/detector_report.py` 重建每一個數字。

## 怎麼運作的

處置流程就是這一頁最上面那張圖。底下是同一條 flow 的文字版，然後是圖上看不出來
的那幾件事。

<details>
<summary>同一條 flow 的文字版</summary>

```
template ─┐
          ├─→ difference + threshold + connected components ──→ candidates
test ─────┘            "AOI simulator"                              │
                                                                    ▼
                                                        ResNet-18 re-verifier
                                                     class + calibrated P(fc)
                                                                    │
                        ┌───────────────────────────┬───────────────┴───────────┐
                        ▼                           ▼                           ▼
                P(false call) ≥ .915          conf ≥ .95 and a          everything else
                                            defect other than `open`            │
                        │                           │                           ▼
                        │                           │              production context
                        │                           │            + criteria for that class
                        │                           │               (three MCP tools)
                        │                           │                           │
                        │                           │                           ▼
                        │                           │            LLM writes the rationale
                        │                           │                           │
                        │                           │                  conf ≥ .915 ?
                        │                           │              ┌────────────┴──────────┐
                        ▼                           ▼              ▼                       ▼
                     dismiss                     confirm    classifier's class      escalate to
                                                                  stands            an operator
```

（這段文字裡的 threshold 是 2026-08-23 出貨時的；上面那張圖由
`scripts/render_diagrams.py` 產生時直接讀 `graph/flow.py` 現在的常數——0.961，以及
它上面的 `CONFIDENT` 帶。）

</details>

圖上每一個 disposition 都是 classifier 下的。2026-08-23 之前，中下那個框寫的是「LLM
的 verdict」，而 escalate 那條邊是照 LLM 自己對自己信心的判斷走的；兩個都拿去跟它們
想取代的 classifier 比過，兩個都輸，兩個都拿掉了。LLM 的輸出現在只到一個地方 ——
作業員螢幕上那段話。

### False call 是哪來的

DeepPCB 裡只有真的 defect，所以複判 model 本來沒有東西可以「判掉」。與其自己編
false call，這個 pipeline 是把它**產生**出來的：用一個單純的 template differencing
偵測器 —— 跟真 AOI 同一個原理 —— 產生高 recall 的 candidate，再拿去跟 ground truth
的框比對，逐一標成真 defect 或 false call。

在完整 trainval split 上量：recall 95.3%，每片板子 7.07 個 false call。這個 recall
用的是 DeepPCB 自己的 IoU 0.33 慣例，所以它同時是一個「框得緊不緊」的數字 —— 在 test
split 上，偵測器在 99.78% 的 defect 上都放了 candidate，只是有不少框得比標註者鬆。
兩個數字都在 [docs/benchmarks.md](docs/benchmarks.md)。

只是把一個真 defect **切碎**的 candidate 會被排除在訓練外，而不是標成 false call。
量過那類佔未匹配框的 6.1%，拿去訓練等於教 model 把真 defect 判掉。

### 為什麼是 graph 不是迴圈

單看 confidence gate，一個 `while` 迴圈就夠了。難的是交給人的那一段：被 escalate 的
candidate 要中途暫停、把狀態整包存下來，然後在作業員有空的時候恢復 —— 可能是幾天以
後，而且不能重跑任何 tool。這就是 LangGraph 的 checkpointer 跟 `interrupt` 在做的事。

它對頭條數字也有影響。≤0.5% 的 escape budget 是靠「不確定的送人」達成的，不是靠
model 夠強。把 escalate 這條邊拿掉，同樣的 budget 就得用複判量去換。

## 複判站

Escalate 這條邊總得有個終點，而 CLI prompt 不是 —— 產線不會停下來等人回答問題。
Escalation 進 queue，作業員有空再回。

```bash
uv run python -m aoi_agent board 20085294 --queue   # 跑一片板子，收不掉的丟進 queue
uv run python -m aoi_agent station                  # http://aoi.test
```

`aoi.test` 由本機的 Caddy 代理（設定在 `~/Projects/Caddyfile`），
`http://127.0.0.1:8110` 一樣可以用。站沒開的時候打 `aoi.test`
會看到啟動指令，而不是一片空白的錯誤頁。

站上顯示的就是 agent 當時看到的證據，多的沒有：

- **golden template、待測板、以及兩者的 difference**，並排、放到看得清楚的比例，被標
  的區域框起來。只看 difference 就是 AOI 看到的東西，而只憑 difference 判斷正是
  false call 的來源。
- model 的 class、confidence、P(false call)，以及它實際吃進去的那個 64 px 視窗 ——
  如果視窗歪掉了，那意見不合是 crop 的 bug，不是 classifier 的。
- agent 撈到的生產履歷跟允收標準（現在已經 scope 在對應的 class 上）。
- agent 為什麼不敢判。

有兩件事在設計上不做。它**絕對不顯示 ground truth**：作業員的答案就是下一輪訓練的
label，照答案抄出來的 label 沒有價值。它也**絕對不為了 render 一個頁面去重跑
flow** —— 暫停的狀態就在 checkpointer 裡，讀它只是一次磁碟 seek，而重跑要再燒一次
20B model，還可能吐出跟螢幕上不一樣的理由。

判定是用一般的 form POST 加 redirect，所以關掉 JavaScript 也能用；數字鍵直接選判定，
給整班都在用它的人。

佇列周圍，2026-08-25 起：`/boards` 是索引——扣住、放行、*等待中*三個數字是對整張表
算的，不是對頁面算的，因為佇列只顯示 agent 收不掉的東西，只看得到失敗的人會把失
敗當成系統。按鍵 `0` 是**「我不確定」**：它不寫任何判定（那張表是下一輪訓練的
label，而「不確定」是唯一絕不能成為 label 的答案），把區域移到一份按「幾個人退回」
排序的 `deferred` 清單，而且只有 `senior` 能回答——交給下一個一般作業員等於交回已
經失敗的那個判斷。這是這個站唯一的權限，角色每次請求都從憑證檔讀、不從 session
讀，所以撤掉一個角色是立刻生效。每一頁都能用繁體中文或 English 讀；切換只換介面，
不改紀錄——問題、理由、規劃段保留當時的語言並標示出來。時間戳存 UTC、顯示 UTC，
而且標明是 UTC。

**作業員要先登入，而登入用的那個名字就是最後寫在 label 上的名字。** 這不是因為 queue
是什麼機密 —— 是因為那個答案就是下一輪訓練的 label，而一個作者是文字框的 label，沒有
人有辦法衡量它。判定表單上已經沒有 `reviewer` 欄位：名字從簽章過的 session 來，store
會拒絕寫入一筆說不出是誰做的人工判定，而且那一列還會記下這個名字是**怎麼**建立的
（站上是 `signed_in`，CLI 是 `host_account`），這樣下一輪 retrain 可以照它挑資料。
作業員就是一個檔案：

```bash
uv run python scripts/add_operator.py mike               # 會問 passphrase
uv run python scripts/add_operator.py --list
```

使用者管理僅此而已，這是設計上的選擇： 一條線上的人是固定的那幾個，加上一個會跑 script 的主管。
這套機制**擋不住**什麼，寫在 `src/aoi_agent/station/auth.py` 跟
[benchmarks](docs/benchmarks.md#the-scheme-and-what-it-does-not-protect-against)
裡，因為一套講清楚自己界線的機制，比一套更強但不講的值錢。

### Escalation 在兩邊之間住在哪

`interrupt()` 把 run checkpoint 到 SQLite，另外一張小的 `escalations` table 記著
「還有人欠這件事一個答案」。兩個 store，各回答一個問題：checkpointer 知道這次 run
的狀態，table 知道還有沒有人在等。判定會在關掉 queue entry **之前**先寫進
`review_decisions` —— 中間掛掉的話這個區域會被再看一次，反過來寫的話會默默吃掉作業
員的答案。

這是交接能否持久的關鍵。用 in-memory checkpointer 的話，在單一 CLI run 裡
看起來一切正常，process 一結束 queue 就沒了；撐不過 process 的 interrupt 只是 prompt，不是交接。`tests/test_checkpoint_durability.py` 會在一個直譯器裡發出 escalation、
讓它結束，再用第二個直譯器把它做完。

## 問產線問題 —— `/ask`

第二個入口，給另一種人用。Queue 回答的是「這個區域我要怎麼處理」；`/ask` 回答的是領
班走過來會問的那種 —— 「M22 是不是在飄，這件事要不要緊」。它讀的是 disposition path
同一組 MCP tool，而且它什麼都不 disposition。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/analysis-flow-dark.zh-TW.svg">
  <img alt="/ask 怎麼回答一個問題：一次 LLM 呼叫產生型別化的計畫；計畫對照真實的工具簽名和 store 的值域驗證，不過就連同每一條錯誤拒答；通過的呼叫平行展開，結果收集後從形狀推出圖表，第二次 LLM 呼叫在數字旁邊寫文字，整次執行存下來，圖是從資料重畫、不是重新規劃。" src="docs/diagrams/analysis-flow-light.zh-TW.svg" width="100%">
</picture>

一次 LLM call 產生一份 typed 的 plan。`validate_plan` 在任何東西跑起來之前分三層檢
查：tool 名字、參數名字對照真的 signature、以及參數**值**對照 store 真的有的 domain。
檢查沒過的 plan 會連同每一條錯誤原封不動秀給人看，不會 retry。通過之後 tool 用
`Send` 展開平行跑，某一支失敗會變成資料而不是例外，chart 是從結果的**形狀**推出來的
而不是 model 挑的，最後第二次 LLM call 把文字寫在那些數字旁邊。

那個 fan-out 是工作本身的形狀（這些查詢彼此獨立），不是效能優化。兩次 model call 在
時間上壓倒性地大，所以這裡沒有任何地方把它講成加速。

Planner 做得好不好看上面那份[盲測](docs/findings.zh-TW.md#planner-是用它作者沒看過的題目打分的)；最後那段
文字有沒有照著資料寫，是它後面那一節。2026-08-26 起 store 有了時間軸：`machine_events`
記機台被動了什麼、什麼時候，`query_defect_history` 吃 `relative_to` 和 `side`，所以
「M32 參數變更有沒有動到 open 的比例」是切開該機台所有板子的前後兩個視窗，各帶一個
Wilson 區間。種子種了一個有效果的事件和三個沒效果的，讓工具有機會錯——它在對照組上
回答「沒差」。

### 為什麼沒有 text-to-SQL

生產相關的 tool 都是固定 query set 上的 typed 參數。Model 填參數，它不寫 SQL。

這是對失效模式的判斷，不是難度問題。一句語法正確但語意錯的 query 會回一個看起來很
合理的數字，而且不會報錯；在判定的情境下，一個貌似合理的錯數字比 crash 還糟 ——
因為它會被拿去用。填參數一樣考得到 tool calling 的能力，而 query set 還留得住人工
review。這也是 `/ask` 要驗參數**值**的原因：`line_id="L4"` 不會噴錯也不會回東西，
所以圖上少一條線，那個缺口讀起來是一個 finding，不是一個答案。

## Tools

三個 MCP server，每一個都可以被任何 MCP client 單獨使用：

| server | tools |
|---|---|
| `aoi-classify` | `classify_defect`, `list_candidates` |
| `aoi-production` | `query_defect_history`（帶 `relative_to`/`side` 可取機台事件前後的視窗）, `query_machine_stats`, `query_board_context`, `query_false_call_rate`, `query_machine_events` |
| `aoi-standards` | `search_standards` |

它們是 in-process 直接呼叫 model 跟 query，不是代理到一個 HTTP backend，所以 MCP 這
一層的成本量得出來，不會被藏在一次網路往返底下。

確認它們起得來、也把 tool 廣播出去：

```bash
uv run python scripts/check_mcp_servers.py
```

要從 Claude Desktop 用，加進 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "aoi-classify": {
      "command": "uv",
      "args": ["--directory", "/path/to/aoi-agent", "run", "python",
               "-m", "aoi_agent.mcp_servers.classify"]
    },
    "aoi-production": {
      "command": "uv",
      "args": ["--directory", "/path/to/aoi-agent", "run", "python",
               "-m", "aoi_agent.mcp_servers.production"]
    },
    "aoi-standards": {
      "command": "uv",
      "args": ["--directory", "/path/to/aoi-agent", "run", "python",
               "-m", "aoi_agent.mcp_servers.standards"]
    }
  }
}
```

## 一個 candidate 要多少錢

Re-verifier 是一個吃 3×64×64（template、test、difference 疊起來）的 ResNet-18：
硬碟上 **42.7 MB、11.2 M 參數**，CPU 上每個 candidate p50 **2.50 ms**（p90 2.53 ms，
300 次，4 個 torch thread）。計時涵蓋的是 pipeline 真正跑的那條路 —— uint8 轉 float、
搬上 device、forward、softmax、搬回 host —— 因為只計 forward 會把搬移藏起來，而那在
MPS 上不是免費的。

**Batch 1 的時候 GPU 是比較慢的那個**：MPS p50 7.34 ms，慢 2.9 倍。model 這麼小的時
候，把 forward 派下去的成本比跑它還高。MPS 要到 batch 8 才追上來。一次判一個區域的
複判站不該用 GPU；一次判整片板子的 seeding 那支才該用。

兩個違反直覺、很容易被不小心改掉的結果：這台無風扇機器上持續 CPU 推論過了第一分鐘
會掉約 20%；而 CPU 的每 candidate 成本在 batch 8 之後會**變差**好幾倍 —— 換過 thread
數確認過，那是 model 在 CPU 上 convolution 路徑的性質，不是這台筆電核心數的問題。
CPU 就 batch 8。

這一節以前寫的是「數十毫秒」，底下沒有任何一次 run。它錯了超過一個數量級，而且錯在
悲觀的方向。
→ [這次的 run](docs/benchmarks.md#re-verifier-latency--what-one-candidate-costs-and-on-what-hardware)

### 量化它，並且用 escape budget 的價格來算

Model 匯出成 ONNX 之後量化成 INT8，兩種做法：dynamic，以及用 **training** split
抽出來的 512 個 patch 做校正的 static。兩個都在完整的官方 test split 上重跑，然後用
這個專案唯一的讀法來讀：**在某個 escape budget 下，砍掉多少人工複判**。

| ≤0.5% escape budget | 砍掉的複判 | 硬碟 | 常駐記憶體 | p50 |
|---|---|---|---|---|
| FP32 torch | **52.8%** | 42.7 MB | 389 MB | 2.53 ms |
| INT8 dynamic | 52.5% | 10.7 MB | 74 MB | 1.93 ms |
| INT8 static | **53.0%** | 10.8 MB | 81 MB | 0.65 ms |

**在這個 checkpoint 上，兩種 INT8 都守住了曲線**，都在報表當初寫死的 1 個百分點
容忍度之內，於是規則挑硬碟省最多的那個：INT8 dynamic，贏 0.1 MB。差距在雜訊範圍內：兩個引擎在部署預算上差 0.5 個百分點，換算是 7,322 個
candidate 裡 15 對 12 個判定不同。活下來的發現不是「哪一種 INT8」，是 INT8 守得住
operating point 這件事本身。

**判決在兩個 checkpoint 之間改變了。** 2026-08-26 之前這一節拒收 INT8 dynamic：在前一個
checkpoint 上它丟了 1.3 個百分點，大約一個班別裡八十個區域回到作業員面前，檔案變小
買不回這件事。那個損失沒有活過 2026-08-24 的重訓 —— 它是某一組權重的量化誤差，不是
dynamic 量化的性質 —— 所以 `scripts/quantisation_report.py` 現在進了重訓的鏈：量化的
判決是對一組權重的判決，下一組要重新定價。

INT8 買到的不是延遲：平均一片板子 16.1 個 candidate，FP32 複判是一片板子 41 ms，
推論從來就不是瓶頸。它買到的是**記憶體** —— 常駐從 389 MB 降到 74–81 MB，約 5 倍 ——
因為 float32 那個 process 大部分是 torch runtime 而不是權重，而 edge 機器是照它要裝下
多少東西去挑的。這是量出來的，不是上線的：這台站台是一台沒有記憶體問題的筆電，而已
部署的 threshold 留在當初掃它出來的那個 float32 model 上。

→ [這次的 run](docs/benchmarks.md#quantisation--what-int8-costs-at-the-escape-budget)

## 怎麼跑

```bash
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
uv sync

uv run python scripts/gate_check.py                      # differencing 真的產得出 false call 嗎？
uv run python scripts/build_patches.py --split trainval
uv run python scripts/build_patches.py --split test
uv run python scripts/train.py                           # M5 Air 上約 4 分鐘
uv run python scripts/report.py                          # operating-point 表

uv run python scripts/seed_store.py --split test --limit 500
uv run python scripts/add_operator.py mike               # 誰可以回 queue
uv run python -m aoi_agent board 20085294                # 跑一片板子過整條 flow
uv run python -m aoi_agent corrections                   # 作業員推翻 model 的紀錄
```

既有的 store 是原地加欄位的 —— `uv run python scripts/seed_store.py --migrate-only`
—— 因為裡面那些更正就是下一輪訓練的 label，不可以為了加一個欄位就重建掉。

上面講的那些量測都是 script，不是截圖 —— `threshold_sweep.py`、
`retrieval_report.py`、`escape_accounting.py`、`opening_kernel_sweep.py`、
`reverifier_latency.py`、`agent_eval.py`、`analysis_eval.py`、
`synthesis_eval.py`。每一支都往
`docs/benchmarks.md` 後面接，新的在最後面，舊的不改。

需要 [Ollama](https://ollama.com) 跟一個會 tool calling 的 model（預設
`gpt-oss:20b`）。全部在本機跑，沒有任何東西離開這台機器 —— 在產線上這是要求，不是
偏好。

**1,276 個測試。** 其中 1,249 個在乾淨 checkout 上就能在 CI 跑完 —— 它們自己在
tmpdir 裡建 store、建 Chroma collection、建板子，model 是 stub 掉的。另外 25 個要磁碟
上有資料集，帶 `dataset` marker；CI job 每次跑完都會把它們列出來，因為「測試
數量默默變少但綠燈照亮」正是那個 job 要防的事。

### 用容器跑

```bash
docker build -t aoi-agent .
docker run --rm -p 8110:8110 \
  -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
  aoi-agent                                              # station 在 :8110
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
  aoi-agent python -m aoi_agent queue                    # 或任何一個 CLI subcommand
```

image 裡沒有重的東西。資料集、patch、權重、SQLite store 跟 Chroma index 全都是上面那
些 script 建出來的，也全都 gitignore；它們從那兩個 mount 進來，image 只裝 code 跟
wheel。什麼都不 mount 直接跑，會得到一個對著空 queue 起來的 station，那比 import 就
炸掉清楚。

作業員檔案跟 station 讀的其他東西一樣放在 `data` mount 上，所以容器拿作業員的方式跟
拿 store 的方式一樣。想讓 session 撐過重啟就設 `AOI_AGENT_SESSION_SECRET`；不設的話
每個 process 自己生一把，重啟之後大家重新登入。

有兩件事這個容器不是。它沒有 GPU：Linux image 拿的是 torch 的 CPU build，這是刻意的
—— CUDA 的 wheel 會為了這個專案從來沒有過的硬體拉進好幾 GB 的 runtime，所以
`pyproject.toml` 在 linux 上解到 PyTorch 的 CPU index，macOS 留在 PyPI，MPS 照樣能用。
它也沒有 model server：Ollama 留在 host，所以 flow 的解釋那一步需要容器連得到它。

## 已知限制

- **全線 escape rate 是 0.61%，而且它是兩個數字不是一個。** 0.22% 的 defect（test
  split 上 3,140 個裡的 7 個）身上完全沒有 candidate —— 那些救不回來。另外 0.38% 是
  有標出來但被 re-verifier 判掉的，那才是 dismiss threshold 管的。這個數字以前寫什麼、
  以及為什麼大了九倍，[在 docs/findings.zh-TW.md](docs/findings.zh-TW.md#全線-escape-rate-被高估了將近一個數量級)。
- **Escape budget 平均有達標，但有一類超標。** 在出貨的 threshold 下 `short` 的
  escape 是 1.55%，整體的 3.1 倍；超標的是哪一類會跟著 checkpoint 移動——2026-08-24
  重訓之前是 `open` 的 1.35%。它們是有把握的錯，模型自己的輸出切不開；flow 仍然不
  管 confidence 多高都把每個 `open` 送去 investigation，因為導通是二元的，WI-201 不
  允許任何一個 open。
- **DeepPCB 是已經對位、已經二值化的**，等於把現實世界兩個最大的 false call 來源拿掉
  了。它的 defect 也有一部分是資料集作者疊上去的，不是自然發生的。
- **3×3 的 opening kernel 就是那 0.22% 的去處，而它還是留著。** 它清掉的是對位誤差在
  trace 邊緣留下的細絲 —— 量過，合成板上 2 px 的 template 位移會動到 456 個 pixel，
  在預設參數下產生零個 candidate —— 但它同時也清掉了那 7 個沒被標出來的 defect。它們
  是**細**不是小：difference blob 有 24–133 個 pixel，最厚的地方離自己邊緣最多
  1.37 px，而 3×3 方形要活下來需要 1.5 px。Sweep 過，把它們救回來的代價是每救回一個
  「re-verifier 之後還會留著」的 defect 要多付 918（3×3 十字，救回 7 個中的 5 個）到
  2,888（2×2 方形，7 個全救）個 false call，而且每片板子的 candidate 變成 1.6–8.5 倍
  —— 這還沒算對位誤差那一欄，那邊帳單還會再漲。這個常數沒有動，而且不重跑
  [那份 sweep](docs/benchmarks.md#the-opening-kernel--what-the-seven-lost-defects-would-cost-to-recover)
  就不會動。真的 AOI 比這個吵多了。
- **標準回答的還是作業員沒在問的問題。** Scope 修好的是「段落來自哪份文件」，不是
  「那段話在說什麼」：`open` 拿到的規則還是「任何確認的 open 都是 critical」，那是怎麼
  **處置**，不是怎麼**確認** —— 而站在影像前面的人正在做的是確認。這已經是文件的問題。
- **一個數字同時做兩件事，代價是站台大部分的解釋都寫不出來。**
  `RESPONSE_BUDGET_S` 既是 WI-300 對「判定」的 10 秒承諾，也是 httpx 的 client
  timeout；而這個 model 量到的 service time 中位數是 12.5 秒，所以 24 次呼叫有 20 次
  被砍掉 —— 而 LLM 從 decision path 上拿掉之後，寫給作業員看的那段解釋是它僅存的工作。
  Queue 上曾經有一筆升級案，全部內容就是 `the model did not answer (ReadTimeout)`，
  而且沒有任何東西在算這種情況發生過幾次。承諾不能跟著 model 走，資源上限必須跟著量測
  走，所以現在是兩個常數：budget 維持 10 秒，管的是判定，而判定是 classifier 的
  2.5 毫秒；`EXPLANATION_DEADLINE_S` 是 60 秒，管的是一段沒有人在等的等待。
  用實際出貨的設定重量一次：中位數 8.6 秒、p90 11.1 秒、**24 次呼叫有 0 次沒寫出解釋**。
  「沒有解釋」現在是一個一級狀態，會以說明的形式顯示，並且由
  `uv run python -m aoi_agent explanations` 計數 ——
  [這次的 run](docs/benchmarks.md#agent-layer-latency--does-the-reason-node-fit-the-explanation-deadline)。
- **生產履歷是模擬的，種了兩個訊號、三個對照。** 公開的缺陷資料集不會附批號或機台
  id。種子從不寫 defect，它只決定哪片 DeepPCB 板去了哪台機器。M22 拿到 open 最多的
  五分之一；M32 帶一個 `parameter_change`，之後它的 open 佔比下降；另外三台帶的事件
  沒有效果，所以拿種子去考工具，考的是「有沒有影響」而不是「有沒有事件」；效果的
  鏡像落在一台指名的機器上，不是抹在對照組裡。見 `src/aoi_agent/store/seed.py`。
- 允收標準是為這個專案寫的原創文件。IPC-A-610 之類的有版權，刻意不放進來。
- **登入讓一個名字可以被追溯，但沒有讓它變成真的。** 兩個人共用一組 passphrase，兩
  個人的 label 上就會是同一個名字，這件事任何不用工號卡的機制都解不掉。Session
  cookie 是 bearer token，安全性就是你把 station 擺在什麼傳輸層後面；登入沒有速率限
  制也沒有鎖定；而任何有 host shell 的人都可以直接寫那個 SQLite —— 這正是 CLI 的判
  定被記成 `host_account` 而不是跟站上登入同一個字的原因。它足夠用來衡量一個訓練
  label，不足以在爭議裡拿來壓住誰 ——
  [完整寫在這裡](docs/benchmarks.md#the-scheme-and-what-it-does-not-protect-against)。
- **有 9,140 筆判定早於這個歸屬欄位，而且它們自己說了。** 它們是 `unrecorded`，由
  migration 蓋上去的，不是留成 `NULL` —— 一筆從來沒記過 reviewer 的判定，不可以被讀
  成一筆本來就沒有 reviewer 的判定。它們就維持這樣；第一輪 retrain 必須自己講清楚它
  放掉了多少。
- **這個專案的十九條不變式裡，有兩條只守住一半，還有一條根本守不住。**
  `CLAUDE.md` 列了十九條不能被悄悄改掉的規則；`scripts/invariant_audit.py` 會報出
  哪幾條真的會在被違反時讓測試失敗，而 `tests/test_invariant_audit.py` 會在某一條
  失去守衛時掛掉。十六條有守。fan-out 那條和官方 split 那條各自只守住一部分，而且逐條寫明守住的是哪一部分；「說清楚哪些是
  模擬的」是散文紀律，被明確宣告為無法測試，而不是算它通過。每一格都是真的去破壞
  那條規則、跑完整套測試得出來的 ——
  [稽核結果](docs/benchmarks.md#the-invariant-audit--which-of-this-projects-own-rules-are-unguarded)。

## 還沒做的

- **給偵測器前端一個排序。** 框上的複判器 2026-08-28 做出來了，在預算下省 2.8%，
  而那條佇列的上限是 22.3%——還不是排序。沒試過的：偵測器用 `imgsz=1280` 訓練（它的
  小目標設計就是為這個尺寸）；用一個*沒看過*訓練影像的偵測器產生裁切（在留出的 fold
  上訓第二個偵測器）；以及同一片板的第二次拍攝——那是唯一能給這條線一張範本的東西。
- **從作業員更正回頭 retrain。** 判定歷史有記（`uv run python -m aoi_agent
  corrections`），每一列寫得出是誰做的、那個名字怎麼建立的，所以下一輪可以只吃
  `signed_in`，或把其他的權重壓低。還沒有東西去用它；改變的是這個選擇存在了。
- **把量化後的 model 真的接上去**，這是一個決定，不是一個缺口。兩種 INT8 都守得住曲
  線、常駐記憶體少約 5 倍；沒接進站台是因為這台站台沒有記憶體問題，而延遲從來不是瓶
  頸。要接的話，需要在 `ReVerifier` 裡開一條 ONNX 路徑，並針對真正要服務的引擎重掃一
  次 threshold。
- **跨 model 比較**：`gpt-oss:20b`、`qwen3:14b`、`qwen2.5:14b`。reason node 的延遲只
  在一個 model 上量過；更小的 model 進不進得了 explanation deadline、還寫不寫得出堪用
  的理由，沒有量。
- **登入刻意沒做的那些。** TLS（cookie 是 bearer token，process 講的是明文 HTTP），
  以及登入端點的速率限制或鎖定。兩件都寫在 `station/auth.py` 裡，不會在沒有寫下理由
  的情況下加上去。
- **示範影片**，格子在這份文件最上面。
