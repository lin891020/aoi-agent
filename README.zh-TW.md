# AOI-Agent

[![tests](https://github.com/lin891020/aoi-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/lin891020/aoi-agent/actions/workflows/tests.yml)
&nbsp;·&nbsp; [English](README.md)

**AOI 標出來的每一個點都要再看一次 —— 但先看的是 model，不是人。**

PCB 產線上的 AOI 是照 recall 去調的，所以一定 over-flag。標出來的每一個區域現在
都要送到複判站給人看，而人看完大部分都是沒事的 false call。這個專案在那條 queue
前面放一個 vision model，後面再放一個 agent，接手 model 自己收不掉的那些。

資料是 [DeepPCB](https://github.com/tangsanli5201/DeepPCB)：真的板子掃描、真的
defect，false call 也是用真的演算法跑出來的，不是編的。

## 結果

官方 DeepPCB test split，499 片沒看過的板子、8,143 個 AOI candidate：

| escape budget | 實際 escape rate | 省掉的人工複判 |
|---|---|---|
| ≤0.25% | 0.23% | **50.2%** |
| ≤0.50% | 0.47% | **56.2%** |
| ≤1.00% | 0.97% | **60.6%** |

Accuracy 故意不放頭條。把真的 defect 判掉是讓壞板子出貨，把 false call 留著只是多
花作業員幾秒鐘；這兩種錯的代價差太多，不能混在一個數字裡。所以這裡報的是一條對
escape budget 的曲線，不是單一數字。（參考用：整體 accuracy 96.5%。）

而且 **82.2% 的 candidate 根本不會碰到 language model** —— 它們由 vision model 直接
處理掉，CPU 上每個 **2.5 ms**（p50，單張，300 次；MPS 是 7.3 ms，batch 1 的時候
GPU 反而輸）。LLM 只花在真正模稜兩可的那 17.8%，這才是 20B model 在產線節拍下還付
得起的原因。

完整報告在 [docs/benchmarks.md](docs/benchmarks.md)。底下每個數字都指得到出處。

## 量出來之後改掉的東西

這個 repo 比較值得看的不是 pipeline，是這五件量完發現是錯的事 —— 其中三件錯在對自己
有利的方向 —— 以及量完之後 code 改成什麼樣子。

### LLM 本來在 decision path 上。量過之後把它拿掉了。

原本的設計是讓本地 LLM 讀完證據直接下判斷。拿 router 送去 investigation 的 60 個
candidate 量：LLM **43/60 = 71.7%**，同一批 candidate 上 classifier 是
**51/60 = 85.0%**。LLM 改掉 classifier 判斷的有 12 次，改對 **1 次**；其中 9 次是
classifier 本來就對的被改壞，2 次是兩邊都錯。

它自己的 `confident` flag —— 決定誰要送人工的那個 —— 也輸給直接對 classifier 已經
算好的 confidence 拉一條 threshold。所以兩件事都還給比較會做的那個 model：
`route_after_reason` 改成看 `ESCALATE_BELOW`，`decide_node` 直接拿 `model_class`，
LLM 寫的東西就是作業員讀的東西。它負責解釋，不再決定誰該被看。改完之後重跑：現在
的系統 **27/30 = 90.0%**，被換掉的那個做法 **22/30 = 73.3%**。

這個量測釘在會被它影響的 code 裡 —— `decide_node` 的 docstring 就寫著 12 改 1 對，
`test_the_classifier_class_stands_when_the_llm_disagrees` 會在有人把判斷權還給 LLM
的時候紅掉。

**這不代表什麼：** 60 個 candidate、一個 model（`gpt-oss:20b`，`think="low"`）、
一份 prompt。它是關於這個 classifier 跟這份 prompt 的結論，不是關於 language model
的結論。
→ [這次的 run](docs/benchmarks.md#agent-layer--does-it-beat-the-classifier-and-is-the-escalation-calibrated)

### Planner 是用它作者沒看過的題目打分的。

第二個入口 `/ask`：把領班問的問題變成一份 typed 的 tool call plan，不然就 refuse。
用寫 prompt 的人自己出的 20 題打，100% —— 這件事講的是題目，不是 planner，那一節
自己也是這樣寫的。

所以另外找了三個沒看過 prompt、few-shot examples 跟 fixture 的作者出 70 題：一個完
全不知道有哪些 tool，只被要求寫「領班會打什麼字」，出 35 題；一個只拿到五個 tool
的 signature，被要求去戳邊界，出 35 題；第三個讀了 tool 跟 store 的 source 但沒看
prompt，負責訂正確答案。

**55/70 = 79%**，而重點是錯的形狀：

| | 題數 | 對 |
|---|---|---|
| 該回答的 | 42 | **27/42 = 64%** |
| 該拒絕的 | 28 | **28/28 = 100%** |

15 個 miss 裡面有 7 個是「該答的它去 refuse」。它是**膽小，不是亂判** —— 在產線
上這是活得下去的那個方向：什麼都敢答的 planner 比會說「我答不了」的危險。照嚴重度
分，答案無爭議的那組 42/51 = 82%，兩個 grader 可能會吵起來的那組 13/18 = 72%。另外
有 7 題不管 planner 怎麼規劃都不可能過（grader 把一個參數釘在根本不吃它的 tool 上），
這 7 題照樣算錯；扣掉之後是 55/63 = 87%。

這套題目還挖出兩個分數裡看不到的真 bug：

- `query_machine_stats` 的 `days` 預設 14，可是 store 只有 9 天。沒帶這個參數的 plan
  會照跑，回整整 9 天的資料，然後標成 `14`。
- **沒有任何一個 tool 回得出彙總層級的 false call rate** —— 而這整個系統的主題就是
  false call。領班那 35 題裡有 6 題在問這個，不管是分機台、分班別還是分線別。以現在
  的系統來說 refuse 是對的，但那是「對的答案，錯的問題」。兩個都沒有順手補掉，因為
  為了讓 set 過而加 tool，那個 set 就不再是量測了。

**這不代表什麼：** 題目是 LLM 作者按不同 brief 寫的，不是真的領班寫的，所以它只框住
這些 brief 生得出來的題型。而且評的是 plan 不是文字 —— 資料對的情況下寫出來的那段
話對不對，是下面那一節。
→ [獨立那次的 run](docs/benchmarks.md#analysis-planner-asked-by-someone-else--does-it-plan-the-right-lookups-and-refuse-the-rest)

### plan 對之後，那段文字有沒有照著資料寫，也量了。

*「我怎麼知道它什麼時候在唬爛？」* Tool 是 deterministic 的，所以對「數字」而言這是
算術題：那個數字要嘛從 payload 裡的某個值算得出來，要嘛算不出來；它被掛在哪個機台、
哪條線、哪一類上，要嘛對得起來，要嘛對不起來。分成五種錯分開報，因為「編出一個數字」
跟「該保留卻沒保留」不是同一個 accuracy：

| 種類 | 件數 | 誰判的 |
|---|---|---|
| payload 裡根本沒有的數字 | **0** | 比對 |
| 數字是真的，但掛錯機台／線／類別 | **0** | 比對 |
| 講了因果，或講了時間上的趨勢 | 3 | 人，看 flag |
| Tool 掛了而文字沒講 | 0 | 人，看 flag |
| 對著沒檢索到的類別講規定 | 1 | 人，看 flag |

**34 份答案、265 個句子、602 個數字裡，沒有一個是編的，也沒有一個掛錯對象。** 那 4 個
flag 一個一個判過，其中 3 個是 pattern 的問題不是 model 的問題 —— `leads to` 出現在
引用的處置規定裡、`because the` 出現在「這個 tool 沒回東西」的正確保留裡、`limit` 出現
在 `limited to copper` 裡（這個是 bug，已經修）。而唯一一題直接問因果的
—— *是因為蝕刻液老化了嗎* —— model 沒有給因果。

**第一次跑出 43 件，其中 41 件是 checker 自己的錯**，這才是比較有用的那一半。`M12` 被
當成數字 12；中文答案根本沒被切句，因為 `。` 後面不接空白；`19 copper, 22 mousebite`
被反著讀。每一個修正都讓 checker 變安靜，而那正是 checker 變瞎的方式，所以每一個修正
兩邊都有測試 —— 現在會過的那個句型，跟同一個句型換掉一個值、它必須還是抓得到。

**這不代表什麼：** 只跑了一次，而且是 sample 不是 deterministic；checker 跟被檢查的
系統同一個作者；還有「數字全對但話講錯」—— *「M22 是最差的機台」* 而它其實是第二差
—— 這五種一種都涵蓋不到。刻意寫錯的那份 summary 是 control，用來擋掉「這套標準根本
沒有東西會不過」。
→ [這次的 run](docs/benchmarks.md#the-prose-over-the-results--is-the-sentence-true-of-the-payload)

### 標準檢索把別的 defect class 的規定當成這一類的答案。

這是讀 queue 讀出來的，不是測試抓到的。store 裡五筆 escalation 全部告訴作業員：open
要看它是不是落在 pad 裡面。沒有任何文件這樣寫 —— 那是 WI-206 管 pin hole 的規則。
WI-201 寫的是任何確認的 open 都是 critical，因為導通與否是二元的。

用 6 個 class × 6 種真實問法、`top_k=2` 量：**27.8% 的段落來自別的 class 的 work
instruction**，`short` 最慘 67%；而且在 disposition path 自己的 `open` query 上，
pin hole 的 disposition 段落排**第一**，排在 WI-201 自己前面。修的是檢索邊界不是
prompt：每份文件宣告自己管哪個 class，這個宣告跟著每個段落走，帶著 class 來問的
caller 只會拿到那個 class 的文件加上兩份管全部 class 的。現在 **0.0%**，用真文件寫
的測試守著。

這個專案平常的辯護 —— 「LLM 只負責解釋，判斷是 classifier 做的」 —— 在這件事上不
成立。那條編出來的規則送到的正是會做判斷的人手上，而且指向放行一個 critical
defect。舊檢索下已經寫出去的八段說明就地標記，沒有刪掉也沒有重生成。

**這不代表什麼：** 這個數字量的是段落來自哪份文件，不是那段話有沒有回答作業員的問
題。`open` 現在拿到的還是「怎麼 disposition」，而盯著影像的人要的是「怎麼確認」。
那已經是文件的問題，還沒解。
→ [污染率表](docs/benchmarks.md#cross-class-contamination-in-the-criteria-retrieval)

### 兩個 threshold 的出處，其實沒寫那件事。

`ESCALATE_BELOW` 的註解寫「不增加 escape 的最低 threshold」，可是那個 sweep 從來沒
人跑過。`CONFIDENT` 引的是 WI-300 裡一段根本沒提到數字的條款。後來把 sweep 補上，
發現前者兩個方向都錯：這個 split 上不增加 escape 的最低 threshold 是 **0.875**，不
是 code 裡的 0.90；而 0.90 也不是 sweep 出來的 —— 它就是一個剛好偏保守的整數。

兩個都沒有 ship。0.875 距離這個分支會判掉的最高信心真 defect 只有 **0.003**，那叫
把 test split 讀到小數第三位。真正不需要 split 的值是 dismiss threshold 本身：
`ESCALATE_BELOW` 現在就**等於**它，agent 分支唯一可能判掉真 defect 的那個帶寬因此
在結構上是空的。**agent 分支可以 confirm defect，永遠不能 dismiss** —— 而且這件事
retrain 之後還成立，sweep 出來的數字則要重 sweep，而且不 sweep 也不會有人發現。代價
是 8,143 個 candidate 裡多 47 筆 escalation，佔 queue 的 0.6%。

`CONFIDENT` 則根本不是品質關卡。`confirm_node` 跟 `decide_node` 寫的是同一個 verdict，
所以在 `ESCALATE_BELOW` 以上，它改變的 disposition 是**零**，增加的 escape 也是零；
一路 sweep 到 0.999 還是零。它決定的是誰會拿到一次 LLM call 跟一段書面理由，不是板
子的下場。它唯一不能做的是掉到 `ESCALATE_BELOW` 以下 —— 那裡它會開始把流程本來要送
人工的區域直接 confirm 掉。約束才是那個 citation，裡面的值只是旋鈕。

現在每個 threshold 都引得到一支跑得起來的 script 或一份文件裡寫著那個數字的那一行，
而且有 **29 個測試**會在值跟出處對不上、出處失效、或有新 threshold 進了 code 卻沒進
表的時候紅掉。
→ [sweep](docs/benchmarks.md#threshold-sweep--escalate_below-and-confident-2026-08-23--commit-68e90b6)
· [表](docs/architecture.md#thresholds-and-where-they-come-from)

### 全線 escape rate 被高估了將近一個數量級。

本來寫 **5.4%**，實際是 **0.61%**。舊的數字把 5.0% 的「AOI 階段漏檢率」加上
re-verifier 自己的，配上一句「這些已經沒了，任何 threshold 都救不回來」 —— 那句話對
7 個 defect 是真的，卻被套在 157 個上面。那 5.0% 從來就不是偵測不到的數量，它算的是
最佳 candidate 沒過 DeepPCB IoU 0.33 那條線的 defect，而其中 150 個身上是有 candidate
的。一個「這個偵測器框得多緊」的統計量，被當成漏檢率發表出去。

改用 defect 而不是 box 重算，它是兩個數字不是一個：**0.22%** 完全沒被標出來（3,140
個裡的 7 個，救不回來），**0.38%** 有標出來但被 re-verifier 判掉（這才是 dismiss
threshold 管的那個）。把兩個加成一個頭條就是 5.4% 的來源，而且它會叫讀者去調那個根
本動不了的東西。

這一項錯在對自己有利的方向：它把 150 個 defect 記在一個其實沒出錯的階段頭上，同時
把它們從唯一能推翻這個說法的量測裡拿掉。
→ [重算](docs/benchmarks.md#whole-line-escape-rate-recounted-on-defects-instead-of-boxes)

## 怎麼運作的

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
uv run python -m aoi_agent station                  # http://127.0.0.1:8000
```

站上顯示的就是 agent 當時看到的證據，多的沒有：

- **golden template、待測板、以及兩者的 difference**，並排、放到看得清楚的比例，被標
  的區域框起來。只看 difference 就是 AOI 看到的東西，而只憑 difference 判斷正是
  false call 的來源。
- model 的 class、confidence、P(false call)，以及它實際吃進去的那個 64 px 視窗 ——
  如果視窗歪掉了，那意見不合是 crop 的 bug，不是 classifier 的。
- agent 撈到的生產履歷跟允收標準（現在已經 scope 在對應的 class 上）。
- agent 為什麼不敢判。

有兩件事是刻意不做的。它**絕對不顯示 ground truth**：作業員的答案就是下一輪訓練的
label，照著答案抄出來的 label 一文不值。它也**絕對不為了 render 一個頁面去重跑
flow** —— 暫停的狀態就在 checkpointer 裡，讀它只是一次磁碟 seek，而重跑要再燒一次
20B model，還可能吐出跟螢幕上不一樣的理由。

判定是用一般的 form POST 加 redirect，所以關掉 JavaScript 也能用；數字鍵直接選判定，
給整班都在用它的人。

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

使用者管理就這樣，刻意的 —— 一條線上的人是固定的那幾個，加上一個會跑 script 的主管。
這套機制**擋不住**什麼，寫在 `src/aoi_agent/station/auth.py` 跟
[benchmarks](docs/benchmarks.md#the-scheme-and-what-it-does-not-protect-against)
裡，因為一套講清楚自己界線的機制，比一套更強但不講的值錢。

### Escalation 在兩邊之間住在哪

`interrupt()` 把 run checkpoint 到 SQLite，另外一張小的 `escalations` table 記著
「還有人欠這件事一個答案」。兩個 store，各回答一個問題：checkpointer 知道這次 run
的狀態，table 知道還有沒有人在等。判定會在關掉 queue entry **之前**先寫進
`review_decisions` —— 中間掛掉的話這個區域會被再看一次，反過來寫的話會默默吃掉作業
員的答案。

這也是這個 graph 誠不誠實的地方。用 in-memory checkpointer 的話，在單一 CLI run 裡
看起來一切正常，process 一結束 queue 就沒了；那時候 `interrupt` 只是穿著 graph 外衣
的 prompt。`tests/test_checkpoint_durability.py` 會在一個直譯器裡發出 escalation、
讓它結束，再用第二個直譯器把它做完。

## 問產線問題 —— `/ask`

第二個入口，給另一種人用。Queue 回答的是「這個區域我要怎麼處理」；`/ask` 回答的是領
班走過來會問的那種 —— 「M22 是不是在飄，這件事要不要緊」。它讀的是 disposition path
同一組 MCP tool，而且它什麼都不 disposition。

一次 LLM call 產生一份 typed 的 plan。`validate_plan` 在任何東西跑起來之前分三層檢
查：tool 名字、參數名字對照真的 signature、以及參數**值**對照 store 真的有的 domain。
檢查沒過的 plan 會連同每一條錯誤原封不動秀給人看，不會 retry。通過之後 tool 用
`Send` 展開平行跑，某一支失敗會變成資料而不是例外，chart 是從結果的**形狀**推出來的
而不是 model 挑的，最後第二次 LLM call 把文字寫在那些數字旁邊。

那個 fan-out 是工作本身的形狀（這些查詢彼此獨立），不是效能優化。兩次 model call 在
時間上壓倒性地大，所以這裡沒有任何地方把它講成加速。

Planner 做得好不好看上面那份[盲測](#planner-是用它作者沒看過的題目打分的)；最後那段
文字寫得對不對，完全沒有量。

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
| `aoi-production` | `query_defect_history`, `query_machine_stats`, `query_board_context` |
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
| FP32 torch | **56.2%** | 42.7 MB | 389 MB | 2.52 ms |
| INT8 dynamic | 54.9% | 10.7 MB | 74 MB | 2.01 ms |
| INT8 static | **56.0%** | 10.8 MB | 81 MB | 0.72 ms |

**INT8 dynamic 不收。** 它拿 1.3 個百分點的複判減量去換一個比較小的檔案 —— 那大約
是一個班別裡八十個區域重新回到作業員面前。快 1.25 倍買不回這件事，因為本來就沒有人
在等那幾毫秒。

**INT8 static 守住了曲線**，只差 0.2 個百分點，是值得留下的那一個。它買到的不是延遲：
平均一片板子 16.3 個 candidate，FP32 複判是一片板子 41 ms，推論從來就不是瓶頸，量化
它只是在十秒的預算裡省下 29 ms。它買到的是**記憶體** —— 常駐從 389 MB 降到 81 MB，
4.8 倍 —— 因為 float32 那個 process 大部分是 torch runtime 而不是權重，而 edge 機器
是照它要裝下多少東西去挑的。這是量出來的，不是上線的：這台站台是一台沒有記憶體問題
的筆電，而已部署的 threshold 留在當初掃它出來的那個 float32 model 上。

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

**931 個測試。** 其中 923 個在乾淨 checkout 上就能在 CI 跑完 —— 它們自己在 tmpdir
裡建 store、建 Chroma collection、建板子，model 是 stub 掉的。另外 8 個要磁碟上有那份
231 MB 的 DeepPCB，帶 `dataset` marker；CI job 每次跑完都會把它們列出來，因為「測試
數量默默變少但綠燈照亮」正是那個 job 要防的事。

### 用容器跑

```bash
docker build -t aoi-agent .
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
  aoi-agent                                              # station 在 :8000
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
  以及為什麼大了九倍，[在上面](#全線-escape-rate-被高估了將近一個數量級)。
- Escape 集中在 `open`（≤0.5% budget 下 1.35%）—— trace 上細細的斷點，最難跟對位誤差
  造成的假影分開。這也是 flow 不管 confidence 多高都把每個 `open` 送去 investigation
  的原因。
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
- **生產履歷是模擬的。** 公開的缺陷資料集不會附批號或機台 id。板子是照 open defect
  佔比排序分配到機台的，這會在某一台上種下一個具體、有記錄的訊號，好讓 context tool
  真的有東西可以找。見 `src/aoi_agent/store/seed.py`。
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
- **這個專案的十四條不變式裡，有兩條只守住一半，還有一條根本守不住。**
  `CLAUDE.md` 列了十四條不能被悄悄改掉的規則；`scripts/invariant_audit.py` 會報出
  哪幾條真的會在被違反時讓測試失敗，而 `tests/test_invariant_audit.py` 會在某一條
  失去守衛時掛掉。十一條有守。fan-out 那條和官方 split 那條各自只守住一部分，而且逐條寫明守住的是哪一部分；「說清楚哪些是
  模擬的」是散文紀律，被明確宣告為無法測試，而不是算它通過。每一格都是真的去破壞
  那條規則、跑完整套測試得出來的 ——
  [稽核結果](docs/benchmarks.md#the-invariant-audit--which-of-this-projects-own-rules-are-unguarded)。

## 還沒做的

- **認證做了，而它刻意沒做的那些沒有做。** 兩個頁面都在登入後面，人工判定寫不進去就
  是寫不進去 —— 這一項本來是擋住站台跑在筆電以外任何地方的那一項。沒做、也不打算做
  的是：TLS（cookie 是 bearer token，而這個 process 講的是明文 HTTP）、登入端點的速率
  限制或鎖定，以及任何「誰可以做什麼」的概念 —— 每個作業員都能回每個區域、都能問
  `/ask` 任何問題。最後這一項是對「複判站是什麼」的一個判斷，不是漏掉。
- **合成的那段文字沒有量。** Planner 評的是 plan，tool 又是確定性的，所以「資料是對的」
  這件事有保證。但寫在那些資料上面的那段話對不對，沒有量，而「正確數字旁邊一句貌似合
  理的錯話」正是這個專案其他時間都在防的失效模式。要評它需要一份 rubric 跟一個沒寫過
  prompt 的評分者，做法照 planner 那套。
- **從作業員更正回頭 retrain。** 判定歷史有記
  （`uv run python -m aoi_agent corrections`），而且每一列現在都寫得出是誰做的、那個
  名字是怎麼建立的，所以下一輪可以只吃 `signed_in`，或者把其他的權重壓低。還沒有東西
  去用它；改變的是這個選擇存在了 —— 在此之前每一筆人工判定都是同一個沒有分別的
  `NULL`。
- **把量化後的 model 真的接上去**，這現在是一個決定，不是一個缺口。INT8 static 量過
  了，守得住曲線，常駐記憶體少 4.8 倍；沒有接進站台是因為這台站台沒有記憶體問題。
  要接的話，需要在 `ReVerifier` 裡開一條 ONNX 路徑，並且針對真正要服務的那個 engine
  重掃一次 threshold。
- **跨 model 比較**：`gpt-oss:20b`、`qwen3:14b`、`qwen2.5:14b`。reason node 的延遲現在
  只在一個 model 上量過；更小的 model 進不進得了 explanation deadline、還寫不寫得出
  堪用的理由，沒有量。
- **板子瀏覽器**，讓 agent 自己收掉的那 82% 也看得到，而不是只看得到 queue。現在站上
  只顯示系統決定不了的東西，那是它最不完整也最不好看的一面。
- **時間戳是存 UTC、顯示 UTC，而且沒有標示。** 一份在 UTC+8 讀的品質紀錄上，那是八小
  時的謊。存 UTC、顯示當地、標清楚是哪個。
