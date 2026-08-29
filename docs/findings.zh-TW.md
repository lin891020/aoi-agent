<!-- 2026-08-28 從 README.zh-TW.md 搬出來：README 留每一項的一句話，完整版在這裡。 -->

# 量出來之後改掉的東西

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
→ [這次的 run](benchmarks.md#agent-layer--does-it-beat-the-classifier-and-is-the-escalation-calibrated)

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

  **2026-08-25：tool 後來補上了，而題庫刻意一字未動** —— 它的價值就在出題的人
  沒看過 prompt，做 tool 的人回頭改答案就把這個價值毀了。照原題庫重跑：47/70，
  裁定後 50/70，先前是 55/70。七題「該拒卻答」裡三題是題庫對
  `query_false_call_rate` 過時；**另外四題是真發現 —— 加一個 tool 讓 planner
  在這個 tool 管不到的題目上也變大膽了**，其中一題是它先前會拒絕的處置要求。
  裁定表和兩種讀法都在 benchmarks；tool 回報的比例在 payload 裡自我標註為
  複判模型的判定，不是真值。

- **第二個 tool 用同樣的方式加進來，把它為之而生的那題答了一半。** 獨立題庫問了兩次
  「同一台機器在時間上有沒有變」，而 store 沒有時間軸。`machine_events` 加上
  `query_defect_history` 的 `relative_to`/`side` 給了它一個，種法是一個有效、三個對照，
  讓 tool 有機會錯。在乾淨的機器上對著沒動過的題庫重跑：答對 28/42（原 26）、
  拒答 22/28（原 23），而目標那題規劃了事件查詢和該機台的歷史，**但沒有組出前後兩個視窗**
  —— 搆得到，還沒組起來。設計文件預測另一題時間軸問題不會被這個 tool 解掉，也確實沒有。
  前一輪跟另一個 session 掛掉的工作共用 Ollama，70 題超時 34 題，那一節留在
  `docs/benchmarks.md` 裡當它本來的樣子；腳本現在會拒絕發佈這種輪。

- **2026-08-27：站台上第一次真的問「事件前後」，被拒答了**，理由是「沒有工具能
  用事件的時間點切資料」——對規劃模型看到的東西而言是真的，對工具而言是假的。工
  具清單只給它每個工具說明書的第一行，所以 `relative_to` 和 `side` 到它手上是兩
  個沒解釋的名字；規則裡列「沒有工具能表達的維度」時還拿「前後界線」當例子，那是
  事件工具出現之前寫的；而且沒列出有哪些事件種類，所以「換燈」被錨在
  `parameter_change`。三處都修了、有測試守著，命令列真的叫模型規劃兩題，在有效
  果的機台和對照機台上都組出前後兩個視窗。**2026-08-28 在安靜的機器上重跑，題庫
  未動、零 timeout：**獨立題庫答對 28/42（不變）、拒答 21/28（原 22）、66/70 穩定；
  裁定後拒答 24/28 對原本 25/28，三題 false-call-rate 仍是題庫過時。S25——事件工具
  為之而生的那題——三次重複都組出前後兩個視窗，照原本的理由不計入分數。有一題往
  反方向走，點名：S32 原本拒答，現在對一個問「信心隨時間」的問題規劃了 false-call
  rate。其餘都在三題的漂移基線內。in-house 二十題 18/20（答對 11/13、拒答 7/7）。
- **2026-08-29，唯讀 SQL 工具，兩組對照。** 登錄表有 `run_sql`：獨立題庫答對 24/42、
  裁定後拒答 22/28；沒有它（對照組，`AOI_SQL_TOOL=0`）：26/42 與 25/28。五題原本無路
  可走的問題有了路（某台機的班別、今天標了幾區、等人看的件數）；S25 的事件組合被一句
  SELECT 取代，另外三題問題裡沒指名實體也被寫成 SELECT。in-house 現在二十二題，
  20/22 對 21/22，新加的日期題兩組都規劃正確。工具留著；針對失敗的兩句 prompt 規則
  寫在裁決裡、還沒放進去。
- **2026-08-29 下午，加了兩句規則後重跑。** 有工具 28/42 答對、裁定後拒答 23/28；對照組
  25/42（四題逾時）與 25/28。S25 回到事件工具、S20 回到拒答、五題新路都還在。兩句 SELECT
  合法但意思錯——一個不存在的狀態值、一個被當成位置的欄位——各回了一個數字；守門層現在
  會拒絕「等號右邊的值一列都沒有」；用第三次重跑的每句 SELECT 離線檢查，擋住了不存在的
  狀態值和用主鍵比板號的兩句，擋不住把序號當位置的那句。

**這不代表什麼：** 題目是 LLM 作者按不同 brief 寫的，不是真的領班寫的，所以它只框住
這些 brief 生得出來的題型。而且評的是 plan 不是文字 —— 資料對的情況下寫出來的那段
話對不對，是下面那一節。
→ [獨立那次的 run](benchmarks.md#analysis-planner-asked-by-someone-else--does-it-plan-the-right-lookups-and-refuse-the-rest)

### plan 對之後，那段文字有沒有照著資料寫，也量了。

*「我怎麼知道它什麼時候在編數字？」* Tool 是 deterministic 的，所以對「數字」而言這是
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
→ [這次的 run](benchmarks.md#the-prose-over-the-results--is-the-sentence-true-of-the-payload)

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
→ [污染率表](benchmarks.md#cross-class-contamination-in-the-criteria-retrieval)

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
是 7,322 個 candidate 裡多 289 筆 escalation，佔 queue 的 3.9%，其中 229 筆是
agent 原本的 dismissal，現在一筆都沒有了。（2026-08-26 之前這裡寫的是 8,143 裡
多 47 筆、0.6%，那是 registration 之前的 candidate 母體。）

`CONFIDENT` 則根本不是品質關卡。`confirm_node` 跟 `decide_node` 寫的是同一個 verdict，
所以在 `ESCALATE_BELOW` 以上，它改變的 disposition 是**零**，增加的 escape 也是零；
一路 sweep 到 0.999 還是零。它決定的是誰會拿到一次 LLM call 跟一段書面理由，不是板
子的下場。它唯一不能做的是掉到 `ESCALATE_BELOW` 以下 —— 那裡它會開始把流程本來要送
人工的區域直接 confirm 掉。約束才是那個 citation，裡面的值只是旋鈕。

現在每個 threshold 都引得到一支跑得起來的 script 或一份文件裡寫著那個數字的那一行，
而且有 **29 個測試**會在值跟出處對不上、出處失效、或有新 threshold 進了 code 卻沒進
表的時候紅掉。
→ [sweep](benchmarks.md#threshold-sweep--escalate_below-and-confident-2026-08-23--commit-68e90b6)
· [表](architecture.md#thresholds-and-where-they-come-from)

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
→ [重算](benchmarks.md#whole-line-escape-rate-recounted-on-defects-instead-of-boxes)

