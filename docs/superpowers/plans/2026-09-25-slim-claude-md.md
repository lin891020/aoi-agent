# CLAUDE.md 從 1,082 行瘦到 80 行以內

狀態：Mike 2026-09-25 核准（第二版），在 aoi-agent-15 執行。
2026-09-25 第二版：與 Mike 審過，加入七點修正（見文末「第二版改了什麼」）。

## 使用者要的是什麼

2026-09-25 審查整套 Claude 設定：「有沒有效能過剩或浪費的」。
量出來最大的一筆是這個檔：78,788 bytes，每次在 aoi-agent 開 session 都整份載入，
粗估兩萬 token。根目錄 CLAUDE.md 定的上限是 60 行。

**上限的例外（Mike 2026-09-25：「超過 60 行應該也沒關係吧」）。** 20 條 invariant
的粗體標題每條一行留在 CLAUDE.md，總長允許到 80 行。理由：invariant 的用處是
「每個 session 都看得到」；搬到 docs/ 只留連結，就只剩有測試的那幾條還守得住，
「LLM 只解釋不決定」「fan-out 不是加速」這類沒有測試抓的規則會從視野裡消失。
新 CLAUDE.md 開頭要寫一行說明這個例外，免得之後的 session 把它當違規砍掉。

## 現在的 1,082 行是什麼

| 段落 | 行數 | 性質 | 去處 |
|---|---|---|---|
| 開頭 | 10 | 專案一句話 | 留 |
| Layout | 74 | 目錄樹加註解（含另一 session 未 commit 的 `docs/experiments/` 3 行） | `docs/architecture.md`（已 527 行，先查重疊） |
| Commands | 91 | 60 多條指令 | `docs/commands.md`；CLAUDE.md 留 3 條最常用 |
| Invariants | 338 | 不可默默改的規則，**有測試在解析它** | 全文 `docs/invariants.md`，原文不動；CLAUDE.md 留 20 行標題索引 |
| Environment gotchas | 21 | 這台機器的陷阱（ollama 搶 GPU、`ollama ps` 不是 GPU 檢查） | 留，這是最值錢的 21 行 |
| Still open | 551 | 2026-08-27 到 09-25 的工作日誌 | `docs/still-open.md`，原文不動 |

## 可能會錯在哪（都可以測）

1. `scripts/invariant_audit.py` 第 63、64 行：`CLAUDE_MD = ROOT / "CLAUDE.md"`、
   `INVARIANTS_HEADING = "## Invariants"`。搬走後常數要指到 `docs/invariants.md`，
   標題字串不能改。**新 CLAUDE.md 的標題索引不可用 `## Invariants` 開頭**
   （例如用 `## The twenty rules`），否則之後有人把常數改回來會讀到只有標題的版本
   卻照樣通過一半。驗證：`uv run python scripts/invariant_audit.py --collect` 零 drift，
   `tests/test_invariant_audit.py` 綠。
2. **published 清單兩個新檔都要加，不只 invariants.md。**
   `tests/test_published_figures.py` 第 38 行與 `tests/test_deck.py` 第 31–38 行的
   `PUBLISHED` 都列了 CLAUDE.md。被取代的數字大多住在 **Still open**
   （"read 52.8% … until"、"It read … until 2026-09-08"），不只 Invariants；
   只加 invariants.md，still-open.md 的 551 行就沒人管「被取代的數字要附日期」。
   兩個清單都加 `docs/invariants.md` 與 `docs/still-open.md`；
   `test_deck.py` 的清單另加 `docs/commands.md`（指令註解裡有數字，例如 ~48 min）。
3. **`test_deck.py:57` 是 `if p.exists()`**：新路徑打錯字時靜悄悄跳過，測試照綠。
   驗證要做到每個新檔、兩個測試各一次：把該檔一個數字改錯（或刪掉日期），測試要紅，
   改回來要綠。四次紅都看到才算過。
4. `.claude/agents/code-reviewer.md` 第 13 行叫 reviewer 讀 CLAUDE.md 的 Invariants 段。
   改指向 `docs/invariants.md`。
5. **連結只用一般 Markdown 連結，不用 `@path`。** `@docs/invariants.md` 會把整份檔
   在開 session 時載入，等於沒瘦。驗證：`grep -n '@docs\|@CLAUDE' CLAUDE.md` 無結果。
6. `docs/benchmarks.md`、`docs/architecture.md`、幾份 spec，以及
   `src/aoi_agent/station/app.py`、多個測試的 docstring 提到「CLAUDE.md 的 invariant」。
   那是歷史敘述，不改；新檔開頭寫一行「2026-09-25 之前住在 CLAUDE.md」。
7. 搬移會改到位元組：驗證是把各段接回去與舊檔逐段 `diff`，必須完全相同
   （見步驟 5）。
8. 另一個 session 還在這個 repo 工作：工作區有它未 commit 的 CLAUDE.md +3 行、
   `docs/experiments/`、`docs/deck/img/screens.png`。開工前它要先 commit 完，
   否則瘦身會把那 3 行覆蓋掉或把它的檔案捲進這個分支。

## 別人怎麼解的

Claude Code 官方建議 CLAUDE.md 短、細節放別的檔案用連結引用（`@path` 是匯入，
會載入全文，這裡刻意不用）；HumanLayer 的 12-factor agents 也是同一句話：
context 是預算，不是倉庫。沒查到有人把「測試解析 CLAUDE.md」當設計的；
這裡的做法是保留那個機制，只換檔名。

## 步驟

1. 確認另一個 session 已把 `docs/experiments/`、CLAUDE.md 的 3 行、screens.png commit
   （`git status` 乾淨）。沒有就停，不 stash 別人的東西。
2. 從 `claude-tooling` 開 `slim-claude-md`。
3. 切段落寫到新檔（sed 按 `## ` 標題切，原文位元組不動）：
   Layout 併進 `docs/architecture.md`、Commands → `docs/commands.md`、
   Invariants → `docs/invariants.md`、Still open → `docs/still-open.md`。
   每個新檔開頭加一行出處（不算在逐段比對內）。
4. 改路徑：可能出錯第 1、2、4 點的三處程式與清單。
5. 寫新 CLAUDE.md：開頭 + 例外說明一行 + 3 條指令 + 20 行 invariant 標題索引（連結
   `docs/invariants.md`）+ Environment gotchas + 四個連結 + 「先呼叫專案 skill」那段。
   目標 ≤ 80 行。
6. 驗證：
   - `uv run pytest -q -p no:warnings`（不含 dataset）全綠
   - `invariant_audit.py --collect` 零 drift
   - 逐段比對：每個新檔去掉出處行後，與舊 CLAUDE.md 對應段 `diff` 無輸出
   - 可能出錯第 3 點的四次紅燈
   - `grep '@docs\|@CLAUDE' CLAUDE.md` 無結果；`wc -l CLAUDE.md` ≤ 80
7. Mike 看 diff 再合。

不在這次範圍：把 Still open 整理成 handoff skill 的 `HANDOFF.md` 四節。那要判斷
內容，不是搬移，另開一件。

## 第二版改了什麼

與 Mike 2026-09-25 審計畫時加入：invariant 標題留在 CLAUDE.md（上限放寬到 80）；
still-open.md 也進 published 清單；`exists()` 靜默跳過要用紅燈驗證；不用 `@path`；
等另一個 session commit 而不是 stash；HANDOFF 拆出去；補逐段位元組比對。
