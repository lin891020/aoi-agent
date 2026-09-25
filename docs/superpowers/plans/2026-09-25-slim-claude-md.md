# CLAUDE.md 從 1,082 行瘦到 60 行以內

狀態：待 Mike 核准（改這一行）。核准後在 aoi-agent 自己的 session 執行。

## 使用者要的是什麼

2026-09-25 審查整套 Claude 設定：「有沒有效能過剩或浪費的」。
量出來最大的一筆是這個檔：78 KB，每次在 aoi-agent 開 session 都整份載入，
粗估兩萬 token。根目錄 CLAUDE.md 定的上限是 60 行。

## 現在的 1,082 行是什麼

| 段落 | 行數 | 性質 | 去處 |
|---|---|---|---|
| 開頭 | 10 | 專案一句話 | 留 |
| Layout | 71 | 目錄樹加註解 | `docs/architecture.md`（已 527 行，先查重疊） |
| Commands | 91 | 60 多條指令 | `docs/commands.md`；CLAUDE.md 留 5 條最常用 |
| Invariants | 338 | 不可默默改的規則，**有測試在解析它** | `docs/invariants.md`，原文不動 |
| Environment gotchas | 21 | 這台機器的陷阱（ollama 搶 GPU、`ollama ps` 不是 GPU 檢查） | 留，這是最值錢的 21 行 |
| Still open | 551 | 2026-08-27 到 09-25 的工作日誌 | `docs/still-open.md`；另開 `HANDOFF.md` 照 handoff skill 四節 |

## 可能會錯在哪（都可以測）

1. `scripts/invariant_audit.py` 第 63、64 行：`CLAUDE_MD = ROOT / "CLAUDE.md"`、
   `INVARIANTS_HEADING = "## Invariants"`。搬走後常數要指到 `docs/invariants.md`，
   標題字串不能改。驗證：`uv run python scripts/invariant_audit.py --collect` 零 drift，
   `tests/test_invariant_audit.py` 綠。
2. `tests/test_published_figures.py` 第 38 行 `PUBLISHED` 與 `tests/test_deck.py` 第 30–37 行
   `PUBLISHED` 都列了 CLAUDE.md。數字搬到 `docs/invariants.md` 後要把新檔加進兩個清單，
   不然檢查靜悄悄地失去覆蓋。驗證：故意把 invariants.md 裡一個數字改錯，測試要紅。
3. `.claude/agents/code-reviewer.md` 第 13 行叫 reviewer 讀 CLAUDE.md 的 Invariants 段。改指向。
4. `docs/benchmarks.md`、`docs/architecture.md`、幾份 spec 的散文提到「CLAUDE.md 的 invariant」。
   那是歷史敘述，不改；新檔開頭寫一行「2026-09-25 之前住在 CLAUDE.md」。
5. Still open 是日誌不是待辦：最新一筆是今天。搬走前先確認今天那個 session 已經收工，
   不要跟它撞。

## 別人怎麼解的

Claude Code 官方建議 CLAUDE.md 短、細節放檔案用 `@path` 或連結引用；
HumanLayer 的 12-factor agents 也是同一句話：context 是預算，不是倉庫。
沒查到有人把「測試解析 CLAUDE.md」當設計的；這裡的做法是保留那個機制，只換檔名。

## 步驟

1. 在 `claude-tooling` 分支開 `slim-claude-md`；先 stash 那張 png。
2. 四個 `git mv` 等價的搬移（用 sed 切段落，原文 byte 不動）。
3. 改上面 1–3 的三處路徑。
4. 寫新 CLAUDE.md：開頭 + 五條指令 + Environment gotchas + 四個連結 + 「先呼叫專案 skill」那段。目標 ≤ 60 行。
5. `uv run pytest -q -p no:warnings`（不含 dataset）全綠；`invariant_audit.py --collect` 零 drift；`wc -l CLAUDE.md` ≤ 60。
6. Mike 看 diff 再合。
