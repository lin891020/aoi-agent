# aoi-agent（複判站） 的設計合約

這份是合約，不是靈感板：**寫數值，不寫形容詞**。Claude 做任何頁面之前先讀這份；
它的優先序在所有設計 skill 之上（Anthropic：你的設計系統 > 內建選擇；你的 prompt > 兩者）。
決定改了就改這裡，不要只改 CSS。

- 「現況」欄是 `claude-config/scripts/design_snapshot.py` 從 CSS/HTML **靜態抽出**的，不是瀏覽器量的；
  真正的量測用 `polishing-pages/scripts/measure.mjs` 與 `sim-users/scripts/axe.mjs`。
- 「決定」欄是 Mike 看過樣品之後填的。`TODO（Mike）` 代表還沒決定，Claude 不要自己填。
- 「依據」指 `claude-config/docs/research/reports/有質感介面怎麼設計.md` 的章節。

| 欄位 | 現況（靜態抽出） | 決定 | 依據 |
|---|---|---|---|
| 這頁是誰用、做什麼 | — | TODO（Mike）一句話 | 第五節 |
| 工具或文宣 | — | TODO（Mike）二選一；決定階比例 1.2 或 1.333 | 第一節 |
| 字體 | — | TODO（Mike）西文一個、中文一個；載入哪幾個字重；不要合成粗體 | 第一節繁中段 |
| 字級 | `0.85rem` ×7、`0.72rem` ×6、`0.82rem` ×6、`0.8rem` ×5、`0.9rem` ×5、`0.88rem` ×4、`0.78rem` ×4、`0.75rem` ×2、`0.95rem` ×2、`0.92em` ×1（另 9 種） | TODO（Mike）內文 17px；一頁最多 4–5 種；input ≥16px | 第一節 |
| 行高與行長 | `2` ×1、`1.4` ×1、`1.9` ×1、`1.6` ×1、`1.75` ×1、`1.7` ×1 | TODO（Mike）中文 1.6–1.8；每行 ≤40 字；桌機 max-width 700–800px | 第一節繁中段 |
| 字重 | `600` ×7、`500` ×5、`400` ×2 | 只用 400–700 | 第一節 |
| 顏色 | `#26262c` ×4、`#2e2e35` ×3、`#f59e0b` ×3、`#8b8b96` ×2、`#60a5fa` ×2、`#34d399` ×2（另 21 種）；共 27 種 | TODO（Mike）底色、文字色、一個強調色；強調色只給可以點的東西；每個顏色寫功能 | 第一節、第三節 |
| 圓角 | `10px` ×7、`8px` ×6、`4px` ×4、`3px` ×4、`6px` ×4、`999px` ×1（另 3 種） | TODO（Mike）最多 3 種，寫數值 | 第一節 |
| 邊框與陰影 | — | TODO（Mike）邊框 1px 淡灰或不用；陰影層數與光源方向 | 第一節 |
| 動效 | `0.1s` ×2、`0.7s` ×1、`1.8s` ×1 | 互動 ≤200ms、只動 transform / opacity；高頻動作不加 | 第一節 |
| 間距基底 | — | TODO（Mike）4 或 8 | 第一節 |
| 中英間距 | — | TODO（Mike）手動半形空格或 pangu.js | 第一節繁中段 |
| 不要 | P1 K4 Colored side-stripe on cards；P1 K7 No visible keyboard focus；P1 SD4 Template chrome: tracked ALL-CAPS labels；P1 SD5 One word accented in the headline | 從第三節清單挑這個專案最容易中的 5 條 | 第三節 |
| 參考對象 | — | TODO（Mike）一個網址 + 保留／改掉／不抄 | 第五節 |
| 三態 | — | TODO（Mike）空、錯、載入各一句：標題陳述缺什麼、描述一句、按鈕呼應標題 | 第二節 |

現況欄抽自：`src/aoi_agent/station/templates`、`src/aoi_agent/station/static/style.css`（13 個檔案，2026-10-24）。

## 刪掉的理由

Mike 看樣品時刪掉的東西和他講的理由，照原話記。這一節就是「不要」欄的來源。

- （例）「不要橢圓按鈕」 — 2026-10-06 keelung
