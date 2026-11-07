# HANDOFF

最後更新：2026-11-07。

## 現在在哪一步

**封存 2026-11-07**：作品集已成形（README 就是報告：55.6% 人工複判移除、escape 0.66%，每個數字指到產生它的 run）。最後一次實質工作 2026-09-27；2026-09-25 的 Jev 拒答評估結論「不接」（F1 0.62 對 gpt-oss 0.67–0.71）。為什麼停：找工作用的作品，Mike 2026-11-07 盤點時決定封裝。今天把 `slim-claude-md`（領先 42 個 commit）fast-forward 合回 `master`，公開的 README 從 2026-09-13 的版本更新到現在。重啟先做：`開始複判站.command`（port 8110，dev.test 的 aoi 按鈕與 Caddy 的 aoi.test 保留），需要 host ollama 的 gpt-oss:20b。

還開著的題目在 `docs/still-open.md`；`DESIGN.md` 的「決定」欄空著，封存不填。

## 已驗證

- 2026-11-07：`origin/master` 是 `slim-claude-md` 的祖先，ff 合併成功並推上。8110 沒在聽。

## 只是寫了、還沒驗證

- 封存當天沒有重跑測試或起複判站。
- `docs/deck/interview-12.zh-TW.{html,pptx}` 兩個檔沒進版控（面試用的 deck，Mike 沒決定要不要進）。

## 下一步

1. 面試前要 demo 時，照上面的重啟步驟起一次。
