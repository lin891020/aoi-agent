#!/bin/bash
# 雙擊即可啟動複判站。也是 dev.test 上「啟動」按鈕呼叫的對象。
# 關掉這個終端機視窗就會停止。
cd "$(dirname "$0")" || exit 1

URL="http://aoi.test"

# 已經在跑就直接開瀏覽器，不要啟動第二個（否則會撞 port 8110）
if curl -s -o /dev/null --max-time 1 http://127.0.0.1:8110; then
  echo "複判站已在執行中。"
  [ "$1" != "--no-open" ] && open "$URL"
  exit 0
fi

echo "啟動中…"
uv run python -m aoi_agent station &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT

for _ in $(seq 1 40); do
  curl -s -o /dev/null --max-time 1 http://127.0.0.1:8110 && break
  sleep 0.5
done

if ! curl -s -o /dev/null --max-time 1 http://127.0.0.1:8110; then
  echo "啟動失敗，錯誤訊息在上方。"
  read -r -p "按 Enter 關閉…"
  exit 1
fi

[ "$1" != "--no-open" ] && open "$URL"
echo
echo "複判站：$URL"
echo "關掉這個視窗即停止。"
wait $SERVER
