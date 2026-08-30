"""The interface's text, in the two languages it is read in.

The line this is built for reads Traditional Chinese; the acceptance criteria
and the benchmark reports are written in English. Both are true at once and the
station had been half of each -- `/ask` in Chinese, the queue and the board
record in English -- which is worse than either.

**Language is a rendering, not a record.** What this module translates is
chrome: headings, column names, buttons, axis labels. It never touches the
question a supervisor typed, and it never touches what the planning call wrote
about that question -- those are what happened, and a record rewritten into
another language is a record of something nobody did. The one piece of prose
that does follow the language is the synthesised answer, and it follows by
being *written again* from the stored results, down the same measured path,
never by being translated. See `analysis/service.py`.

Top level rather than under `station/` because the flow reaches it too: a
refusal is written where the plan is, and `station/` already depends on
`analysis/`.

A key missing from one table is a bug the suite catches --
`tests/test_i18n.py` compares the two key sets and fails on any difference,
which is the only thing standing between "changed the Chinese" and "forgot the
English". At runtime a missing key renders as the key itself: visible and ugly,
rather than a 500 on a page whose figures are all still correct.
"""

from __future__ import annotations

#: What the shop floor reads. English is the second language here, not the
#: first, and the default says so.
DEFAULT_LOCALE = "zh-TW"

#: Where the choice is kept. A cookie and not the signed session: how a
#: person reads the screen is not a claim about who they are, it needs no
#: integrity guarantee, and on a shared terminal it should outlive a sign-out.
LOCALE_COOKIE = "aoi_locale"

STRINGS: dict[str, dict[str, str]] = {
    "zh-TW": {
        # -- charts ---------------------------------------------------------
        # Identifiers are not translated. `mousebite` is what the store calls
        # that class and what the work instruction calls it; a chart axis
        # showing 鼠咬 and a payload showing `mousebite` are two vocabularies
        # for a reader to reconcile in their head.
        "chart.title.defects_by_class": "各類缺陷數量",
        "chart.title.share_by_machine": "各機台缺陷佔比",
        "chart.title.false_call_rate": "誤判駁回率（複判模型的判定，非真值）",
        "chart.title.open_share_around_event": "事件前後的 open 佔比（複判模型的判定，非真值；區間重疊代表未證明有差）",
        "chart.axis.side_of_event": "事件前 / 後",
        "chart.axis.open_share": "被判為 open 的旗標區域比例",
        "chart.series.machine_around_event": "機台 {machine_id}，{kind} 前後",
        "chart.axis.group": "分組",
        "chart.axis.dismissal_rate": "被駁回為誤判的比例",
        "chart.series.dismissal_rate_by": "依{group_by}分組",
        "chart.axis.defect_class": "缺陷類別",
        "chart.axis.count": "數量",
        "chart.axis.machine": "機台",
        "chart.axis.share_of_own_defects": "佔該機台缺陷的比例",
        "chart.series.line_id": "產線 {line_id}",
        "chart.series.machine_id": "機台 {machine_id}",
        "chart.series.lot_id": "批號 {lot_id}",
        "chart.series.everything": "全部",
        "chart.series.share_of": "{defect_type} 佔比",
        "chart.series.fleet_average": "全廠平均",
        "chart.series.defects_per_board": "每片缺陷數",
        "chart.axis.defects_per_board": "每片缺陷數",
        "chart.title.defects_per_board_by_machine": "各機台每片缺陷數",
        "queue.more": "展開全文",

        # -- a question that produced no lookup -------------------------------
        "analysis.refused.opening":
            "這個問題沒有可以查的資料，所以沒有作答。",
        "analysis.refused.capabilities":
            "這套系統可以回答的是：",
        # -- the page asked what it can be asked ------------------------------
        "analysis.capabilities.opening":
            "這一頁能回答的是下面這幾類問題。問題會先變成一份查詢計畫，驗證過才執行；"
            "它不改任何資料、不做預測，也不判斷原因。",
        "analysis.capabilities.interpretation":
            "問的是這一頁能查什麼。這由工具登錄表回答，不需要規劃。",
        "analysis.can": "這一頁能回答什麼",
        "analysis.claims.title": "數字核對：{count} 處與回傳資料對不上，請對照第 2 節的資料再讀這段。",
        "analysis.claims.fabricated_figure": "回傳的資料裡沒有這個數字",
        "analysis.claims.misattributed_figure": "這個數字在資料裡屬於另一台機台",
        # -- the stage table under the answer ----------------------------------
        "analysis.timing.title": "6 · 花了多久",
        "analysis.timing.stage": "階段",
        "analysis.timing.wall": "等待",
        "analysis.timing.model": "其中模型推論",
        "analysis.timing.plan": "規劃（模型讀題、決定查什麼）",
        "analysis.timing.tools": "查詢（工具，平行）",
        "analysis.timing.chart": "繪圖（由結果推導）",
        "analysis.timing.synthesise": "撰寫回答（模型）",
        "analysis.timing.total": "合計",
        "analysis.timing.seconds": "{s} 秒",
        "analysis.timing.unrecorded": "未記錄",
        "analysis.timing.note":
            "「等待」是頁面實際等的時間，含排隊與載入模型；「模型推論」是 Ollama 回報的 "
            "eval_duration。兩者差很多，通常是機器同時在忙別的事。",
        "analysis.cannot":
            "它不會改任何資料、不做預測、不判斷原因；沒有資料的日期不存在，也不會拿鄰近的日期代替。",
        "tool.query_defect_history.does":
            "某條線、某台機、某個批號或某幾天的缺陷數量，按類別；也能取某個機台事件的前後兩段。",
        "tool.query_machine_stats.does":
            "每台機台的缺陷率排名：指定一類看佔比，不指定看每片缺陷數；可以指定日期、只取前 N 名。",
        "tool.query_false_call_rate.does":
            "AOI 標出來、被本系統駁回為誤判的比例，按機台、產線或班別。",
        "tool.query_machine_events.does":
            "某台機台發生過的事：參數變更、保養，什麼時候。",
        "tool.query_board_context.does":
            "一片 PCB 的來歷：批號、產線、機台、班別，以及同批的狀況。",
        "tool.search_standards.does":
            "某一類缺陷的驗收標準寫了什麼。",
        "tool.list_candidates.does":
            "一片 PCB 上 AOI 標出的區域，以及模型對每一區的判定。",
        "tool.run_sql.does":
            "上面的工具都不涵蓋的維度（班別、批號對機台、駁回件數…）用一句 SELECT 查；"
            "只讀一份沒有答案欄的唯讀副本，最多 200 列，SQL 會印在結果旁邊。",
        "analysis.sql_rows": "回傳 {count} 列（顯示 {shown} 列）",
        "analysis.sql_truncated": "已達 200 列上限，結果被截斷",
        "analysis.sql_hidden": "另外 {count} 列未顯示",
        "chart.title.sql_rows": "查詢結果",
        "chart.series.sql_column": "{column}",
        "chart.axis.sql_column": "{column}",
        "chart.axis.sql_value": "值",

        "station.name": '複判站',
        "nav.queue": '待複判',
        "nav.corrections": '修正紀錄',
        "nav.ask": '產線查詢',
        "nav.sign_out": '登出',
        "locale.switch": '切換語言',
        "common.region": '區域',
        "common.model_said": '模型判定',
        "common.confidence": '信心',
        "common.when_utc": '時間（UTC）',
        "common.none": '—',
        "queue.title": '待人工複判的區域',
        "queue.waiting": '件待處理',
        "queue.sub": '每一列都是視覺模型無法定案、agent 也不願臆測的區域。最舊的在前。',
        "queue.unexplained":
            '{count} 件（共 {total} 件）沒有書面說明。每一件的處置仍是複判模型的判定、不受影響；缺的是作業員讀的那段文字。',
        # A page that shows part of a list has to say so. Before 2026-08-25 the
        # queue rendered its first 200 rows and reported that as the total.
        "queue.truncated":
            '顯示最舊的 {shown} 件，共 {total} 件在等 —— 還有 {hidden} 件不在這一頁上。先把上面的清掉。',
        "corrections.truncated":
            '顯示最近的 {shown} 筆，共 {total} 筆。下方的彙總涵蓋全部 {total} 筆，不只是這一頁。',
        # The eighth option, and the only one that is not an answer.
        "region.defer": '我不確定',
        "region.defer_hint": '看不出來就按這個。它不會寫成判定，區域會留在待判清單上換下一個人看。',
        "region.defer_note_label": '你看不出來的是什麼？（可留白）',
        "region.defer_note_placeholder": '例：缺口和鍍層分不出來',
        "region.declined_before": '已經有 {count} 個人看過這一區並表示無法判斷：',
        "region.declined_by": '{operator} 於 {when} UTC',
        "region.declined_no_note": '（未說明原因）',
        "deferred.title": '待資深複判的區域',
        "deferred.sub": '每一列都是至少一個人看過、並且說看不出來的區域。被越多人退回的排越前面 —— 那是這裡唯一帶資訊的排序。',
        "deferred.no_senior": '目前沒有任何人被設為 senior，所以這一頁上的區域沒有人可以回答。用 scripts/add_operator.py --role senior 指定一個。',
        "deferred.needs_senior": '你的權限是 operator，看得到但不能回答這裡的區域 —— 這些是別人已經判不出來的。',
        "deferred.may_answer": '你的權限是 senior，可以回答這裡的區域。',
        "deferred.empty": '目前沒有被退回的區域。',
        "deferred.declines": '幾個人退回',
        "deferred.no_routing_note":
            '這一頁只是清單，不是派工。誰答得了寫在下面一行，由憑證檔決定；'
            '但這裡沒有任何東西會把某一區指給某個人，也沒有人會收到通知。'
            '要有人來清這份清單，得靠交接，不靠這個站台。',
        "nav.boards": 'PCB 處置',
        "nav.deferred": '待資深複判',
        "queue.deferred_link": '另外有 {count} 區待資深複判 →',
        "queue.empty": '沒有待處理項目。',
        "queue.empty_hint": '用下列指令餵入佇列：',
        "queue.false_call_probability": 'P(誤判)',
        "queue.line_machine": '產線 / 機台',
        "queue.why_handed_over": '為什麼交給人',
        "queue.waiting_since": '等候起點（UTC）',
        "queue.no_explanation": '無說明',
        "queue.unsourced_figures": '{count} 個數字不在證據裡',
        "queue.rationale_language":
            '「為什麼交給人」以{lang}撰寫（環境變數 AOI_LINE_LANGUAGE，切換語言不會改寫既有紀錄）。'
            '判定不等這段文字，2.5 ms 就出來了；文字本身中文一則約 17 秒、英文約 12 秒（2026-08-30 量測），'
            '只影響一片 PCB 全部跑完的時間。',
        "lang.name.zh-TW": "中文",
        "lang.name.en": "英文",
        "queue.review": '複判 →',
        "queue.start_at_top": '從第一件開始 →',
        "queue.unattributed": '{count} 件已關閉的項目背後沒有人的判定',
        "region.lot": '批號',
        "region.line": '產線',
        "region.machine": '機台',
        "region.shift": '班別',
        "region.board_record": 'PCB 紀錄',
        "region.waiting_on_you": '等你判定',
        "region.already_answered": '已判定',
        "region.handed_over_because": 'agent 交出來的理由',
        "region.leaned_towards": '它傾向 {verdict}，但沒有把握',
        "region.no_explanation": '沒有書面說明',
        "region.unsourced_figures": '說明裡有 {count} 個數字，模型看到的證據裡沒有',
        "region.unsourced_figures_note": '這些數字不在分類結果、產線資料或任何準則段落裡。判定不受影響；讀說明時把它們當成未經證實。',
        "region.triptych_alt": '旗標區域周圍的樣板、待測與差異影像',
        "region.template": '黃金樣板',
        "region.under_test": '待測 PCB',
        "region.difference": '差異',
        "region.what_model_read": '模型讀到什麼',
        "region.class": '類別',
        "region.box": '框',
        "region.patch_alt": '模型實際分類的 64 px 視窗',
        "region.patch_caption": '模型實際分類的 64 px 視窗——如果它偏離了區域，分歧在於裁切而不是分類器',
        "region.production_context": '產線脈絡',
        "region.lot_average": '批平均',
        "region.this_board": '本片 PCB',
        "region.inspected": '檢測時間',
        "region.by_machine": '{defect_type} 各機台佔比，最近 {days} 天',
        "region.fleet": '全廠',
        "region.simulated_metadata": '批號 / 產線 / 機台 / 班別為模擬 metadata',
        "region.criteria_retrieved": '檢索到的驗收標準',
        "region.criteria_none": '此區域沒有檢索到標準',
        "region.criteria_simulated": '本專案自撰文件，不是 IPC-A-610',
        "region.your_verdict": '你的判定',
        "region.measure_reset": "重新量測",
        "region.measure.ask_reference": "點兩下，量出基準：",
        "region.measure.ask_measured": "再點兩下，量出實際值：",
        "region.measure.within": "{ratio}% — 在允收範圍內（需 ≥{limit}%）",
        "region.measure.outside": "{ratio}% — 超出允收範圍（需 ≥{limit}%）",
        "region.measure.incomparable":
            "兩段量在不同格：那是拿樣板的長度去比待測 PCB的長度，不能相除。",
        "region.verdict_note":
            "你的答案會喚醒暫停中的執行，並以你的名義記為人工判定。"
            "它會成為下一輪訓練的標註——這也是這一頁不顯示 ground truth 的原因。",
        "region.no_longer_waiting": "這個區域已經不在等待中。",
        "region.answering_as": '判定人：',
        "region.next_waiting": '下一件待處理區域 →',
        "region.back_to_queue": '回到佇列 →',
        "region.path": '路徑',
        "region.timings": '耗時',
        "boards.title": 'PCB 處置紀錄',
        "boards.sub":
            "佇列上的是 agent 無法定案的區域，也就是失敗的那一小塊。這一頁是另一邊："
            "已經有PCB 處置的每一片 PCB，往哪一邊倒，以及底下有多少區域支撐它。"
            "只讀——這裡不判定任何東西，只是已經做過的判定的紀錄。",
        "boards.all": '已定案 {count}',
        "boards.held": '扣住 {count}',
        "boards.released": '放行 {count}',
        "boards.waiting": '等待中 {count}',
        "boards.waiting_chip": '等待人工',
        "boards.truncated":
            "顯示最新 {shown} 片，共 {total} 片；另外 {hidden} 片不在這一頁上。"
            "上面的計數是對整張表數的，不是對這一頁數的。",
        "boards.empty": '尚無 PCB 處置紀錄。',
        "boards.empty_hint": '執行一片 PCB 即會產生：',
        "boards.board": 'PCB 編號',
        "boards.disposition": '處置',
        "boards.regions": '旗標區域',
        "boards.confirmed": '確認為缺陷',
        "boards.still_waiting": '等待人工',
        "boards.open": '看紀錄 →',
        "board.title": 'PCB {stem}',
        "board.disposition": 'PCB 處置',
        "board.sub":
            "本片 PCB被判定了什麼，以及在什麼條件下判定的。一整批退回來時稽核員"
            "會問的問題，都在這一頁：誰放行的、什麼時候、當時生效的是哪個模型"
            "和哪組門檻。",
        "board.none_yet": "尚未記錄 PCB 處置。",
        "board.live_basis":
            "{count} 個旗標區域：{confirmed} 個確認為缺陷、{pending} 個仍等待人工、"
            "{dismissed} 個已排除",
        "board.handed_back": '退回，等待資深',
        "board.absences_note":
            "模型欄顯示 unrecorded，代表這筆判定寫在這個 store 有 provenance "
            "欄位之前；unavailable 是寫在之後、但仍無法指出權重的。兩種缺席不同，"
            "而且刻意分開——兩者都不是 null，也都不是 digest。這一頁同樣不顯示 "
            "ground truth，理由和佇列頁一樣。",
        "board.authority": '判定者',
        "board.basis": '依據',
        "board.model": '模型',
        "board.thresholds": '門檻',
        "board.code": '程式版本',
        "board.every_region": '每個區域，以及決定它的是什麼',
        "board.verdict": '判定',
        "board.by": '由',
        "board.waiting_on_person": '等待人工',
        "board.absence_unrecorded": 'unrecorded：此列早於這些欄位存在',
        "board.absence_unavailable": 'unavailable：寫入時仍無法指出模型',
        "board.absence_null": 'null：不應出現，出現即為缺陷',
        "board.regions_count": '{count} 個旗標區域',
        "board.not_dispositioned": '尚未有 PCB 處置——仍有區域待複判',
        "corrections.title": '作業員推翻模型的紀錄',
        "corrections.sub":
            "每一列都是 agent 交出來、由人判定過的區域。標記為「推翻」的，"
            "就是下一輪訓練的修正——一個模型的類別和人的判定不一致的區域，"
            "價值高過一百個它判對的。",
        "corrections.none_yet": "尚未記錄任何人工判定。",
        "corrections.none_yet_hint": "去回答一件，它就會出現在這裡：",
        "corrections.overruled_chip": "推翻",
        "corrections.ground_truth_note":
            "這一頁同樣不顯示 ground truth。作業員自己判得對不對，是評估腳本的"
            "問題，不是一個離佇列只有一個連結的頁面該回答的。",
        "corrections.human_decisions": '人工判定',
        "corrections.overruled": '推翻模型',
        "corrections.overrule_rate": '推翻率',
        "corrections.by_model_class": '依模型判定分',
        "corrections.reviewed": '已複判',
        "corrections.corrected_to": '修正為',
        "corrections.every_decision": '每一筆判定，最新在前',
        "corrections.person_said": '人判定',
        "corrections.reviewer": '判定人',
        "corrections.attribution": '身分依據',
        "corrections.agreed": '一致',
        "login.title": '登入',
        "login.attribution_note":
            "你的名字會出現在你判定的每一個區域上，而那些答案是下一輪訓練的標註。"
            "這裡要的是這個——不是權限，是歸屬。",
        "login.ephemeral_sessions":
            "AOI_AGENT_SESSION_SECRET 未設定，session 以本行程產生的金鑰簽章，"
            "重啟後所有人都會被登出。筆電上無妨；要長期執行請設定它。",
        "login.name": '姓名',
        "login.passphrase": '通行碼',
        "login.submit": '登入',
        "login.no_operators": '尚未設定任何作業員。用下列指令新增：',
        "login.secret_hint": 'session 簽章金鑰來自 AOI_AGENT_SESSION_SECRET。',
        "login.failed": '姓名或通行碼錯誤。',
        "analysis.title": '產線查詢',
        "analysis.ask": '問',
        "analysis.placeholder": "例如：L2-M22 的 open 是不是不尋常？",
        "analysis.s4_timings":
            "4 · {tools} 個工具 · 平行 {wall}ms（最長的單一分支 {longest}ms）/ "
            "依序 {sequential}ms · 規劃 {plan}ms · 撰寫 {synthesise}ms",
        "analysis.try": '試試看：',
        "analysis.coverage":
            '資料涵蓋最近 {days} 天。超出這個範圍的問題會被拒答，而不是用最接近的窗口代答。產線 metadata 是模擬的；驗收標準是本專案自撰文件，不是 IPC-A-610。',
        "analysis.s1": '1 · 它怎麼理解你的問題',
        "analysis.as_asked":
            "以提問時的語言記錄",
        "analysis.as_asked_title":
            "這一段是規劃那次呼叫寫下的。規劃不會重跑，所以它是當時發生的事的"
            "紀錄，不會跟著語言切換重寫。",
        "analysis.s2": '3 · 它呼叫了什麼',
        "analysis.s3": '2 · 它假設了什麼',
        "analysis.s5": '5 · 回答',
        "analysis.tool": '工具',
        "analysis.args": '參數',
        "analysis.why": '為什麼',
        "analysis.elapsed": '耗時',
        "analysis.failed": '失敗',
        "analysis.no_data": '查無資料',
        "analysis.tool_reported": '工具回報：',
        "analysis.returned_data": '回傳的資料（{count} 項）',
        "analysis.recent": '最近問過的',
        "analysis.refused_chip": '拒答',
        # -- the live progress panel, read by `static/flow.js` ---------------
        # Same table as the rest of the page. The browser gets these as JSON
        # rather than a second set of translations kept in step by hand.
        "flow.stage.plan": "規劃",
        "flow.stage.fan": "獨立查詢",
        "flow.stage.join": "匯整",
        "flow.stage.write": "撰寫回答",
        "flow.phase.planning": "規劃中…",
        "flow.phase.running": "查詢中…（{done}/{total} 完成）",
        "flow.phase.synthesising": "撰寫回答中…",
        "flow.phase.refused": "沒有可執行的查詢",
        "flow.phase.failed": "已中止",
        "flow.phase.done": "完成",
        "flow.branches.unplanned": "尚未規劃",
        "flow.branches.none": "沒有查詢",
        "flow.elapsed": "{seconds} 秒",
        "flow.since": "，已 {seconds} 秒",
        "flow.disconnected": "連線中斷",
        # -- what a tool is called on screen ---------------------------------
        # The readable name is beside the registry name, never instead of it:
        # the registry name is what the plan actually called and what the
        # validator checked the signature of. Replacing it swaps something
        # auditable for decoration.
        "tool.query_defect_history": "缺陷歷史",
        "tool.query_machine_stats": "機台比較",
        "tool.query_false_call_rate": "誤判駁回率",
        "tool.query_machine_events": '機台事件紀錄',
        "tool.query_board_context": "PCB 脈絡",
        "tool.search_standards": "驗收標準檢索",
        "tool.list_candidates": "PCB 區域清單",
        "tool.run_sql": "唯讀 SQL 查詢",
    },
    "en": {
        # -- charts ---------------------------------------------------------
        "chart.title.defects_by_class": "Defects by class",
        "chart.title.share_by_machine": "Defect share by machine",
        "chart.title.false_call_rate": "False-call dismissal rate (the re-verifier's judgement, not ground truth)",
        "chart.title.open_share_around_event": "Open share before and after the event (the re-verifier's judgement, not ground truth; overlapping intervals mean no difference shown)",
        "chart.axis.side_of_event": "Before / after the event",
        "chart.axis.open_share": "Share of flagged regions classified open",
        "chart.series.machine_around_event": "Machine {machine_id}, around {kind}",
        "chart.axis.group": "Group",
        "chart.axis.dismissal_rate": "Share dismissed as false calls",
        "chart.series.dismissal_rate_by": "grouped by {group_by}",
        "chart.axis.defect_class": "class",
        "chart.axis.count": "count",
        "chart.axis.machine": "machine",
        "chart.axis.share_of_own_defects": "share of that machine's defects",
        "chart.series.line_id": "Line {line_id}",
        "chart.series.machine_id": "Machine {machine_id}",
        "chart.series.lot_id": "Lot {lot_id}",
        "chart.series.everything": "All",
        "chart.series.share_of": "share of {defect_type}",
        "chart.series.fleet_average": "fleet average",
        "chart.series.defects_per_board": "defects per board",
        "chart.axis.defects_per_board": "defects per board",
        "chart.title.defects_per_board_by_machine": "defects per board, by machine",
        "queue.more": "show all",

        # -- a question that produced no lookup -------------------------------
        "analysis.refused.opening":
            "No lookup in this system answers that, so nothing was run.",
        "analysis.refused.capabilities":
            "What it can answer:",
        # -- the page asked what it can be asked ------------------------------
        "analysis.capabilities.opening":
            "This page answers the kinds of question below. A question becomes a "
            "plan of lookups, which is validated before it runs; nothing here "
            "changes data, forecasts, or establishes cause.",
        "analysis.capabilities.interpretation":
            "A question about what can be asked here. The registry answers it; "
            "nothing is planned.",
        "analysis.can": "What this page can answer",
        "analysis.claims.title": "Figure check: {count} figure(s) do not match the returned data; read this against the data in section 2.",
        "analysis.claims.fabricated_figure": "no returned value renders as this figure",
        "analysis.claims.misattributed_figure": "in the data this figure belongs to another machine",
        # -- the stage table under the answer ----------------------------------
        "analysis.timing.title": "6 · How long it took",
        "analysis.timing.stage": "stage",
        "analysis.timing.wall": "waited",
        "analysis.timing.model": "of which model inference",
        "analysis.timing.plan": "planning (the model reads the question, decides what to look up)",
        "analysis.timing.tools": "lookups (tools, in parallel)",
        "analysis.timing.chart": "chart (derived from the results)",
        "analysis.timing.synthesise": "writing the answer (model)",
        "analysis.timing.total": "total",
        "analysis.timing.seconds": "{s} s",
        "analysis.timing.unrecorded": "unrecorded",
        "analysis.timing.note":
            "\"Waited\" is what the page actually waited, queueing and model "
            "loading included; \"model inference\" is the eval_duration Ollama "
            "reported. A wide gap between them usually means the machine was "
            "busy with something else.",
        "analysis.cannot":
            "It changes nothing, forecasts nothing and establishes no cause; a day "
            "with no data does not exist here and is not replaced with a nearby one.",
        "tool.query_defect_history.does":
            "Defect counts by class for a line, a machine, a lot or a span of "
            "days; also the two windows either side of a machine event.",
        "tool.query_machine_stats.does":
            "Every machine ranked by defect rate: the share of one class when a "
            "class is named, defects per board when none is; by date, cut to the top N.",
        "tool.query_false_call_rate.does":
            "How much of what the AOI flags this system dismisses as a false "
            "call, by machine, line or shift.",
        "tool.query_machine_events.does":
            "What has happened to a machine -- parameter changes, maintenance -- "
            "and when.",
        "tool.query_board_context.does":
            "Where one PCB was made: lot, line, machine, shift, and how the lot "
            "is doing.",
        "tool.search_standards.does":
            "What the acceptance criteria say about one defect class.",
        "tool.list_candidates.does":
            "The regions the AOI flagged on one PCB, and what the model called each of them.",
        "tool.run_sql.does":
            "One SELECT for a dimension none of the tools above take (a shift, "
            "the machines of a lot, dismissed counts); over a read-only copy with "
            "no answer column, at most 200 rows, the SQL printed beside the result.",
        "analysis.sql_rows": "{count} rows returned ({shown} shown)",
        "analysis.sql_truncated": "cut at the 200-row cap",
        "analysis.sql_hidden": "{count} more rows not shown",
        "chart.title.sql_rows": "query result",
        "chart.series.sql_column": "{column}",
        "chart.axis.sql_column": "{column}",
        "chart.axis.sql_value": "value",

        "station.name": 're-verification station',
        "nav.queue": 'Review queue',
        "nav.corrections": 'Corrections',
        "nav.ask": 'Line analytics',
        "nav.sign_out": 'Sign out',
        "locale.switch": 'switch language',
        "common.region": 'region',
        "common.model_said": 'model said',
        "common.confidence": 'conf',
        "common.when_utc": 'when (UTC)',
        "common.none": '—',
        "queue.title": 'Regions awaiting review',
        "queue.waiting": 'waiting',
        "queue.sub":
            'Every row is a region the vision model could not settle and the agent declined to guess at. Oldest first.',
        "queue.unexplained":
            "{count} of {total} carry no written explanation. The disposition on each is the re-verification model's and is unaffected; what is missing is the paragraph the operator reads.",
        "queue.truncated":
            "Showing the {shown} oldest of {total} waiting — {hidden} are not on this page. Clear the ones above first.",
        "corrections.truncated":
            "Showing the {shown} most recent of {total}. The summary below covers all {total}, not just this page.",
        "region.defer": "I can't tell",
        "region.defer_hint": "Press this if you cannot see it. It is not recorded as a verdict; the region stays on the list for someone else and you move on.",
        "region.defer_note_label": "What could you not tell? (optional)",
        "region.defer_note_placeholder": "e.g. cannot separate the notch from the plating",
        "region.declined_before": "{count} people have looked at this region and said they could not judge it:",
        "region.declined_by": "{operator} at {when} UTC",
        "region.declined_no_note": "(no reason given)",
        "deferred.title": "Regions awaiting senior review",
        "deferred.sub": "Every row is a region at least one person looked at and could not read. The ones the most people handed back are first — that is the only ranking here that carries information.",
        "deferred.no_senior": "Nobody is configured as senior, so nothing on this page can be answered by anyone. Give one operator --role senior with scripts/add_operator.py.",
        "deferred.needs_senior": "You are an operator: you can see these but not answer them — they are the ones somebody else already could not read.",
        "deferred.may_answer": "You are a senior and can answer these.",
        "deferred.empty": "Nothing has been handed back.",
        "deferred.declines": "handed back by",
        "deferred.no_routing_note":
            "This page is a list, not an assignment. Who may answer is the "
            "line below, and it comes from the credential file; but nothing "
            "here hands a particular region to a particular person and nobody "
            "is notified. Emptying this list happens at handover, not here.",
        "nav.boards": 'PCB dispositions',
        "nav.deferred": "Senior review",
        "queue.deferred_link": "{count} more awaiting senior review →",
        "queue.empty": 'Nothing waiting.',
        "queue.empty_hint": 'Feed the queue with',
        "queue.false_call_probability": 'P(false call)',
        "queue.line_machine": 'line / machine',
        "queue.why_handed_over": 'why it was handed over',
        "queue.waiting_since": 'Waiting since (UTC)',
        "queue.no_explanation": 'no explanation',
        "queue.unsourced_figures": '{count} figure(s) not in the evidence',
        "queue.rationale_language":
            'The "why it was handed over" text is written in {lang} (AOI_LINE_LANGUAGE; '
            'the switch never rewrites a stored one). The verdict does not wait on this text '
            '-- it is decided in 2.5 ms; the text itself takes about 17 s per region in Chinese '
            'and 12 s in English (measured 2026-08-30); the only thing that moves is how long '
            'a whole board takes to finish.',
        "lang.name.zh-TW": "Chinese",
        "lang.name.en": "English",
        "queue.review": 'review →',
        "queue.start_at_top": 'Start at the top →',
        "queue.unattributed": '{count} closed entries carry no human decision',
        "region.lot": 'lot',
        "region.line": 'line',
        "region.machine": 'machine',
        "region.shift": 'shift',
        "region.board_record": 'PCB record',
        "region.waiting_on_you": 'waiting on you',
        "region.already_answered": 'already answered',
        "region.handed_over_because": 'the agent handed this over because',
        "region.leaned_towards": 'it leaned towards {verdict}, without confidence',
        "region.no_explanation": 'no written explanation',
        "region.unsourced_figures": 'The explanation cites {count} figure(s) the model was never shown',
        "region.unsourced_figures_note": 'None of these appears in the classifier reading, the production context or any retrieved criterion. The disposition is unaffected; read them as unverified.',
        "region.triptych_alt":
            'template, test and difference around the flagged region',
        "region.template": 'golden template',
        "region.under_test": 'PCB under test',
        "region.difference": 'difference',
        "region.what_model_read": 'What the model read',
        "region.class": 'class',
        "region.box": 'box',
        "region.patch_alt": 'the 64 px window the model classified',
        "region.patch_caption":
            'the 64 px window the model actually classified — if it is off the region, the disagreement is the crop, not the classifier',
        "region.production_context": 'Production context',
        "region.lot_average": 'lot average',
        "region.this_board": 'this PCB',
        "region.inspected": 'inspected',
        "region.by_machine": '{defect_type} by machine, last {days} days',
        "region.fleet": 'fleet',
        "region.simulated_metadata":
            'lot / line / machine / shift are simulated metadata',
        "region.criteria_retrieved": 'Acceptance criteria retrieved',
        "region.criteria_none": 'none retrieved for this region',
        "region.criteria_simulated":
            'original documents written for this project, not IPC-A-610',
        "region.your_verdict": 'Your verdict',
        "region.measure_reset": "Measure again",
        "region.measure.ask_reference": "Click twice for the reference:",
        "region.measure.ask_measured": "Now click twice for the measurement:",
        "region.measure.within": "{ratio}% — within limits (needs ≥{limit}%)",
        "region.measure.outside": "{ratio}% — outside limits (needs ≥{limit}%)",
        "region.measure.incomparable":
            "The two segments are in different panels: that divides a length on "
            "the template by a length on the board under test.",
        "region.verdict_note":
            "Your answer resumes the suspended run and is recorded as a human "
            "decision under your name. It becomes a label in the next training "
            "round, which is why the ground truth is not shown on this page.",
        "region.no_longer_waiting": "This region is no longer waiting.",
        "region.answering_as": 'answering as',
        "region.next_waiting": 'Next waiting region →',
        "region.back_to_queue": 'Back to the queue →',
        "region.path": 'path',
        "region.timings": 'timings',
        "boards.title": 'PCB dispositions',
        "boards.sub":
            "The queue holds the regions the agent could not settle -- the "
            "small part that failed. This page is the other side: every board "
            "with a standing disposition, which way it went, and how many "
            "regions are behind that. Read-only -- nothing here dispositions "
            "anything, it is the record of dispositions already made.",
        "boards.all": 'dispositioned {count}',
        "boards.held": 'held {count}',
        "boards.released": 'released {count}',
        "boards.waiting": 'waiting {count}',
        "boards.waiting_chip": 'waiting on a person',
        "boards.truncated":
            "Showing the {shown} most recent of {total}; {hidden} are not on "
            "this page. The counts above are taken over the whole table, not "
            "over this page.",
        "boards.empty": 'No PCB has been dispositioned yet.',
        "boards.empty_hint": 'Run one PCB to populate this page:',
        "boards.board": 'PCB',
        "boards.disposition": 'disposition',
        "boards.regions": 'flagged',
        "boards.confirmed": 'confirmed',
        "boards.still_waiting": 'waiting on a person',
        "boards.open": 'record →',
        "board.title": 'PCB {stem}',
        "board.disposition": 'PCB disposition',
        "board.sub":
            "What was decided about this board, and under what. One page for "
            "the question an auditor asks after a batch comes back: who "
            "released it, when, and which model and thresholds were in force "
            "when they did.",
        "board.none_yet": "No PCB-level disposition recorded yet.",
        "board.live_basis":
            "{count} flagged regions: {confirmed} confirmed as defects, "
            "{pending} still waiting on a person, {dismissed} dismissed",
        "board.handed_back": 'handed back, waiting on a senior',
        "board.absences_note":
            "A model column reading unrecorded is a decision written before "
            "this store had provenance columns; unavailable is one written "
            "after, whose weights still could not be named. They are two "
            "different absences and are kept apart on purpose -- neither is "
            "null, and neither is a digest. The ground truth is not on this "
            "page either, for the same reason it is not on the queue.",
        "board.authority": 'authority',
        "board.basis": 'basis',
        "board.model": 'model',
        "board.thresholds": 'thresholds',
        "board.code": 'code',
        "board.every_region": 'Every region, and what decided it',
        "board.verdict": 'verdict',
        "board.by": 'by',
        "board.waiting_on_person": 'waiting on a person',
        "board.absence_unrecorded": 'unrecorded: the row predates these columns',
        "board.absence_unavailable":
            'unavailable: written after, and the model still could not be named',
        "board.absence_null": 'null: should not appear, and is a defect if it does',
        "board.regions_count": '{count} flagged regions',
        "board.not_dispositioned":
            'not dispositioned — regions are still waiting on a person',
        "corrections.title": 'Where operators overruled the model',
        "corrections.sub":
            "Every row is a region a person judged after the agent handed it "
            "over. The ones marked overruled are the next training round's "
            "corrections -- a region where the model's class and a person's "
            "verdict disagree is worth more than a hundred it got right.",
        "corrections.none_yet": "No human decisions recorded yet.",
        "corrections.none_yet_hint": "Answer something on the",
        "corrections.overruled_chip": "overruled",
        "corrections.ground_truth_note":
            "The ground truth is not shown here either. Whether the operators "
            "were themselves right is a question for the evaluation scripts, "
            "not for a page one link away from the queue.",
        "corrections.human_decisions": 'human decisions',
        "corrections.overruled": 'overruled the model',
        "corrections.overrule_rate": 'overrule rate',
        "corrections.by_model_class": 'By what the model said',
        "corrections.reviewed": 'reviewed',
        "corrections.corrected_to": 'corrected to',
        "corrections.every_decision": 'Every decision, newest first',
        "corrections.person_said": 'person said',
        "corrections.reviewer": 'reviewer',
        "corrections.attribution": 'attribution',
        "corrections.agreed": 'agreed',
        "login.title": 'Sign in',
        "login.attribution_note":
            "Your name goes on every region you answer, and those answers are "
            "the next training round's labels. That is what this asks for -- "
            "not a permission, an attribution.",
        "login.ephemeral_sessions":
            "AOI_AGENT_SESSION_SECRET is unset, so sessions are signed with a "
            "key generated for this process and everyone is signed out when it "
            "restarts. Fine on a laptop; set it for anything longer-lived.",
        "login.name": 'name',
        "login.passphrase": 'passphrase',
        "login.submit": 'sign in',
        "login.no_operators": 'No operators are configured. Add one with',
        "login.secret_hint":
            'The session signing key comes from AOI_AGENT_SESSION_SECRET.',
        "login.failed": 'That name and passphrase do not match.',
        "analysis.title": 'Line analytics',
        "analysis.ask": 'Ask',
        "analysis.placeholder": "e.g. is L2-M22's open rate unusual?",
        "analysis.s4_timings":
            "4 · {tools} tools · in parallel {wall}ms (longest single branch "
            "{longest}ms) / sequentially {sequential}ms · planning {plan}ms · "
            "writing {synthesise}ms",
        "analysis.try": 'Try:',
        "analysis.coverage":
            'The data covers the last {days} days. A question outside that span is refused rather than answered from the nearest window. Production metadata is simulated; the acceptance criteria are documents written for this project, not IPC-A-610.',
        "analysis.s1": '1 · How it read your question',
        "analysis.as_asked":
            "recorded in the language it was asked in",
        "analysis.as_asked_title":
            "This section is what the planning call wrote. The planning call "
            "is not made again, so it is a record of what happened and is not "
            "rewritten when the language changes.",
        "analysis.s2": '3 · What it called',
        "analysis.s3": '2 · What it assumed',
        "analysis.s5": '5 · Answer',
        "analysis.tool": 'tool',
        "analysis.args": 'arguments',
        "analysis.why": 'why',
        "analysis.elapsed": 'elapsed',
        "analysis.failed": 'failed',
        "analysis.no_data": 'no data',
        "analysis.tool_reported": 'the tool reported:',
        "analysis.returned_data": 'data returned ({count} fields)',
        "analysis.recent": 'Asked recently',
        "analysis.refused_chip": 'refused',
        # -- the live progress panel, read by `static/flow.js` ---------------
        "flow.stage.plan": "Planning",
        "flow.stage.fan": "Independent lookups",
        "flow.stage.join": "Join",
        "flow.stage.write": "Writing the answer",
        "flow.phase.planning": "Planning…",
        "flow.phase.running": "Looking up… ({done}/{total} done)",
        "flow.phase.synthesising": "Writing the answer…",
        "flow.phase.refused": "No lookup to run",
        "flow.phase.failed": "Stopped",
        "flow.phase.done": "Done",
        "flow.branches.unplanned": "Not planned yet",
        "flow.branches.none": "No lookups",
        "flow.elapsed": "{seconds}s",
        "flow.since": ", {seconds}s in",
        "flow.disconnected": "Connection lost",
        # -- what a tool is called on screen ---------------------------------
        "tool.query_defect_history": "Defect history",
        "tool.query_machine_stats": "Machine comparison",
        "tool.query_false_call_rate": "False-call dismissal rate",
        "tool.query_machine_events": 'Machine events',
        "tool.query_board_context": "Board context",
        "tool.search_standards": "Criteria retrieval",
        "tool.list_candidates": "Regions on a board",
        "tool.run_sql": "Read-only SQL",
    },
}

#: The locales a caller may ask for. Derived from the tables rather than
#: declared beside them, so adding a language cannot leave a list behind.
LOCALES: tuple[str, ...] = tuple(STRINGS)


#: The language the line reads in, which is what the disposition path's
#: explanation is written in. A deployment property rather than a per-request
#: one: the rationale is written once, when the region is assessed, by a CLI
#: run or by the station, and it is then a record -- the queue shows it in the
#: language it was written in, whichever way the switch is set. Unset, the
#: station's default language.
LINE_LANGUAGE_ENV = "AOI_LINE_LANGUAGE"


def line_language() -> str:
    import os

    return normalise(os.environ.get(LINE_LANGUAGE_ENV) or DEFAULT_LOCALE)


#: Appended to a prompt that produces prose a person reads. One sentence, and
#: only about the language: everything else in those prompts is a constraint
#: that has been measured, and re-wording a measured constraint invalidates
#: the measurement it was taken under. Identifiers stay as the data spells
#: them, in both languages, so a rationale and the record it is stored
#: against name the same class and the same document.
LANGUAGE_NOTE = {
    "zh-TW": "Write all prose you produce in Traditional Chinese (繁體中文). "
             "Leave identifiers -- defect classes, line, machine and lot ids, "
             "document numbers, tool names -- exactly as they appear in the "
             "data.",
    "en": "Write all prose you produce in English. Leave identifiers -- defect "
          "classes, line, machine and lot ids, document numbers, tool names -- "
          "exactly as they appear in the data.",
}


def normalise(locale: str | None) -> str:
    """The locale to use for a value that may have come from a cookie."""
    return locale if locale in STRINGS else DEFAULT_LOCALE


def translate(key: str, locale: str | None = None, /, **args: object) -> str:
    """One string, in one language.

    Falls back to the key rather than raising. A page whose figures are all
    correct should not 500 over a heading, and the key on screen names the
    thing to fix. Bad arguments fall back the same way: a template that renders
    `{line_id}` literally is a visible fault, an exception is a lost page.
    """
    table = STRINGS[normalise(locale)]
    template = table.get(key)
    if template is None:
        return key
    try:
        return template.format(**args)
    except (KeyError, IndexError):
        return template


def label_from(spec: dict, field: str, locale: str | None = None) -> str:
    """A chart specification's label, whichever way it was stored.

    ``field`` is the bare name -- ``title``, ``y_label``, ``name``. A spec
    written after 2026-08-23 carries ``<field>_key`` and optional
    ``<field>_args`` and is translated. One written before carries the rendered
    English sentence and is shown as it is: the language it was drawn in is not
    recoverable from it, and inventing one would be worse than the gap. That
    absence is named by `tests/test_chart_svg.py` rather than left for a reader
    to find in a redrawn chart from last quarter.
    """
    key = spec.get(f"{field}_key")
    if key:
        return translate(key, locale, **(spec.get(f"{field}_args") or {}))
    return str(spec.get(field, "") or "")
