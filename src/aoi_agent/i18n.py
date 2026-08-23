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

        # -- a question that produced no lookup -------------------------------
        "analysis.refused.opening":
            "這個問題沒有可以查的資料，所以沒有作答。",
        "analysis.refused.capabilities":
            "這套系統可以回答的是：",

        "station.name": '複判站',
        "nav.queue": '佇列',
        "nav.corrections": '修正紀錄',
        "nav.ask": '提問',
        "nav.sign_out": '登出',
        "locale.switch": '切換語言',
        "common.region": '區域',
        "common.model_said": '模型判定',
        "common.confidence": '信心',
        "common.when_utc": '時間（UTC）',
        "common.none": '—',
        "queue.title": '等待人工判定的區域',
        "queue.waiting": '件待處理',
        "queue.sub": '每一列都是視覺模型無法定案、agent 也不願臆測的區域。最舊的在前。',
        "queue.unexplained":
            '{count} 件（共 {total} 件）沒有書面說明。每一件的處置仍是複判模型的判定、不受影響；缺的是作業員讀的那段文字。',
        "queue.empty": '沒有待處理項目。',
        "queue.empty_hint": '用下列指令餵入佇列：',
        "queue.false_call_probability": 'P(誤判)',
        "queue.line_machine": '產線 / 機台',
        "queue.why_handed_over": '為什麼交給人',
        "queue.no_explanation": '無說明',
        "queue.review": '複判 →',
        "queue.start_at_top": '從第一件開始 →',
        "queue.unattributed": '{count} 件已關閉的項目背後沒有人的判定',
        "region.lot": '批號',
        "region.line": '產線',
        "region.machine": '機台',
        "region.shift": '班別',
        "region.board_record": '板級紀錄',
        "region.waiting_on_you": '等你判定',
        "region.already_answered": '已判定',
        "region.handed_over_because": 'agent 交出來的理由',
        "region.leaned_towards": '它傾向 {verdict}，但沒有把握',
        "region.no_explanation": '沒有書面說明',
        "region.triptych_alt": '旗標區域周圍的樣板、待測與差異影像',
        "region.template": '黃金樣板',
        "region.under_test": '待測板',
        "region.difference": '差異',
        "region.what_model_read": '模型讀到什麼',
        "region.class": '類別',
        "region.box": '框',
        "region.patch_alt": '模型實際分類的 64 px 視窗',
        "region.patch_caption": '模型實際分類的 64 px 視窗——如果它偏離了區域，分歧在於裁切而不是分類器',
        "region.production_context": '產線脈絡',
        "region.lot_average": '批平均',
        "region.this_board": '這片板',
        "region.inspected": '檢測時間',
        "region.by_machine": '{defect_type} 各機台佔比，最近 {days} 天',
        "region.fleet": '全廠',
        "region.simulated_metadata": '批號 / 產線 / 機台 / 班別為模擬 metadata',
        "region.criteria_retrieved": '檢索到的驗收標準',
        "region.criteria_none": '此區域沒有檢索到標準',
        "region.criteria_simulated": '本專案自撰文件，不是 IPC-A-610',
        "region.your_verdict": '你的判定',
        "region.verdict_note":
            "你的答案會喚醒暫停中的執行，並以你的名義記為人工判定。"
            "它會成為下一輪訓練的標註——這也是這一頁不顯示 ground truth 的原因。",
        "region.no_longer_waiting": "這個區域已經不在等待中。",
        "region.answering_as": '判定人：',
        "region.next_waiting": '下一件待處理區域 →',
        "region.back_to_queue": '回到佇列 →',
        "region.path": '路徑',
        "region.timings": '耗時',
        "board.title": '板 {stem}',
        "board.disposition": '板級處置',
        "board.sub":
            "這片板被判定了什麼，以及在什麼條件下判定的。一整批退回來時稽核員"
            "會問的問題，都在這一頁：誰放行的、什麼時候、當時生效的是哪個模型"
            "和哪組門檻。",
        "board.none_yet": "尚未記錄板級處置。",
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
        "board.not_dispositioned": '尚未有板級處置——仍有區域等待判定',
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
        "analysis.title": '問一個關於產線的問題',
        "analysis.ask": '問',
        "analysis.placeholder": "例如：L2-M22 的 open 是不是不尋常？",
        "analysis.s4_timings":
            "4 · {tools} 個工具 · 平行 {wall}ms（最長的單一分支 {longest}ms）/ "
            "依序 {sequential}ms · 規劃 {plan}ms · 撰寫 {synthesise}ms",
        "analysis.try": '試試看：',
        "analysis.coverage":
            '資料涵蓋最近 {days} 天。超出這個範圍的問題會被拒答，而不是用最接近的窗口代答。產線 metadata 是模擬的；驗收標準是本專案自撰文件，不是 IPC-A-610。',
        "analysis.s1": '1 · 它怎麼理解你的問題',
        "analysis.s2": '2 · 它呼叫了什麼',
        "analysis.s3": '3 · 它假設了什麼',
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
        "tool.query_board_context": "單板脈絡",
        "tool.search_standards": "驗收標準檢索",
        "tool.list_candidates": "單板區域清單",
    },
    "en": {
        # -- charts ---------------------------------------------------------
        "chart.title.defects_by_class": "Defects by class",
        "chart.title.share_by_machine": "Defect share by machine",
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

        # -- a question that produced no lookup -------------------------------
        "analysis.refused.opening":
            "No lookup in this system answers that, so nothing was run.",
        "analysis.refused.capabilities":
            "What it can answer:",

        "station.name": 're-verification station',
        "nav.queue": 'queue',
        "nav.corrections": 'corrections',
        "nav.ask": 'ask',
        "nav.sign_out": 'sign out',
        "locale.switch": 'switch language',
        "common.region": 'region',
        "common.model_said": 'model said',
        "common.confidence": 'conf',
        "common.when_utc": 'when (UTC)',
        "common.none": '—',
        "queue.title": 'Regions waiting for a person',
        "queue.waiting": 'waiting',
        "queue.sub":
            'Every row is a region the vision model could not settle and the agent declined to guess at. Oldest first.',
        "queue.unexplained":
            "{count} of {total} carry no written explanation. The disposition on each is the re-verification model's and is unaffected; what is missing is the paragraph the operator reads.",
        "queue.empty": 'Nothing waiting.',
        "queue.empty_hint": 'Feed the queue with',
        "queue.false_call_probability": 'P(false call)',
        "queue.line_machine": 'line / machine',
        "queue.why_handed_over": 'why it was handed over',
        "queue.no_explanation": 'no explanation',
        "queue.review": 'review →',
        "queue.start_at_top": 'Start at the top →',
        "queue.unattributed": '{count} closed entries carry no human decision',
        "region.lot": 'lot',
        "region.line": 'line',
        "region.machine": 'machine',
        "region.shift": 'shift',
        "region.board_record": 'board record',
        "region.waiting_on_you": 'waiting on you',
        "region.already_answered": 'already answered',
        "region.handed_over_because": 'the agent handed this over because',
        "region.leaned_towards": 'it leaned towards {verdict}, without confidence',
        "region.no_explanation": 'no written explanation',
        "region.triptych_alt":
            'template, test and difference around the flagged region',
        "region.template": 'golden template',
        "region.under_test": 'board under test',
        "region.difference": 'difference',
        "region.what_model_read": 'What the model read',
        "region.class": 'class',
        "region.box": 'box',
        "region.patch_alt": 'the 64 px window the model classified',
        "region.patch_caption":
            'the 64 px window the model actually classified — if it is off the region, the disagreement is the crop, not the classifier',
        "region.production_context": 'Production context',
        "region.lot_average": 'lot average',
        "region.this_board": 'this board',
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
        "board.title": 'board {stem}',
        "board.disposition": 'Board disposition',
        "board.sub":
            "What was decided about this board, and under what. One page for "
            "the question an auditor asks after a batch comes back: who "
            "released it, when, and which model and thresholds were in force "
            "when they did.",
        "board.none_yet": "No board-level disposition recorded yet.",
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
        "analysis.title": 'Ask a question about the line',
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
        "analysis.s2": '2 · What it called',
        "analysis.s3": '3 · What it assumed',
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
        "tool.query_board_context": "Board context",
        "tool.search_standards": "Criteria retrieval",
        "tool.list_candidates": "Regions on a board",
    },
}

#: The locales a caller may ask for. Derived from the tables rather than
#: declared beside them, so adding a language cannot leave a list behind.
LOCALES: tuple[str, ...] = tuple(STRINGS)


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
