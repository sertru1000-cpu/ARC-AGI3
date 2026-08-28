"""Prompt templates for the analyzer agent."""

from inference.utils.grid_utils import ARC_COLOR_LEGEND

TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)

GAME_OVERVIEW_ADDENDUM = (
    "\n\nGame overview:\n"
    "- You are solving a multi-level grid puzzle game. \n"
    "- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle: re-understand the current state from the newest frame, update your working world model in Python, choose the next best action or short sequence against the goal as currently understood, execute it, and expect to re-evaluate on the next turn from the updated state.\n"
    "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
    "- On every LEVEL-UP, expect the level to be structurally harder, not just relocated. Entity and goal counts often multiply (one pair -> three pairs, one opponent -> three -- re-count them on the new board). Dependency chains also tend to get longer, so a wrong assumption made early can silently break something several steps later without an obvious signal at the point of the mistake. The rule you just verified with predict() may not be the rule that applies now -- re-verify it, and re-derive which region of the board is invariant/static, before trusting a plan or theory carried over from the previous level.\n"
    "- Optimize for as few in-game actions as possible while still being reliable.\n"
    "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

VISUAL_GAME_ADDENDUM = (
    "\n\nVisual-game guidance:\n"
    "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
    "- Game entities are usually be rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometime they might also be 1x1 tokens."
    "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
    "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
    "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
)

STRUCTURED_RUNTIME_STATE_ADDENDUM = (
    "\n\nRuntime variables inside every `python` tool call:\n"
    "- `current_frame` is a lightweight frame view for the latest environment state.\n"
    "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n"
    "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores its position -- equal hashes mean the same object regardless of where it is, so use it to track an object across frames or to spot multiple identical objects in one frame), `pixels` (cell count), `boundary` (clockwise outer-perimeter corner points as `[row, col]`), and `children` (ids of objects fully enclosed by this one).\n"
    "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
    
    "- `current_frame.step` is the current environment step count.\n"
    "- `current_frame.level` is the current level number.\n"
    "- `current_frame.shape` is a `(rows, cols)` tuple.\n"
    "- The raw numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object hashes. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n"

    "- `history` is a chronological list of action/frame snapshots.\n"
    "- `history` is a Python list of objects, not a dict.\n"
    "- Each history entry exposes only `.action` and `.frame`; entries are not subscriptable like `entry['action']`.\n"
    "- Each `history[i].frame` is the frame after `history[i].action`; each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n"
    "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
    "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
    "- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n"
    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`.\n"
    "- `last_transition` is `transitions[-1]` or `None`. Its `.result` mirrors `last_action_result`; older transitions may have an empty `.result`. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n"
    "- `last_action_result` is the persisted result dict from the most recent `action(...)` call. It remains available across later Python inspection calls that do not call `action(...)`, and is `{}` before any action result exists. Read transition metadata from fields/keys such as `last_action_result['board_changed']`, `last_action_result['done']`, `last_action_result['level_completed']`, `last_action_result['game_over']`, `last_action_result['run_complete']`, `last_action_result['reward']`, and `last_action_result['valid_actions']`.\n"
    "- `valid_actions` is the current list of valid action names.\n"
    "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
    "- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
    "- One action usually returns one frame, but a single action can result in a short multi-frame animation. `current_frame` is always the final frame of that animation.\n"
    "- When an action animated, `last_action_result['animation']` describes it: `frames` (how many frames came back), `unique_frames`, `board_unchanged` (the final board is identical to the board before the action), and `transient_pixels` plus `transient_bbox` for cells that only ever appear mid-animation.\n"
    "- `animation['board_unchanged']` together with `board_changed == False` does NOT mean the action did nothing. It means the effect -- a rejected click, a consumed attempt, a bounce off a wall -- was shown only in frames you are not looking at. Treat it as a real outcome and read the transient region.\n"
    "- When there is no `animation` key, the action returned a single frame and nothing was hidden from you.\n"
    "- `animation()` returns a compact diff timeline of the frames the last animated action produced: one entry per distinct frame with `changed`, `bbox`, and either `changes` (cells as `'W>R @ (row,col) (row,col)'`) or a `transitions` census when too many cells changed. Identical consecutive frames are collapsed into `held_for_frames`.\n"
    "- `animation(frame=k)` reads frame `k` verbatim, cropped to the transient region; pass `region=(top, left, bottom, right)` to choose the crop yourself. Full frames are never returned -- one 64x64 board alone exceeds the tool response budget.\n"
    "- `animation(action_num=n)` looks back at an earlier animated action instead of the most recent one. `animation()` costs no in-game action and no action budget.\n"
    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_result` are refreshed.\n"
)

MULTIMODAL_CONTEXT_ADDENDUM = (
    "\n\nMultimodal context:\n"
    "- User turns include an attached image of the current ARC grid.\n"
    "- The image and `current_frame.ascii` are two representations of the same current frame.\n"
    "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
)

PYTHON_ADDENDUM = (
    "\n\nPython tool guidance:\n"
    "- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object hashes.\n"
    "- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n"
    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n"
    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n"
    "- The only tool is `python`; call it with one ephemeral `code` string.\n"
    "- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
    "- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `history[-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n"
    "- Maintain a compact working world model: what entities or regions exist, what actions seem to do, what the goal likely is, what remains uncertain, and what plan best fits the evidence so far.\n"
    "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
    "- `verify_theory(predict, actions=None, extract=None, transitions=None)` tests a `predict(grid, action) -> next_grid` function you write against every recorded real transition, for free (zero environment actions). `grid` and the returned prediction are plain lists of rows of int color values (same values as `current_frame.grid`/`last_transition.before_frame.grid`); `action` is the exact string seen on `transitions[i].action` (e.g. `'UP'`, `'MOUSE(row=4, col=7)'`). It returns `{'accuracy': ..., 'transitions_tested': ..., 'counterexamples': [...]}` -- refine `predict()` against the counterexamples rather than guessing blind. Pass `transitions=[...]` (a hand-picked sublist of the `transitions` global -- e.g. clean forward/reverse probes you just ran) to test against exactly those instead of the full, possibly noisy history.\n"
    "- Whole-board pixel-perfect prediction is often the WRONG bar: decorative motion, HUD elements, or rendering noise unrelated to the actual mechanic can make a pixel-perfect predict() impossible even when you understand the puzzle completely. When that happens, pass `extract(grid) -> state`, a small function that reduces the raw grid to the handful of facts that actually matter (e.g. `{'player': (r, c), 'keys': [...]}` or a coarse tile grid) -- `verify_theory`/`plan_with_theory` then compare/search in THAT abstraction instead of raw pixels: `predict(state, action) -> next_state` and `goal(state) -> bool` take/return whatever `extract()` returns, not full grids. This is the difference between the best plays we've seen and getting stuck unable to clear the 0.6 accuracy bar: abstract to a small discrete state FIRST, then reason over that, rather than fighting the raw pixels. If `extract()` returns a dict, `predict()` only has to get the KEYS IT ACTUALLY PREDICTS right -- e.g. `{'player': (r, c)}` is checked as a subset of the real extracted state, so noise your `extract()` also happens to pick up never counts against you. An empty dict never counts as a match (a predict() that predicts nothing is not a theory).\n"
    "- Once `predict()` scores at least 0.6 accuracy, `plan_with_theory(predict, goal, actions=None, extract=None, max_depth=6, force=False, transitions=None)` searches for an action sequence entirely in simulation -- also zero environment actions. `goal(grid) -> bool` describes what you are aiming at (or `goal(state) -> bool` if you passed `extract`). It refuses to plan below 0.6 accuracy (planning on an unverified theory is guessing with extra steps) and returns `{'plan': [...] | None, ...}`; when a plan is found it is a list of specs `action(...)` accepts verbatim, so the usual next line is `action(res['plan'])`. `MOUSE` is excluded from the default candidate set (64x64 targets would blow up branching) -- pass explicit `{'action': 'MOUSE', 'row': r, 'col': c}` specs in `actions` to plan toward a click. Pass the SAME `extract` you verified with; the search then runs over your compact state space instead of full grids, which is usually both faster and the only way an unreliable-to-render mechanic becomes searchable at all. If verify_theory keeps scoring low because of noise your theory correctly judges irrelevant (not because your theory of the MECHANIC is wrong), pass `force=True` to search anyway -- do not use it to skip refining a theory you have not actually checked against the counterexamples.\n"
    "- CAUTION on multi-step plans: `verify_theory` only checks single, already-observed transitions -- `plan_with_theory` then CHAINS `predict()` across states it only ever imagined, never actually visited. If the mechanic saturates, collides, or otherwise changes after a few moves, that composition can be wrong even at high verified_accuracy (seen live: a 7-step plan verified at 1.0 accuracy still failed partway through because the pushed object stopped moving). When `res['note']` is non-null (plans with more than one step), use `execute_plan(res['plan'], predict, extract=extract, goal=goal)` instead of `action(res['plan'])` directly.\n"
    "- `execute_plan(plan, predict, stop_on_mismatch=True, extract=None, goal=None)` runs a `plan_with_theory()` plan one REAL step at a time, comparing each real outcome to what `predict()` forecast for that exact step, and stops itself the moment they diverge -- instead of firing every step in one `action()` call and only finding out at the very end whether it worked. Pass the same `extract` used to build the plan so the comparison happens in that abstraction, not raw grids (dicts matched as a subset, same rule as verify_theory). Pass the same `goal` used to build the plan and it is checked BEFORE any mismatch abort -- if the plan already reached it, a merely cosmetic divergence elsewhere does not cost you the win (`stop_reason='goal_reached'`). Also stops on a terminal result (`level_completed`/`done`/`game_over`/`run_complete`) or if a step failed to execute. Returns `{'steps_executed': ..., 'stopped_early': ..., 'stop_reason': 'predicted_state_mismatch' | 'action_not_executed' | 'goal_reached' | None, 'last_action_result': {...}}`. Costs the same real actions as executing the plan yourself, just stops earlier if the theory turns out to be wrong partway through -- this is the safe default for any plan with more than one step.\n"
    "- This is a genuine alternative to writing your own BFS by hand: once you trust a `predict()`, `plan_with_theory` searches over its predicted states for you and hands back a ready-to-execute plan.\n"
    "- `memo` is a dict (starts empty) that survives across every `python` call for the rest of this game -- unlike everything else here, it is NOT reset each call. Use it to remember things you would otherwise have to re-derive: `memo['tried_targets'] = memo.get('tried_targets', []) + [(row, col)]`. Only JSON-safe values survive (dicts, lists, strings, numbers, booleans, null); anything else is silently stringified when read back next turn, so keep it small and simple.\n"
    "- For alignment/positioning puzzles (move an object to a target, dock two shapes, navigate to a spot), persist the CONFIRMED position in `memo` once you have it (e.g. `memo['anchor'] = {'row': r, 'col': c, 'step': current_frame.step}`), rather than re-deriving absolute coordinates from the image from scratch every turn. Re-deriving from scratch each turn is a known failure mode -- small per-turn read errors compound over many turns into a position that never actually converges on the target, even when the mechanic itself is well understood. Use `segmentation['nodes'][i]['hash']` to re-identify the SAME tracked object across frames regardless of where it moved (equal hash = same object), then update the anchor from the confirmed diff (`last_transition.before_frame` vs `after_frame`) instead of reading the whole board fresh.\n"
    "- `save_checkpoint(label)` records a named point you can return to later: it snapshots the real game state, `memo`, and where you are in the conversation, and returns a `checkpoint_id` string. The host ALSO auto-creates one at game start (`sys_start`) and at every level-up (`sys_level_N`) -- these are announced each turn, so you always have somewhere safe to return to even if you never call this yourself.\n"
    "- `rollback(checkpoint_id, lesson_learned)` reverts the real game state to that checkpoint AND erases the conversation since then -- everything after it is gone from your context. `lesson_learned` is the ONE thing that survives: a short summary of what you were trying and why it failed, injected as a message from your past self once the rollback lands. Use this when you recognize you are in a dead end (repeating a failed idea, or the board has looped back to a state you already tried), instead of continuing to reason from context that the failed attempt has already corrupted.\n"
    "- `try_actions(sequences)` SPECULATIVELY executes one or more action sequences on the REAL engine, then rewinds -- zero recorded actions, the real game does not move. Pass a list of sequences (each a list of specs exactly like `action()` takes) or one flat sequence; returns one outcome summary per sequence (what executed, cells changed, level_completed/game_over). This is ground truth, not a prediction -- prefer it over spending real actions to test a 'what would happen if' hypothesis, and over building a predict() when you only need to compare a few concrete plans. Replay the winner with `action(...)` to make it count. Available whenever save_checkpoint is. HARD LIMIT: after 3 probe calls in a row with no real action, try_actions LOCKS (returns an error) until you execute a real `action(...)`.\n"
    "- `plan_real(actions=None, max_depth=6, max_nodes=250, rollouts=True)` searches the REAL engine for a sequence that completes the current level -- first a novelty-guided systematic frontier over actual engine states via snapshot/rewind (deduplicating repeated boards, biggest board-changes first), then random DEEP rollouts (Monte-Carlo, ~24 steps) with the leftover budget, which can find solutions far deeper than max_depth. NO predict() or theory needed: the engine itself is the world model. Defaults to every currently-valid non-MOUSE action as the move set; for click games pass a SHORT list of promising clicks via `actions=[{'action': 'MOUSE', 'row': r, 'col': c}, ...]`. Returns `{'plan': [...] | None, 'reason': ..., 'found_by': 'frontier'|'rollout'}`; a found plan is engine-verified and the harness EXECUTES it immediately on your behalf (`res['note']` reports the outcome -- do NOT replay it; analyze the executed sequence to learn the mechanics). Budgeted (~10s); 'state_space_exhausted' means every reachable state within max_depth was genuinely tried (a rollout miss, by contrast, proves nothing). Zero recorded actions either way.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described by your world model. If confidence is low, program a discriminating probe and revise the world model from the result.\n"
    "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
    "- Inspect current and history frames from Python instead of describing frames freehand.\n"
    "- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n"
    "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable\n"
    "- A strong default loop is: summarize the board, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n"
    "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
    "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
    "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
    "- Call `action(...)` inside Python rather than returning action text in the chat.\n"
    "- `action(...)` accepts an ordered list of one or more actions. Once your code has selected a reliable sequence, it is often useful to batch it.\n"
    "- You can also call `action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
)

# atlas 27.08: found live on r11l -- the human baseline solves this game in
# 2 moves, but the model spent 45 minutes and tens of thousands of tokens
# verifying a pixel-frequency theory of a MOUSE-only game without ever
# trying more than 2 real actions. The static PYTHON_ADDENDUM already says
# "probe... then plan", but that guidance alone did not stop the model
# reaching for verify_theory()'s statistical rigor before it had even tried
# every kind of control the game currently offers -- confirmed by the user's
# own play-testing ("я, как человек, всегда начинаю с проб, что какие
# рычаги правления делают, как реагируют фигуры"). This fires BEFORE the
# theory checkpoint below (see the priority chain in tool_agent.py) --
# cheap exploration of the control surface is more foundational than either
# a goal reframe or a dynamics theory, and it's the one thing a human does
# first that this checkpoint chain never explicitly forced.
ATLAS_EXPLORE_FIRST_CHECKPOINT = (
    "[atlas checkpoint] This game currently offers {valid_actions}, but you have only tried "
    "{tried} of them for real so far -- {untried} still untested. Before building or refining a "
    "predict() theory, spend one real action() on each untried control (any single action is "
    "enough) and observe what changed. The fastest way to learn what a lever does is to pull it, "
    "not to infer it from pixel statistics alone. Once every available control has been tried at "
    "least once, verify_theory()'s statistical rigor is worth the actions it costs; before that, "
    "it is premature."
)

# atlas: injected by tool_agent.py into a specific turn's prompt (not static
# prompt furniture) when the harness's own state says the model has drifted
# back to poking instead of using verify_theory/plan_with_theory. Mentioning
# a tool once in static instructions gets it used in ~0.2% of turns (measured
# on our own harness's C0 mechanism); a harness-triggered reminder that
# reappears until the model acts is the fix that actually worked there.
ATLAS_THEORY_CHECKPOINT = (
    "[atlas checkpoint] You have several recorded transitions but no predict() "
    "verified at accuracy >= 0.6 yet. THIS turn, write predict(grid, action) "
    "encoding your best theory of the dynamics and call verify_theory(predict); "
    "if you already tried, refine predict() against the counterexamples it "
    "returned and call verify_theory again. A partial theory "
    "(verify_theory(predict, actions=['UP', 'DOWN'])) is fine. If whole-board "
    "pixel-perfect prediction seems out of reach (decorative motion, HUD, "
    "rendering noise unrelated to the mechanic), write extract(grid) -> a "
    "small state instead and call verify_theory(predict, extract=extract) -- "
    "predict/compare then work on that compact state, not raw pixels. A "
    "verified theory turns trial-and-error into a free, zero-action search "
    "(plan_with_theory) instead of guessing with real moves -- worth doing "
    "before spending many more actions blind. If a couple of attempts still "
    "don't clear 0.6, call plan_with_theory(..., force=True) once you have "
    "SOME predict() rather than giving up on the idea -- but keep playing in "
    "the meantime, this does not mean pause the game. Also: when available, "
    "plan_real() searches the REAL engine for a level-completing sequence "
    "with NO predict() needed at all, and try_actions() tests concrete "
    "sequences on the real engine speculatively -- both cost zero recorded "
    "actions and are often faster than building a theory first."
)

# atlas: hard backstop, independent of how ATLAS_THEORY_CHECKPOINT above is
# worded -- found live on r11l (v12) that a model can read ANY "try to build
# a theory" nudge as a gate against acting further at all (1 real action in
# 4.4h). Wording alone is not reliable enough to prevent that a second time,
# so the harness tracks real python-tool calls since the last real action()
# call and, once it crosses a threshold, overrides BOTH theory-style
# checkpoints with an unambiguous "just act" instruction for that turn.
#
# 26.08: strengthened again after a live Kaggle debug run (cn04, ka59) showed
# the mechanism firing correctly (8 -> 9 -> 10 -> 11 calls, unresetting) while
# the model's own reasoning explicitly ACKNOWLEDGED the checkpoint each time
# ("the atlas checkpoint is telling me to take action... let me do one quick
# check and then act") and still delayed 2-4 more calls before complying.
# Both zero-score games in that run ended their entire 3600s budget mid-cycle
# on exactly this pattern. The old wording never said the delaying move
# itself was the problem -- "execute SOMETHING now" is compatible with "one
# quick check, then act". Names that exact rationalization and forbids it
# for the very next call, rather than leaving "now" open to interpretation.
ATLAS_FORCE_ACT_OVERRIDE = (
    "[atlas checkpoint] You have made {calls} `python` calls in a row without "
    "a single real action() call. This is not a suggestion for a future turn "
    "-- your VERY NEXT `python` call, the one you are about to write right "
    "now, MUST include a real action(...) call. Do not respond with 'let me "
    "check one more thing first' or 'one quick probe and then I'll act' -- "
    "that exact reasoning is what produced this checkpoint, and repeating it "
    "will only trigger it again with a higher count. Whatever your current "
    "best guess is, execute it now: a probe, a partial plan, even a guess. A "
    "wrong real action teaches you more than another turn of analysis, and "
    "costs far less than the turns you have already spent not acting. "
    "Refining a theory (verify_theory/plan_with_theory) can continue on "
    "LATER turns alongside real actions -- it does not have to finish first."
)

# atlas 27.08: hard backstop for a THIRD failure mode on the same test game
# (r11l) that ATLAS_THEORY_CHECKPOINT above has already been tuned against
# twice (v12: total paralysis; v16: soft wording silently ignored, 0/662
# real calls). This time (Kaggle v21) the model was neither paralyzed nor
# silent -- it kept acting normally AND kept engaging with the checkpoint in
# its own reasoning -- but never once converted 5 checkpoint firings into a
# verify_theory() call, despite explicitly deriving the correct dynamics
# rule in prose twice ("the head moves halfway toward the gray blob's
# center on each click"). Diagnosed with Gemini as an external critic: this
# is a different ask than ATLAS_FORCE_ACT_OVERRIDE's ("call action()", a
# zero-authoring API trigger) -- verify_theory requires the model to author
# original code encoding a rule it already holds only in prose, and a soft
# nudge to do that generalizes far worse than a hard directive to call an
# already-well-defined function (see the rollback-ultimatum contrast: first
# showing on a totally different game, wa30, got immediate compliant use).
# Deliberately does NOT gate on accuracy (unlike the v12 mistake) -- it only
# requires the ATTEMPT (one verify_theory() call, any result), which can
# never become an unsatisfiable gate the way ">= 0.6 accuracy" can, and it
# does not forbid action() in the same turn (nothing stops both calls living
# in the same python snippet, so there is no actual "theory OR play" choice
# being forced here, unlike ATLAS_FORCE_ACT_OVERRIDE's wording might imply).
ATLAS_THEORY_FORCE_OVERRIDE = (
    "[atlas checkpoint] You have made {calls} `python` calls this game with "
    "ZERO verify_theory() calls. This is not a suggestion anymore -- your "
    "VERY NEXT `python` call MUST define a predict(grid, action) function "
    "and call verify_theory(predict) with it. This does not cost you the "
    "turn: action(...) can still be called in the exact SAME snippet, "
    "before or after verify_theory -- writing a theory and taking a real "
    "step are not exclusive, and you do not have to pick one. A low or even "
    "0% accuracy result is a GOOD outcome here, not a failure to avoid -- "
    "it hands you concrete counterexamples to refine against next turn, "
    "which is strictly more useful than the same guess held only in your "
    "own reasoning and never tested. If you already have a specific rule in "
    "mind (e.g. 'the object moves halfway toward the target each click'), "
    "that rule already IS a predict() function -- write exactly that: `def "
    "predict(grid, action): ...; return new_grid`. Do not respond with 'let "
    "me verify this a bit more in my head first' or 'I will write the "
    "theory once I understand this anomaly' -- that exact delay is what "
    "produced this checkpoint, and holding a correct rule in prose without "
    "ever encoding it does not count as progress toward the 0.6 bar. If you "
    "already called verify_theory() but it tested 0 transitions (e.g. right "
    "after a rollback wiped your history), that call does not clear this -- "
    "take one real action() first so a transition exists, then call "
    "verify_theory() again against it."
)

# atlas 27.08 (late): principle-force for plan_real, prepared the same day
# the tool shipped -- the plan_real RunPod adoption run showed the familiar
# ~0% voluntary uptake (0 calls across the run's first hour despite the
# static docs and the theory-checkpoint pointer), the exact pattern every
# un-nudged tool here has shown (memo 0%, verify_theory 0.2%). Same
# two-layer playbook as rollback: a hard checkpoint at stall, then the
# harness runs the search ITSELF if ignored (possible precisely because
# plan_real needs no model-authored code for non-MOUSE games). For
# MOUSE-only games the harness cannot pick candidate clicks, so there the
# checkpoint asks the model for a short click list and never auto-runs.
ATLAS_PLAN_REAL_FORCE_CHECKPOINT = (
    "[atlas checkpoint] You have taken {stalled} real actions on this level "
    "with no progress, and plan_real() has not been tried once. Your VERY "
    "NEXT `python` call MUST call plan_real({args_hint}) -- it searches the "
    "REAL engine for a level-completing sequence (snapshot -> try -> "
    "rewind), needs NO predict() or theory, costs zero recorded actions, "
    "and either finds a verified plan (which the harness then EXECUTES "
    "on your behalf) or PROVES no such sequence exists within its "
    "depth, which is also real information (the level then needs something "
    "outside that move set). Do not respond with 'let me investigate a bit "
    "more first' -- {stalled} actions of investigation are exactly what "
    "produced this checkpoint. {escalation}"
)
ATLAS_PLAN_REAL_ESCALATION_DEFAULT = (
    "This has been shown {streak} time(s) in a row; if it is ignored again, "
    "the harness will run the search itself with default settings."
)
ATLAS_PLAN_REAL_ESCALATION_MOUSE = (
    "This is a MOUSE game: pass a SHORT list (3-8) of the most promising "
    "clicks -- your read of the board beats any heuristic. If you ignore "
    "this, the harness will run the search itself with auto-derived clicks "
    "(centers of the largest segmentation objects), which are blunter than "
    "yours."
)
ATLAS_PLAN_REAL_MOUSE_ARGS_HINT = (
    "actions=[{'action': 'MOUSE', 'row': r, 'col': c}, ...], max_depth=3"
)

# atlas 25.08: found on dc22 (a Gemini teacher-data transcript from our old
# harness, not this one) -- 221 verify_theory( calls, but the model was
# cycling through 4 unrelated high-level theories of what KIND of mechanic
# this is (a rotating dial -> a camera capture -> a lathe/silhouette -> an
# assembly arm), each one abandoned rather than falsified by evidence.
# ATLAS_THEORY_CHECKPOINT only ever pushes "refine predict()" -- it has no
# way to say the dynamics might not be the actual problem. This checkpoint
# fires instead once verify_theory has been tried many times with no
# success, explicitly naming the GOAL model (not the mechanic) as the other
# thing that could be wrong.
ATLAS_GOAL_RECONSIDER_CHECKPOINT = (
    "[atlas checkpoint] You have called verify_theory {calls} times this game "
    "without ever reaching 0.6 accuracy. If you have been refining the SAME "
    "predict() against its counterexamples, that is still worth continuing. "
    "But if you have rewritten predict() for several DIFFERENT high-level "
    "theories of what kind of mechanic this is (not just tightening one "
    "theory), that is a signal the dynamics may not be the real problem -- "
    "reconsider your GOAL model instead: what are you actually trying to "
    "build, match, or reach, based on the evidence gathered so far? A wrong "
    "goal model makes every predict() attempt fail for reasons that have "
    "nothing to do with the mechanic itself."
)

ATLAS_NOTE_ENFORCEMENT_CHECKPOINT = (
    "[atlas checkpoint] Last turn, {detail} If it did not fully succeed "
    "(check last_action_result / board_changed / level_completed now), that is "
    "exactly the composed-rollout risk res['note'] was warning about: predict() "
    "is only verified on single, already-observed transitions, and chaining it "
    "across steps it only imagined can be wrong even at high verified_accuracy. "
    "From now on, when a plan_with_theory() result has more than one step, use "
    "execute_plan(res['plan'], predict) instead of action(res['plan']) directly "
    "-- it runs the plan one real step at a time and stops itself the moment a "
    "real outcome diverges from predict()'s forecast, instead of spending every "
    "remaining step on a plan that has already stopped working."
)

# atlas 26.08: found on a live Kaggle debug run across 3 games (81 actions
# total) -- `memo` was never written to even once, despite the existing
# prose bullet in PYTHON_ADDENDUM describing it AND a dedicated example for
# alignment/positioning puzzles. Same lesson as everything else in this file:
# a passive prompt mention gets used in ~0.2% of turns (the C0 finding);
# here it was 0%. The model instead tracks facts purely in its own
# natural-language reasoning and silently re-derives/corrects them each turn
# (caught live: "OK, so white = 129 pixels (my '135' memory was wrong)") --
# exactly the drift `memo` exists to prevent. This checkpoint applies the
# same harness-triggered-nag fix that worked for verify_theory/plan_with_theory
# adoption. Lowest priority in the chain (fires only once nothing more urgent
# is active) since forcing memo use on a game that genuinely doesn't need it
# would just waste a turn inventing something to store.
ATLAS_MEMO_CHECKPOINT = (
    "[atlas checkpoint] You have made {calls} `python` calls this game and "
    "written nothing to `memo` yet. `memo` is the one thing that survives "
    "between calls -- everything else here is rebuilt from scratch every "
    "turn. If there is any fact you keep re-deriving from the raw board each "
    "turn (a confirmed position, which object is which via its `hash`, "
    "which region is invariant, a rule you already verified), write it into "
    "`memo` THIS turn (e.g. `memo['anchor'] = {{'row': r, 'col': c, 'step': "
    "current_frame.step}}`) and read it back next turn instead of "
    "recomputing it from the image. Re-deriving the same fact from scratch "
    "every turn is a known failure mode: small per-turn read errors compound "
    "over many turns into a value that never actually converges, even when "
    "the underlying mechanic is already well understood."
)

# atlas 26.08 (from an external design review of the memo-adoption finding):
# theory-building keeps failing on some games not because the MECHANIC is
# hard, but because pixel-perfect predict() is the wrong bar for it (noise,
# HUD, rendering) -- ATLAS_THEORY_CHECKPOINT already mentions `extract=` as an
# option, but mentioning-as-an-option clearly isn't a strong enough nudge on
# its own (extract= adoption was real but partial, ~30% of games, even after
# the memo-checkpoint fix). This fires when verify_theory keeps failing AND
# extract was never tried this game -- a narrower, more specific suggestion
# than ATLAS_GOAL_RECONSIDER_CHECKPOINT (which questions the GOAL model);
# this one questions the PREDICTION APPROACH instead, and fires first since
# it is the cheaper thing to try before concluding the goal itself is wrong.
ATLAS_EXTRACT_CHECKPOINT = (
    "[atlas checkpoint] You have called verify_theory {calls} times this game, "
    "still below 0.6 accuracy, and never once passed `extract=`. Before "
    "concluding the mechanic or goal is wrong, try abstracting the board "
    "first: write extract(grid) -> a small JSON-safe state (e.g. "
    "{{'player': (r, c), 'keys': [...]}} or a coarse tile grid) and call "
    "verify_theory(predict, extract=extract) THIS turn. Pixel-perfect "
    "whole-board prediction is often the wrong bar -- decorative motion, HUD "
    "elements, or rendering noise can make it fail even when your "
    "understanding of the actual mechanic is already correct."
)

# atlas 26.08 (from the same external design review, addressing "action
# paralysis" -- clicking without progress -- as distinct from the
# "analysis paralysis" ATLAS_FORCE_ACT_OVERRIDE already covers). Fires on
# either of two host-detected signals: (a) many real actions with zero level
# progress, or (b) the board has returned to a state it was already in a
# couple of actions ago (a tight ping-pong loop). Names a specific SYSTEM
# anchor (auto-created by the host at game start and at every level-up, not
# something the model had to remember to create itself -- voluntary
# save_checkpoint adoption would suffer the exact same near-zero rate memo
# did) and, like ATLAS_FORCE_ACT_OVERRIDE, forces the very next call rather
# than leaving compliance timing open to interpretation.
ATLAS_FORCE_ROLLBACK_CHECKPOINT = (
    "[atlas checkpoint] {reason} The current line of reasoning is corrupted "
    "-- more actions from here are unlikely to help. Your VERY NEXT `python` "
    "call MUST call rollback('{checkpoint_id}', lesson_learned=...). In "
    "`lesson_learned`, summarize what specific hypothesis you were testing "
    "since that checkpoint and why it failed -- that summary is the ONLY "
    "thing that survives the rollback, so make it count. Do not attempt any "
    "other action first. This has been shown {streak} time(s) in a row; if "
    "it is ignored much longer the harness will perform the rollback for you "
    "with a generic note instead of your own diagnosis."
)

ATLAS_PLAN_CHECKPOINT_TEMPLATE = (
    "[atlas checkpoint] Your predict() verified at accuracy {acc:.2f}. Stop "
    "probing one action at a time and search with it instead: write "
    "goal(grid) -> bool for the state you want, then call "
    "plan_with_theory(predict, goal); if res['plan'] is not None, run "
    "action(res['plan']). The search costs zero real actions and a searched "
    "plan scores far better than the same moves found by trial and error. "
    "If res['note'] is set (plans with more than one step), verify_theory "
    "only checked single already-observed transitions -- a long plan can "
    "still fail mid-way on a mechanic that saturates or changes after a few "
    "moves. Use execute_plan(res['plan'], predict) instead of action(res['plan']) "
    "directly -- it stops itself as soon as a real step diverges from "
    "predict()'s forecast, instead of firing the whole plan blind."
)

# atlas 27.08: hard backstop for the soft PLAN_CHECKPOINT above, per the same
# Gemini-diagnosed pattern as ATLAS_THEORY_FORCE_OVERRIDE -- a recurring soft
# nudge risks the same silent-ignore failure verify_theory's had before that
# fix. Gated on the ATTEMPT (a real plan_with_theory( call), never on
# whether a plan was actually found -- res['plan'] being None is itself
# useful information (the goal is unreachable from here as modeled), not a
# failure to avoid, so this can never become an unreachable/unsafe gate.
# Threshold is 2x _ATLAS_PLAN_NAG_EVERY, same "let the soft version get a
# real chance first" logic as the theory-force threshold vs. its own nag.
ATLAS_PLAN_FORCE_OVERRIDE = (
    "[atlas checkpoint] Your predict() has been verified at accuracy {acc:.2f} "
    "for {calls} `python` calls without a single plan_with_theory() or "
    "execute_plan() call. This is not a suggestion anymore -- your VERY NEXT "
    "`python` call MUST call plan_with_theory(predict, goal) with a "
    "goal(grid) -> bool you define for the state you want. This does not "
    "cost you the turn: if a plan comes back, execute it in the SAME "
    "snippet (execute_plan(...) if it has more than one step). The search "
    "costs ZERO real actions even when plan_with_theory() returns "
    "plan=None -- learning the goal is unreachable from here (with the "
    "theory and actions you gave it) is itself useful information, not a "
    "wasted call. Do not respond with 'let me take a couple more manual "
    "steps first' -- you already have a verified theory; searching with it "
    "is strictly better than guessing one action at a time."
)

COMPACT_TOOL_SESSION_ADDENDUM = (
    "\n\nTool session rules:\n"
    "- You have exactly one tool: `python`.\n"
    f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n"
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_result`.\n"
    "- Each `python` tool call has a hard time limit of 30 seconds.\n"
    "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
)
