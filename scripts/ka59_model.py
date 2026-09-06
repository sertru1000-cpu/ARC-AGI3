"""ka59 L2 pure model (cells of 3 px) + A* over (piece positions, selected)."""
import json, heapq, sys, time
SCR = "C:/Users/SERTRU~1/AppData/Local/Temp/2/claude/c--Users-sertru1000-Projects-ARC-AGI-3/9d87a9f6-c490-482e-bf0c-21f2334f08e7/scratchpad"
CELLS = json.load(open(f"{SCR}/ka59_cells.json"))["cells"]      # 0 floor 1 wall 2 boundary
N = 21
SIZES = [(1, 1), (1, 2), (2, 1), (2, 2)]                              # A 3x3, B 3x6, C 6x3, D 6x6 (w,h)
START = ((12, 18), (11, 14), (13, 11), (14, 15))
TARGET = ((17, 17), (3, 3), (18, 13), (2, 14))
SEL0 = 0
SLIDE = 5
DIRS = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}
def cells_of(i, pos):
    w, h = SIZES[i]; x, y = pos
    return [(x + a, y + b) for a in range(w) for b in range(h)]
def inside(i, pos):
    w, h = SIZES[i]; x, y = pos
    return 0 <= x and x + w <= N and 0 <= y and y + h <= N
def has(i, pos, kinds):
    return any(CELLS[cy][cx] in kinds for cx, cy in cells_of(i, pos))
def overlap(i, pos, j, pos2):
    return bool(set(cells_of(i, pos)) & set(cells_of(j, pos2)))
def push_one(i, positions, d, exclude):
    """ifoelczjjh: move piece i by d one step; recursively displace other pieces. returns new positions or None."""
    x, y = positions[i]; np_ = (x + d[0], y + d[1])
    if not inside(i, np_) or has(i, np_, (2,)): return None
    positions = list(positions); positions[i] = np_
    for j in range(4):
        if j == i or j in exclude: continue
        if overlap(i, np_, j, positions[j]):
            res = push_one(j, positions, d, exclude | {i})
            if res is None: return None
            positions = res
    return tuple(positions)
def apply(state, action):
    positions, sel = state
    if action.startswith("SEL"):
        j = int(action[3]); return (positions, j) if j != sel else None
    d = DIRS[action]; x, y = positions[sel]; np_ = (x + d[0], y + d[1])
    if not inside(sel, np_) or has(sel, np_, (1, 2)): return state          # blocked: action spent, nothing moves
    pushed = [j for j in range(4) if j != sel and overlap(sel, np_, j, positions[j])]
    if not pushed:
        p = list(positions); p[sel] = np_; return (tuple(p), sel)
    # slide pushed pieces; the selected one stays
    pos = positions; moving = list(pushed)
    for step in range(60):
        still = []
        for j in moving:
            if step >= SLIDE and not has(j, pos[j], (1,)): continue
            res = push_one(j, pos, d, {sel})
            if res is None: continue
            pos = res; still.append(j)
        moving = still
        if not moving: break
    return (pos, sel)
def h(state):
    positions, sel = state
    tot = 0; unplaced = 0
    for i in range(4):
        dx = abs(positions[i][0] - TARGET[i][0]); dy = abs(positions[i][1] - TARGET[i][1])
        if dx or dy:
            unplaced += 1; tot += (dx + dy) / (SLIDE + 3)
    return tot + max(0, unplaced - 1)
def solve(limit=3_000_000):
    start = (START, SEL0); goal = TARGET
    pq = [(h(start), 0, start)]; g = {start: 0}; parent = {start: None}; n = 0; t0 = time.time()
    while pq:
        f, cost, s = heapq.heappop(pq); n += 1
        if s[0] == goal:
            path = []
            while parent[s]: s, a = parent[s]; path.append(a)
            return path[::-1], n, time.time() - t0
        if cost > g[s]: continue
        for a in list(DIRS) + [f"SEL{j}" for j in range(4)]:
            ns = apply(s, a)
            if ns is None or ns == s: continue
            c = cost + 1
            if c < g.get(ns, 1e9):
                g[ns] = c; parent[ns] = (s, a); heapq.heappush(pq, (c + h(ns), c, ns))
        if n >= limit: return None, n, time.time() - t0
    return None, n, time.time() - t0
if __name__ == "__main__":
    print("as-is model sanity: push B left from probe:", apply(((( 12, 15), (11, 14), (13, 11), (14, 15)), 0), "ACTION3")[0][1], "(expect (6,14))")
    path, n, dt = solve()
    print("expanded", n, f"in {dt:.0f}s; plan length", None if path is None else len(path)); print(path)
    if path: json.dump(path, open(f"{SCR}/ka59_plan.json", "w"))
