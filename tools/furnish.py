"""Enhanced-mode data for each plan: rooms, furniture layout, window glass, lintels, styled floor texture.

Appends `MODELS[key].plus = {...}` to the per-type model JS files written by extract.py.
"""
import base64, glob, io, json, os, sys
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

from extract import CONCRETE, BLOCK, replay, boxes_from_mask, scale_of

SRC, OUT = 'plans', (sys.argv[1] if len(sys.argv) > 1 else 'out')
CELL = 0.02
C = lambda m: int(round(m / CELL))          # metres -> cells
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- rasters

def rasters(page, bb, zoom):
    W, H = page.rect.width, page.rect.height
    fills = [d for d in page.get_drawings()
             if d.get('fill') is not None and d['rect'].x1 < W * 0.655 and d['rect'].y0 > H * 0.22]

    def draw(erase):
        doc = fitz.open(); pg = doc.new_page(width=W, height=H)
        for d in fills:
            c = round(d['fill'][0], 2)
            col = (0, 0, 0) if c in (CONCRETE, BLOCK) else (1, 1, 1) if (c == 1.0 and erase) else None
            if col is None:
                continue
            sh = pg.new_shape(); replay(sh, d['items'])
            sh.finish(fill=col, color=None, closePath=d.get('closePath', True), even_odd=d.get('even_odd', False)); sh.commit()
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bb, alpha=False, colorspace=fitz.csGRAY)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width) < 128

    walls, full = draw(True), draw(False)
    return walls, full, full & ~walls          # windows = wall cells cut by white fills


# ---------------------------------------------------------------- labels

KINDS = [  # (kind, all keywords that must appear)
    ('master', ['שינה', 'הורים']), ('bath', ['רחצה']), ('wc', ['אורחים']), ('bedroom', ['שינה']),
    ('living', ['דיור']), ('kitchen', ['מטבח']), ('dining', ['אוכל']), ('service', ['שירות']),
    ('laundry', ['כביסה']), ('garden', ['חצר']), ('garden', ['גינה']), ('balcony', ['מרפסת']),
]


def labels(page, bb, zoom):
    out = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            r = fitz.Rect(l['bbox'])
            if not r.intersects(bb):
                continue
            out.append((r, ' '.join(s['text'] for s in l['spans']).strip()))
    # merge fragments on the same line that touch horizontally
    merged = sorted(out, key=lambda t: t[0].x0)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(len(merged)):
                if i == j:
                    continue
                (ra, ta), (rb, tb) = merged[i], merged[j]
                same_line = abs((ra.y0 + ra.y1) / 2 - (rb.y0 + rb.y1) / 2) < 3
                if same_line and -2 < rb.x0 - ra.x1 < 7:
                    merged[i] = (ra | rb, tb + ' ' + ta)       # RTL: right fragment first
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    res = []
    for r, t in merged:
        if len(t) > 22:
            continue
        for kind, kws in KINDS:
            if all(k in t for k in kws):
                cx, cy = (r.x0 + r.x1) / 2 - bb.x0, (r.y0 + r.y1) / 2 - bb.y0
                res.append(dict(kind=kind, text=t, x=int(cx * zoom), y=int(cy * zoom)))
                break
    return res


# ---------------------------------------------------------------- doors

def door_arcs(page, bb, zoom, m_per_pt):
    """Door swings are drawn as quarter-circle polylines. Returns their bounding boxes in cells."""
    W, H = page.rect.width, page.rect.height
    out = []
    for d in page.get_drawings():
        r = d['rect']
        if d.get('fill') is not None or not r.intersects(bb) or r.x1 > W * 0.655 or r.y0 < H * 0.22:
            continue
        ls = [it for it in d['items'] if it[0] == 'l']
        if len(ls) < 6:
            continue
        side = max(r.width, r.height) * m_per_pt
        if not (0.5 <= side <= 1.25) or min(r.width, r.height) / max(r.width, r.height) < 0.8:
            continue
        pts = np.array([[it[1].x, it[1].y] for it in ls] + [[ls[-1][2].x, ls[-1][2].y]])
        fit = min((np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).std() / np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).mean())
                  for cx, cy in ((r.x0, r.y0), (r.x1, r.y0), (r.x0, r.y1), (r.x1, r.y1)))
        if fit < 0.03:
            corners = ((r.x0, r.y0), (r.x1, r.y0), (r.x0, r.y1), (r.x1, r.y1))
            cx, cy = min(corners, key=lambda c: np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]).std())
            box = [int((r.x0 - bb.x0) * zoom), int((r.y0 - bb.y0) * zoom),
                   int(np.ceil((r.x1 - bb.x0) * zoom)), int(np.ceil((r.y1 - bb.y0) * zoom))]
            ends = [pts[0], pts[-1]]
            out.append(dict(box=box, hinge=((cx - bb.x0) * zoom, (cy - bb.y0) * zoom),
                            ends=[((e[0] - bb.x0) * zoom, (e[1] - bb.y0) * zoom) for e in ends],
                            radius=float(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).mean() * m_per_pt)))
    return out


def dashed_arcs(page, bb, zoom, m_per_pt):
    """Dashed swings (e.g. the front door) come as many tiny separate segments: chain them, fit a circle."""
    W, H = page.rect.width, page.rect.height
    segs = []
    for d in page.get_drawings():
        r = d['rect']
        if d.get('fill') is not None or not r.intersects(bb) or r.x1 > W * 0.655 or r.y0 < H * 0.22:
            continue
        if len(d['items']) != 1 or d['items'][0][0] != 'l':
            continue
        p1, p2 = d['items'][0][1], d['items'][0][2]
        L = np.hypot(p2.x - p1.x, p2.y - p1.y) * m_per_pt
        if 0.015 <= L <= 0.2:
            segs.append(((p1.x, p1.y), (p2.x, p2.y)))
    if not segs:
        return []
    P = np.array([[*a, *b] for a, b in segs])
    n = len(P)
    gap = 0.12 / m_per_pt
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ends = np.concatenate([P[:, :2], P[:, 2:]])
    idx = np.concatenate([np.arange(n), np.arange(n)])
    from scipy.spatial import cKDTree
    for i, j in cKDTree(ends).query_pairs(gap):
        a, b = find(idx[i]), find(idx[j])
        if a != b:
            parent[a] = b
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) < 6:
            continue
        pts = np.concatenate([P[g, :2], P[g, 2:], (P[g, :2] + P[g, 2:]) / 2])
        tol = 0.015 / m_per_pt
        best = None
        rs = np.random.default_rng(3)
        for _ in range(400):                              # RANSAC: frame/jamb lines get chained in too
            i, j, k = rs.choice(len(pts), 3, replace=False)
            (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[j], pts[k]
            D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(D) < 1e-9:
                continue
            ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / D
            uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / D
            R = np.hypot(x1 - ux, y1 - uy)
            if not (0.6 <= R * m_per_pt <= 1.25):
                continue
            inl = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - R) < tol
            if best is None or inl.sum() > best[0]:
                best = (inl.sum(), ux, uy, R, inl)
        if best is None or best[0] < 12:
            continue
        _, cx, cy, R, inl = best
        pts = pts[inl]
        x, y = pts[:, 0], pts[:, 1]
        ang = np.arctan2(y - cy, x - cx)
        srt = np.sort(ang)
        gaps = np.diff(np.r_[srt, srt[0] + 2 * np.pi])
        k = int(np.argmax(gaps))                     # largest empty sector separates the ends
        e1, e2 = srt[(k + 1) % len(srt)], srt[k]
        span = (e2 - e1) % (2 * np.pi)
        if not (np.radians(70) <= span <= np.radians(110)):
            continue
        E = [(cx + R * np.cos(e), cy + R * np.sin(e)) for e in (e1, e2)]
        box = [min(cx, *[e[0] for e in E]), min(cy, *[e[1] for e in E]), max(cx, *[e[0] for e in E]), max(cy, *[e[1] for e in E])]
        out.append(dict(box=[int((box[0] - bb.x0) * zoom), int((box[1] - bb.y0) * zoom),
                             int(np.ceil((box[2] - bb.x0) * zoom)), int(np.ceil((box[3] - bb.y0) * zoom))],
                        hinge=((cx - bb.x0) * zoom, (cy - bb.y0) * zoom),
                        ends=[((e[0] - bb.x0) * zoom, (e[1] - bb.y0) * zoom) for e in E],
                        radius=float(R * m_per_pt), dashed=True))
    return out


def jamb_hits(full, hinge, end):
    h, w = full.shape
    hx, hy = hinge; ex, ey = end
    n = np.hypot(ex - hx, ey - hy) or 1
    ux, uy = (ex - hx) / n, (ey - hy) / n
    hits, r = 0, C(0.08)
    for s in range(2, C(0.25)):
        x, y = int(ex + ux * s), int(ey + uy * s)
        hits += bool(full[max(0, y - r):max(0, y + r), max(0, x - r):max(0, x + r)].any())
    return hits


def keep_dashed(a, full, space, room_ids, indoor):
    """A dashed arc is an apartment door only if it sweeps a room, hinges on a wall and closes onto a jamb."""
    h, w = full.shape
    x0, y0, x1, y1 = a['box']
    cx, cy = min(w - 1, max(0, (x0 + x1) // 2)), min(h - 1, max(0, (y0 + y1) // 2))
    if space[cy, cx] not in room_ids or not indoor[cy, cx]:
        return False
    hx, hy = int(a['hinge'][0]), int(a['hinge'][1])
    r = C(0.15)
    if not full[max(0, hy - r):hy + r, max(0, hx - r):hx + r].any():
        return False
    return max(jamb_hits(full, a['hinge'], e) for e in a['ends']) >= 3


def door_leaves(arcs, full):
    """3D door leaf per swing: hinge (m), width (m), closed and open angles (rad, plan x/z frame).

    Closed position = the radius lying in the wall line: the far jamb is beyond its end and the
    hinge-side wall is behind the hinge. The open radius looks back through the doorway instead.
    """
    h, w = full.shape
    r = C(0.06)

    def wall_along(x0, y0, ux, uy, s0, s1):
        n = 0
        for s in range(s0, s1):
            x, y = int(x0 + ux * s), int(y0 + uy * s)
            n += bool(full[max(0, y - r):max(0, y + r), max(0, x - r):max(0, x + r)].any())
        return n
    leaves = []
    for a in arcs:
        hx, hy = a['hinge']
        scored = []
        for ex, ey in a['ends']:
            L = np.hypot(ex - hx, ey - hy) or 1
            ux, uy = (ex - hx) / L, (ey - hy) / L
            beyond = wall_along(ex, ey, ux, uy, 2, C(0.25))
            behind = wall_along(hx, hy, -ux, -uy, C(0.05), C(0.3))
            scored.append((beyond + behind, np.arctan2(ey - hy, ex - hx)))
        closed, opened = (scored[0], scored[1]) if scored[0][0] >= scored[1][0] else (scored[1], scored[0])
        leaves.append([round(hx * CELL, 3), round(hy * CELL, 3), round(a['radius'], 3),
                       round(float(closed[1]), 4), round(float(opened[1]), 4)])
    return leaves


# ---------------------------------------------------------------- rooms

def wall_gaps(full, max_gap=2.8, max_thick=0.42):
    """Thin bands that bridge gaps between collinear wall ends (doors and windows)."""
    bands = np.zeros_like(full)
    L, T = C(max_gap), C(max_thick)
    for axis in (1, 0):                       # 1: horizontal walls (gap runs along x)
        k = np.ones((1, L) if axis == 1 else (L, 1), bool)
        cand = ndi.binary_closing(full, structure=k) & ~full
        # thickness across the wall = run length perpendicular to the gap
        st = np.array([[0, 1, 0]] * 3) if axis == 1 else np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        lab, n = ndi.label(cand, structure=st)
        if n:
            size = ndi.sum(cand, lab, index=np.arange(1, n + 1))
            keep = np.zeros(n + 1, bool); keep[1:] = size <= T
            bands |= keep[lab]
    return bands


def flood(dist, markers, sealed):
    """Priority-flood watershed: deepest (farthest-from-wall) cells are claimed first."""
    import heapq
    h, w = dist.shape
    lab = markers.copy()
    lab[sealed] = -1
    elev = (-dist).ravel()
    flat = lab.ravel()
    heap = [(elev[i], i) for i in np.flatnonzero(flat > 1)]      # label 1 (outside) is a fixed boundary
    heapq.heapify(heap)
    while heap:
        _, i = heapq.heappop(heap)
        y, x = divmod(i, w)
        L = flat[i]
        for j in ((i - w) if y else -1, (i + w) if y < h - 1 else -1, (i - 1) if x else -1, (i + 1) if x < w - 1 else -1):
            if j >= 0 and flat[j] == 0:
                flat[j] = L
                heapq.heappush(heap, (elev[j], j))
    lab = flat.reshape(h, w)
    lab[lab < 0] = 0
    return lab


def rooms_from(full, labs):
    """Watershed from room labels (plus the outside) over distance-to-wall; rooms meet at doors/windows."""
    bands = wall_gaps(full)
    sealed = full | bands
    h, w = full.shape
    dist = ndi.distance_transform_edt(~sealed)

    # building outline = convex hull of the walls; only space outside it seeds the exterior
    from scipy.spatial import ConvexHull
    from PIL import ImageDraw
    pts = np.argwhere(full)[:, ::-1]
    hp = pts[ConvexHull(pts).vertices]
    him = Image.new('1', (w, h), 0)
    ImageDraw.Draw(him).polygon([tuple(map(int, q)) for q in hp], fill=1)
    hull = np.array(him, bool)
    # guard for furniture/finishes: a wall in at least three of the four directions (one may be a big window)
    f = sealed
    left, right = np.maximum.accumulate(f, axis=1), np.maximum.accumulate(f[:, ::-1], axis=1)[:, ::-1]
    up, down = np.maximum.accumulate(f, axis=0), np.maximum.accumulate(f[::-1], axis=0)[::-1]
    indoor = (left.astype(np.int8) + right + up + down) >= 3
    markers = np.zeros((h, w), np.int32)
    EXT = 1
    markers[~hull & ~sealed] = EXT
    seeds, groups = [], []
    for lb in labs:
        x, y = lb['x'], lb['y']
        ys, xs = np.mgrid[max(0, y - 20):min(h, y + 21), max(0, x - 20):min(w, x + 21)]
        ok = ~sealed[ys, xs]
        if not ok.any():
            continue
        i = np.argmin(np.where(ok, (ys - y) ** 2 + (xs - x) ** 2, 1e9))
        sy, sx = int(ys.flat[i]), int(xs.flat[i])
        # same-kind labels close together share one zone (e.g. two kitchen notes)
        gid = next((g['id'] for g in groups if g['kind'] == lb['kind'] and np.hypot(g['x'] - sx, g['y'] - sy) < C(2.5)), None)
        if gid is None:
            gid = len(groups) + 2
            groups.append(dict(lb, id=gid, x=sx, y=sy))
        markers[max(0, sy - 3):sy + 4, max(0, sx - 3):sx + 4] = np.where(
            ~sealed[max(0, sy - 3):sy + 4, max(0, sx - 3):sx + 4], gid, 0)
    space = flood(dist, markers, sealed)
    exterior = space == EXT
    # Seeds whose flood got cut off (open plans, open balconies) end up tiny: find the space around them.
    yy, xx = np.mgrid[0:h, 0:w]
    owner = {}
    for g in groups:
        reg = space == g['id']
        if reg.sum() * CELL * CELL >= 1.5:
            owner.setdefault(g['id'], []).append(g)
            continue
        ring = ndi.binary_dilation(reg, iterations=4) & ~reg & (space > 0)
        vals = space[ring]
        vals = vals[vals != g['id']]
        if not len(vals):
            continue
        host = int(np.bincount(vals).argmax())
        space[reg] = host
        if host == EXT:
            if g['kind'] in ('balcony', 'garden', 'service'):
                local = (space == EXT) & ((yy - g['y']) ** 2 + (xx - g['x']) ** 2 < C(2.6) ** 2)
                ll, _ = ndi.label(local)
                space[ll == ll[g['y'], g['x']]] = g['id']
        else:
            owner.setdefault(host, []).append(g)
    for host, gs in owner.items():
        hosts = [g for g in gs if g['id'] == host]
        if len(gs) < 2:
            continue
        reg = space == host
        d = np.stack([np.hypot(yy - g['y'], xx - g['x']) for g in gs])
        near = np.argmin(d, axis=0)
        for k, g in enumerate(gs):
            space[reg & (near == k)] = g['id']
            g['siblings'] = [x['id'] for x in gs]
    exterior = space == EXT
    rooms = []
    for g in groups:
        reg = space == g['id']
        area = float(reg.sum() * CELL * CELL)
        if area < 1.0:
            continue
        rooms.append(dict(kind=g['kind'], text=g['text'], x=g['x'], y=g['y'], rid=g['id'], area=area,
                          zone=g.get('siblings', [g['id']])))
    # windows vs doorways: look at what lies on each side of every opening
    OUTDOOR = {'balcony', 'garden', 'service', 'laundry'}
    kind_of = {g['id']: g['kind'] for g in groups}
    blab, _ = ndi.label(bands)
    glass = np.zeros_like(bands); doors = np.zeros_like(bands)
    for i, sl in enumerate(ndi.find_objects(blab), 1):
        sl2 = tuple(slice(max(0, s.start - 6), s.stop + 6) for s in sl)
        comp = blab[sl2] == i
        ring = ndi.binary_dilation(comp, iterations=5) & ~comp
        ids = set(np.unique(space[sl2][ring]).tolist()) - {0}
        outdoor = EXT in ids or any(kind_of.get(x) in OUTDOOR for x in ids) or len(ids) < 2
        (glass if outdoor else doors)[sl2] |= comp
    # any opening with usable floor on both sides is a (potential) door: keep 1 m clear on each side
    ok_ids = {g['id'] for g in groups}
    keep_clear = np.zeros_like(bands)
    for i, sl in enumerate(ndi.find_objects(blab), 1):
        (ys, xs) = sl
        y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
        along_x = (x1 - x0) >= (y1 - y0)
        wide = max(x1 - x0, y1 - y0) * CELL > 1.3          # sliding doors to balconies
        o, m, reach = C(0.5), C(0.1), C(0.8 if wide else 1.0)
        if along_x:
            sides = [space[max(0, y0 - o):y0, x0:x1], space[y1:y1 + o, x0:x1]]
        else:
            sides = [space[y0:y1, max(0, x0 - o):x0], space[y0:y1, x1:x1 + o]]

        def walkable(a):
            return a.size and np.isin(a, list(ok_ids)).mean() >= 0.5
        if (x1 - x0) * CELL > 2.9 and (y1 - y0) * CELL > 2.9:
            continue
        if all(walkable(s) for s in sides):
            if along_x:
                keep_clear[max(0, y0 - reach):y1 + reach, max(0, x0 - m):x1 + m] = True
            else:
                keep_clear[max(0, y0 - m):y1 + m, max(0, x0 - reach):x1 + reach] = True
    return sealed, space, rooms, glass, doors, indoor & hull, keep_clear


# ---------------------------------------------------------------- placement

class Grid:
    def __init__(self, closed, full, space, door_zone, indoor=None):
        self.indoor = indoor
        self.wall = closed.astype(np.int32)
        self.full = full
        self.space = space
        self.occ = np.zeros_like(full, bool)
        self.door = door_zone
        self.Iw = self.integral(self.wall)

    @staticmethod
    def integral(a):
        return np.pad(a.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    @staticmethod
    def rect_sum(I, y0, x0, y1, x1):
        return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]

    def place(self, rid, w, d, near=None, wall_frac=0.85, door_ok=False, need_back=True, clear=0,
              prefer_center=True, near_weight=1.0, along=None):
        """Find best spot for a w x d (m) item with its back on a wall. Returns dict or None."""
        H, W = self.wall.shape
        reg = np.isin(self.space, rid) if isinstance(rid, (list, tuple)) else self.space == rid
        blocked = ~reg | self.occ | (False if door_ok else self.door)
        if self.indoor is not None:
            blocked = blocked | ~self.indoor
        Ib = self.integral(blocked)
        ys, xs = np.where(reg)
        y0r, y1r, x0r, x1r = ys.min(), ys.max(), xs.min(), xs.max()
        best = None
        cw, cd, cl, t = C(w), C(d), C(clear), 6
        for o in 'NSEW':                                   # side the item's back faces
            fw, fh = (cw, cd) if o in 'NS' else (cd, cw)
            gx = np.arange(max(x0r, cl), min(x1r - fw + 2, W - fw - cl - 1), 2)
            gy = np.arange(max(y0r, cl), min(y1r - fh + 2, H - fh - cl - 1), 2)
            if len(gx) == 0 or len(gy) == 0:
                continue
            X, Y = np.meshgrid(gx, gy)
            free = self.rect_sum(Ib, Y - cl, X - cl, Y + fh + cl, X + fw + cl) == 0
            if need_back:
                if o == 'N':
                    by0, bx0, by1, bx1, n = Y - t, X, Y, X + fw, t * fw
                elif o == 'S':
                    by0, bx0, by1, bx1, n = Y + fh, X, Y + fh + t, X + fw, t * fw
                elif o == 'W':
                    by0, bx0, by1, bx1, n = Y, X - t, Y + fh, X, t * fh
                else:
                    by0, bx0, by1, bx1, n = Y, X + fw, Y + fh, X + fw + t, t * fh
                by0, bx0 = np.clip(by0, 0, H), np.clip(bx0, 0, W)
                by1, bx1 = np.clip(by1, 0, H), np.clip(bx1, 0, W)
                back = self.rect_sum(self.Iw, by0, bx0, by1, bx1) / n
                free &= back >= wall_frac
            else:
                back = np.ones_like(X, float)
            if along is not None and o not in along:
                continue
            if not free.any():
                continue
            cx, cy = X + fw / 2, Y + fh / 2
            score = back * 2.0
            if near is not None:
                score = score - near_weight * np.hypot(cx - near[0], cy - near[1]) * CELL
            if prefer_center:
                # distance from centre of the free run along the wall
                rc = (x0r + x1r) / 2 if o in 'NS' else (y0r + y1r) / 2
                score = score - 0.6 * np.abs((cx if o in 'NS' else cy) - rc) * CELL
            score = np.where(free, score, -1e9)
            i = np.argmax(score)
            if score.flat[i] > -1e8 and (best is None or score.flat[i] > best[0]):
                best = (score.flat[i], o, int(X.flat[i]), int(Y.flat[i]), fw, fh)
        if best is None:
            return None
        _, o, x, y, fw, fh = best
        if need_back:                                  # slide back until flush with the wall
            dx, dy = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}[o]
            for _ in range(8):
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or self.rect_sum(Ib, ny, nx, ny + fh, nx + fw) > 0:
                    break
                x, y = nx, ny
        return dict(o=o, x=x, y=y, fw=fw, fh=fh)

    def claim(self, p, pad=0):
        self.occ[max(0, p['y'] - pad):p['y'] + p['fh'] + pad, max(0, p['x'] - pad):p['x'] + p['fw'] + pad] = True

    def claim_box(self, x0, y0, x1, y1):
        self.occ[max(0, y0):y1, max(0, x0):x1] = True


ROT = {'N': 0.0, 'E': -np.pi / 2, 'S': np.pi, 'W': np.pi / 2}   # item faces +z(south) when back is N
FWD = {'N': (0, 1), 'S': (0, -1), 'W': (1, 0), 'E': (-1, 0)}       # direction the item faces (grid x,y)


def item(kind, p, w, d, **extra):
    cx, cy = (p['x'] + p['fw'] / 2) * CELL, (p['y'] + p['fh'] / 2) * CELL
    return dict(k=kind, x=round(cx, 3), z=round(cy, 3), w=w, d=d, r=round(ROT[p['o']], 4), **extra)


def front_of(p, dist_cells, w, d):
    """Placement dict for an item centred `dist` in front of p, facing back toward p."""
    fx, fy = FWD[p['o']]
    cx, cy = p['x'] + p['fw'] / 2 + fx * dist_cells, p['y'] + p['fh'] / 2 + fy * dist_cells
    o = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}[p['o']]
    fw, fh = (C(w), C(d)) if o in 'NS' else (C(d), C(w))
    return dict(o=o, x=int(cx - fw / 2), y=int(cy - fh / 2), fw=fw, fh=fh)


def ray_to_wall(g, p, max_cells):
    fx, fy = FWD[p['o']]
    cx, cy = p['x'] + p['fw'] / 2, p['y'] + p['fh'] / 2
    start = (p['fh'] if p['o'] in 'NS' else p['fw']) / 2
    for s in range(int(start), max_cells):
        x, y = int(cx + fx * s), int(cy + fy * s)
        if not (0 <= y < g.wall.shape[0] and 0 <= x < g.wall.shape[1]) or g.wall[y, x]:
            return s
    return None


EXTRA = {'dining': 0.84, 'desk': 0.84, 'patio': 0.94}   # chairs reach beyond the table/desk footprint


def footprint(it, shape):
    w, d = it['w'], it['d']
    if it['k'] == 'dining':
        w, d = w + (EXTRA['dining'] if it.get('n', 4) >= 6 else 0), d + EXTRA['dining']
    elif it['k'] == 'desk':
        d += EXTRA['desk']
    elif it['k'] == 'patio':
        w += EXTRA['patio']
    if abs(np.sin(it['r'])) > 0.5:
        w, d = d, w
    x0, x1 = int(round((it['x'] - w / 2) / CELL)) + 1, int(round((it['x'] + w / 2) / CELL)) - 1   # 1-cell rounding tolerance
    y0, y1 = int(round((it['z'] - d / 2) / CELL)) + 1, int(round((it['z'] + d / 2) / CELL)) - 1
    return max(0, y0), min(shape[0], y1), max(0, x0), min(shape[1], x1)


def audit(items, door_zone, walls):
    keep, dropped, worst = [], [], 0
    for it in items:
        y0, y1, x0, x1 = footprint(it, door_zone.shape)
        hit = int(door_zone[y0:y1, x0:x1].sum())
        if hit:
            dropped.append(it['k'])
        else:
            keep.append(it)
    for it in keep:
        y0, y1, x0, x1 = footprint(it, door_zone.shape)
        worst = max(worst, int(door_zone[y0:y1, x0:x1].sum()))
    return keep, dropped, worst


def furnish(g, rooms):
    items = []
    by = {}
    for r in rooms:
        by.setdefault(r['kind'], []).append(r)

    def try_sizes(rid, sizes, **kw):
        for w, d in sizes:
            p = g.place(rid, w, d, **kw)
            if p:
                return p, w, d
        return None, None, None

    def try_hard(r, sizes, **kw):
        """Own zone, then the whole open-plan space, then a part-solid wall. Doorways stay clear throughout."""
        for rid, extra in ((r['rid'], {}), (r['zone'], {}), (r['zone'], dict(wall_frac=0.6)), (r['zone'], dict(wall_frac=0.4))):
            p, w, d = try_sizes(rid, sizes, **{**kw, **extra})
            if p:
                return p, w, d
        return None, None, None

    # bedrooms
    for r in by.get('master', []) + by.get('bedroom', []):
        near = (r['x'], r['y'])
        master = r['kind'] == 'master' or r['area'] > 11.5
        bw = 1.6 if master else (1.2 if r['area'] > 9 else 0.9)
        p, w, d = try_sizes(r['rid'], [(bw + 1.0, 2.05), (bw, 2.05), (0.9, 2.0)])
        if not p:     # headboard under a window / part-solid wall is fine; doorways still stay clear
            p, w, d = try_sizes(r['rid'], [(bw + 1.0, 2.05), (bw, 2.05), (0.9, 2.0)], wall_frac=0.45)
        if p:
            stands = w > bw + 0.5
            items.append(item('bed', p, w, d, bw=bw if stands else w, stands=stands))
            g.claim(p, pad=C(0.45))
        p, w, d = try_sizes(r['rid'], [(2.4, 0.6), (1.8, 0.6), (1.2, 0.6), (0.9, 0.55)], prefer_center=False)
        if not p:
            p, w, d = try_sizes(r['rid'], [(1.8, 0.6), (1.2, 0.6), (0.9, 0.55)], prefer_center=False, wall_frac=0.7)
        if p:
            items.append(item('wardrobe', p, w, d)); g.claim(p, pad=C(0.5))
        if not master and r['area'] > 7:
            p, w, d = try_sizes(r['rid'], [(1.2, 0.6), (1.0, 0.55)], prefer_center=False)
            if p:
                items.append(item('desk', p, w, d)); g.claim(p, pad=C(0.5))

    # kitchen: counter run + fridge next to it
    for r in by.get('kitchen', []):
        near = (r['x'], r['y'])
        p, w, d = try_hard(r, [(3.0, 0.62), (2.6, 0.62), (2.2, 0.62), (1.8, 0.62), (1.4, 0.62)],
                           near=near, near_weight=1.5, prefer_center=False)
        if p:
            items.append(item('counter', p, w, d)); g.claim(p, pad=C(0.15))
            cc = (p['x'] + p['fw'] / 2, p['y'] + p['fh'] / 2)
            q, w2, d2 = None, 0.75, 0.72
            for zone in (r['rid'], r['zone']):
                q = g.place(zone, w2, d2, near=cc, near_weight=3.0, prefer_center=False)
                if q and np.hypot(q['x'] + q['fw'] / 2 - cc[0], q['y'] + q['fh'] / 2 - cc[1]) * CELL < 2.6:
                    break
                q = None
            if q:
                items.append(item('fridge', q, w2, d2)); g.claim(q, pad=C(0.1))
            # keep a 1 m working aisle in front of the counter
            a = front_of(p, (p['fh'] if p['o'] in 'NS' else p['fw']) / 2 + C(0.5), w, 1.0)
            g.claim_box(a['x'], a['y'], a['x'] + a['fw'], a['y'] + a['fh'])

    # living: sofa, coffee table, rug, TV opposite; retry the sofa until its group is clear of doorways
    def clear_of_doors(q, zone):
        y0, x0 = max(0, q['y']), max(0, q['x'])
        sub = zone[y0:q['y'] + q['fh'], x0:q['x'] + q['fw']]
        walls = g.wall[y0:q['y'] + q['fh'], x0:q['x'] + q['fw']]
        doors = g.door[y0:q['y'] + q['fh'], x0:q['x'] + q['fw']]
        return sub.size and sub.mean() > 0.97 and not walls.any() and not doors.any() and not g.occ[y0:q['y'] + q['fh'], x0:q['x'] + q['fw']].any()

    for r in by.get('living', []):
        near = (r['x'], r['y'])
        zone = np.isin(g.space, r['zone']) & (g.indoor if g.indoor is not None else True)
        tried = []
        for _ in range(8):
            p, w, d = try_hard(r, [(2.6, 0.92), (2.2, 0.9), (1.8, 0.88)], near=near, near_weight=0.4)
            if not p:
                break
            depth = p['fh'] if p['o'] in 'NS' else p['fw']
            ct = front_of(p, depth / 2 + C(0.75), 1.1, 0.6)
            if clear_of_doors(ct, zone):
                break
            tried.append(p); g.claim(p)                    # spot is no good: block it and try the next best
            p = None
        for q in tried:
            g.occ[q['y']:q['y'] + q['fh'], q['x']:q['x'] + q['fw']] = False
        if not p:
            continue
        items.append(item('sofa', p, w, d)); g.claim(p, pad=C(0.1))
        items.append(item('coffee', ct, 1.1, 0.6)); g.claim(ct, pad=C(0.1))
        for rw, rd in ((w + 0.4, 2.0), (w, 1.6), (1.6, 1.2)):
            rug = front_of(p, depth / 2 + C(0.75), rw, rd)
            if not g.door[max(0, rug['y']):rug['y'] + rug['fh'], max(0, rug['x']):rug['x'] + rug['fw']].any() and \
                    not g.wall[max(0, rug['y']):rug['y'] + rug['fh'], max(0, rug['x']):rug['x'] + rug['fw']].any():
                items.append(item('rug', rug, rw, rd)); break
        dist = ray_to_wall(g, p, C(5.5))
        cands = ([front_of(p, dist - C(0.22), 1.8, 0.42)] if dist and C(2.4) <= dist <= C(5.5) else []) + \
                [front_of(p, depth / 2 + C(s), 1.8, 0.42) for s in (3.0, 2.6, 2.3)]
        tv = next((q for q in cands if clear_of_doors(q, zone)), None)
        if tv:
            items.append(item('tv', tv, 1.8, 0.42)); g.claim(tv, pad=C(0.1))
            g.claim_box(min(p['x'], tv['x']), min(p['y'], tv['y']), max(p['x'] + p['fw'], tv['x'] + tv['fw']),
                        max(p['y'] + p['fh'], tv['y'] + tv['fh']))

    # dining table: at dining label, else between kitchen and living labels
    seeds = [(r['zone'], (r['x'], r['y'])) for r in by.get('dining', [])]
    if not seeds and by.get('kitchen') and by.get('living'):
        k, l = by['kitchen'][0], by['living'][0]
        if k['rid'] == l['rid']:
            seeds = [(k['zone'], ((k['x'] + l['x']) / 2, (k['y'] + l['y']) / 2))]
    for rid, near in seeds[:1]:
        for w, d, n in [(1.8, 0.95, 6), (1.5, 0.9, 6), (1.2, 0.8, 4)]:
            p = g.place(rid, w, d, near=near, near_weight=2.0, need_back=False, clear=0.55, prefer_center=False)
            if p:
                items.append(item('dining', p, w, d, n=n)); g.claim(p, pad=C(0.6)); break

    # balconies / gardens
    for r in by.get('balcony', []) + by.get('garden', []):
        near = (r['x'], r['y'])
        big = r['area'] > 22
        p = g.place(r['rid'], 0.8, 0.8, near=near, need_back=False, clear=0.4, prefer_center=False, near_weight=0.5)
        if p:
            items.append(item('patio', p, 0.8, 0.8)); g.claim(p, pad=C(0.55))
        if big:
            for _ in range(2):
                q = g.place(r['rid'], 0.7, 1.9, need_back=True, wall_frac=0.6, prefer_center=False)
                if q:
                    items.append(item('lounger', q, 0.7, 1.9)); g.claim(q, pad=C(0.15))
        q = g.place(r['rid'], 0.5, 0.5, need_back=True, wall_frac=0.6, prefer_center=False)
        if q:
            items.append(item('plant', q, 0.5, 0.5)); g.claim(q)

    for r in by.get('service', []):
        p = g.place(r['rid'], 0.62, 0.62, near=(r['x'], r['y']), prefer_center=False)
        if p:
            items.append(item('washer', p, 0.62, 0.62)); g.claim(p)

    for r in by.get('living', [])[:1]:
        p = g.place(r['rid'], 0.5, 0.5, need_back=True, prefer_center=False, near=(r['x'], r['y']), near_weight=-0.3)
        if p:
            items.append(item('plant', p, 0.5, 0.5)); g.claim(p)
    return items


# ---------------------------------------------------------------- floor styling

def wood(h, w):
    img = np.zeros((h, w, 3), np.float32)
    plank_w, plank_l = 18, 120
    base = np.array([196, 160, 118], np.float32)
    for i in range(0, h, plank_w):
        off = rng.integers(0, plank_l)
        for j in range(-off, w, plank_l):
            tone = base * rng.uniform(0.86, 1.08)
            img[i:i + plank_w, max(0, j):j + plank_l] = tone
        img[i:i + 1, :] *= 0.78
    grain = rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    grain = ndi.gaussian_filter(grain, (0.6, 12, 0))
    return np.clip(img + grain, 0, 255)


def tiles(h, w, color, size):
    img = np.ones((h, w, 3), np.float32) * np.array(color, np.float32)
    img += rng.normal(0, 3, (h, w, 1))
    img[::size, :] *= 0.86
    img[:, ::size] *= 0.86
    return np.clip(img, 0, 255)


def styled_floor(plan_img, space, rooms, scale_px):
    pw, ph = plan_img.size
    lab = np.array(Image.fromarray(space.astype(np.int32)).resize((pw, ph), Image.NEAREST))
    out = np.array(plan_img.convert('RGB'), np.float32) * 0 + np.array([238, 236, 231], np.float32)
    wd = wood(ph, pw)
    tl_light = tiles(ph, pw, (226, 224, 219), int(0.6 * scale_px))
    tl_out = tiles(ph, pw, (200, 194, 185), int(0.5 * scale_px))
    mat = {'master': wd, 'bedroom': wd, 'living': wd, 'dining': wd, 'kitchen': tl_light, 'bath': tl_light,
           'wc': tl_light, 'service': tl_light, 'laundry': tl_out, 'balcony': tl_out, 'garden': None}
    for r in rooms:
        m = mat.get(r['kind'])
        sel = lab == r['rid']
        if r['kind'] == 'garden':
            g = np.ones((ph, pw, 3), np.float32) * np.array([126, 168, 96], np.float32) + rng.normal(0, 8, (ph, pw, 1))
            out[sel] = g[sel]
        elif m is not None:
            out[sel] = m[sel]
    # multiply the plan's line-work on top, faintly
    plan = np.array(plan_img.convert('L'), np.float32) / 255.0
    out *= (0.55 + 0.45 * plan)[..., None]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, 'JPEG', quality=60, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- main

def process(path):
    key = os.path.basename(path)[:-4]
    js = os.path.join(OUT, key + '.js')
    src = open(js).read().split('\n(window.MODELS')[0].rstrip('\n') + '\n'
    model = json.loads(src[src.index(']=') + 2:src.rstrip().rindex(';')])
    page = fitz.open(path)[0]
    m_per_pt = 0.0254 / 72 * scale_of(page)
    W, H = page.rect.width, page.rect.height
    walls_paths = [d for d in page.get_drawings() if d.get('fill') is not None and round(d['fill'][0], 2) in (CONCRETE, BLOCK)
                   and d['rect'].x1 < W * 0.655 and d['rect'].y0 > H * 0.22]
    bb = fitz.Rect(walls_paths[0]['rect'])
    for d in walls_paths:
        bb |= d['rect']
    bb = fitz.Rect(bb.x0 - 4, bb.y0 - 4, bb.x1 + 4, bb.y1 + 4)
    zoom = m_per_pt / CELL

    walls, full, _ = rasters(page, bb, zoom)
    labs = labels(page, bb, zoom)
    closed, space, rooms, windows, openings, indoor, keep_clear = rooms_from(full, labs)

    # no-furniture zones: every door swing (+20 cm) and 1 m either side of every passable opening
    door_zone = keep_clear.copy()
    arcs = door_arcs(page, bb, zoom, m_per_pt)
    room_ids = {r['rid'] for r in rooms}
    def overlaps(a, b):
        ax0, ay0, ax1, ay1 = a['box']; bx0, by0, bx1, by1 = b['box']
        ix = max(0, min(ax1, bx1) - max(ax0, bx0)); iy = max(0, min(ay1, by1) - max(ay0, by0))
        return ix * iy > 0.25 * min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    solid = list(arcs)
    arcs += [a for a in dashed_arcs(page, bb, zoom, m_per_pt)
             if keep_dashed(a, full, space, room_ids, indoor) and not any(overlaps(a, s) for s in solid)]
    for x0, y0, x1, y1 in (a['box'] for a in arcs):
        door_zone[max(0, y0 - C(0.2)):y1 + C(0.2), max(0, x0 - C(0.2)):x1 + C(0.2)] = True
    door_zone &= ~closed
    g = Grid(closed, full, space, door_zone, indoor)
    items = furnish(g, rooms)
    items, dropped, worst = audit(items, door_zone, closed)

    def to_m(bs, keep=lambda w, h: True):
        res = []
        for x0, y0, x1, y1 in bs:
            w, h = (x1 - x0) * CELL, (y1 - y0) * CELL
            if w * h >= 0.004 and keep(w, h):
                res.append([round(x0 * CELL, 3), round(y0 * CELL, 3), round(x1 * CELL, 3), round(y1 * CELL, 3)])
        return res

    lintel = lambda w, h: min(w, h) <= 0.32 and 0.55 <= max(w, h) <= 1.4
    plan_img = Image.open(io.BytesIO(base64.b64decode(model['floor'].split(',')[1])))
    plus = dict(
        glass=to_m(boxes_from_mask(windows), keep=lambda w, h: max(w, h) >= 0.3),
        doors=to_m(boxes_from_mask(openings), keep=lintel),
        items=items,
        clear=to_m(boxes_from_mask(door_zone)),
        leaves=door_leaves(arcs, full),
        rooms=[dict(kind=r['kind'], area=round(r['area'], 1)) for r in rooms],
        floor=styled_floor(plan_img, np.where(indoor, space, 0), rooms, plan_img.size[0] / model['w']),
    )
    with open(js, 'w') as f:
        f.write(src)
        f.write('(window.MODELS)[%s].plus=%s;\n' % (json.dumps(key), json.dumps(plus, separators=(',', ':'))))
    kinds = {}
    for it in items:
        kinds[it['k']] = kinds.get(it['k'], 0) + 1
    print(key, 'items', kinds, '| door swings', len(arcs), '| dropped', dropped or '-', '| overlap cells after audit', worst)
    return key, rooms, items, space, closed


if __name__ == '__main__':
    only = sys.argv[2:] or None
    for p in sorted(glob.glob(os.path.join(SRC, '*.pdf'))):
        k = os.path.basename(p)[:-4]
        if '-' in k or (only and k not in only):
            continue
        process(p)
