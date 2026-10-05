"""Vector key plans: floor outline + core + this apartment (red), one SVG per apartment plan PDF.

Each apartment plan PDF has a small key plan in its corner: the building's floor outline (black lines),
the stair/elevator core (solid grey) and the apartment itself (grey hatch). The page is rendered at
high resolution around the core and traced:
  floor   = the black outline that encloses the core
  core    = the solid grey vector path (exact)
  apt     = the white space the grey hatch lines sit in, flood-filled up to the black lines
Run from the site folder:  python tools/keyplans.py
"""
import glob, json, os
import numpy as np
import cv2
import pymupdf as fitz
from scipy import ndimage as ndi

SRC, OUT = 'pdfs/apartments', 'keyplans'
PX = 12                      # render pixels per PDF point


def core_of(page):
    W, H = page.rect.width, page.rect.height
    cores = [d for d in page.get_drawings() if d.get('fill') is not None and round(d['fill'][0], 2) == 0.75
             and d['rect'].x0 > W * 0.6 and d['rect'].y1 < H * 0.6]
    return max(cores, key=lambda d: d['rect'].width * d['rect'].height) if cores else None


def rectilinear(poly, gap):
    """Turn a near-rectilinear polygon (building axes) into a strictly rectilinear one.

    Diagonal edges are bevels of a square corner, so they get their missing outer corner back;
    tiny steps (shorter than ~half a hatch gap) are merged away.
    """
    P = [tuple(map(float, p)) for p in poly]
    if len(P) < 3:
        return P
    cnt = np.array(P, np.float32).reshape(-1, 1, 2)
    tol = PX * 0.35
    out = []
    for i, p in enumerate(P):
        q = P[(i + 1) % len(P)]
        out.append(p)
        if abs(q[0] - p[0]) > tol and abs(q[1] - p[1]) > tol:          # diagonal: restore the corner
            c1, c2 = (q[0], p[1]), (p[0], q[1])
            out.append(c1 if cv2.pointPolygonTest(cnt, c1, False) < 0 else c2)
    # edges -> H/V lines; vertices = intersections of consecutive lines
    for _ in range(6):
        E = []
        for i, p in enumerate(out):
            q = out[(i + 1) % len(out)]
            h = abs(q[0] - p[0]) >= abs(q[1] - p[1])
            E.append(['H' if h else 'V', (p[1] + q[1]) / 2 if h else (p[0] + q[0]) / 2, abs(q[0] - p[0]) + abs(q[1] - p[1])])
        # merge runs of same orientation
        F = []
        for e in E:
            if F and F[-1][0] == e[0]:
                w0, w1 = F[-1][2], e[2]
                F[-1] = [e[0], (F[-1][1] * w0 + e[1] * w1) / max(w0 + w1, 1e-6), w0 + w1]
            else:
                F.append(e)
        if len(F) > 1 and F[0][0] == F[-1][0]:
            a, b = F[-1], F[0]
            F[0] = [b[0], (a[1] * a[2] + b[1] * b[2]) / max(a[2] + b[2], 1e-6), a[2] + b[2]]; F.pop()
        # drop the shortest tiny step, then rebuild
        short = [i for i, e in enumerate(F) if e[2] < gap * 0.5]
        if len(F) > 4 and short:
            F.pop(min(short, key=lambda i: F[i][2]))
        out = []
        for i, e in enumerate(F):
            f = F[(i + 1) % len(F)]
            if e[0] == f[0]:
                continue
            out.append((e[1], f[1]) if e[0] == 'V' else (f[1], e[1]))
        if not short or len(F) <= 4:
            break
    return out


def squared(mask, gap):
    """Rectilinearise every blob of a mask that is already rotated into building axes."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    for c in cs:
        if cv2.contourArea(c) < (PX * 2) ** 2:
            continue
        ap = cv2.approxPolyDP(c, PX * 0.8, True)[:, 0, :]
        rp = rectilinear(ap, gap)
        if len(rp) >= 4:
            cv2.fillPoly(out, [np.array(rp, np.int32)], 1)
    return out.astype(bool)


def hatch_band(H, shape):
    """Fill the strips between neighbouring parallel hatch lines; returns (mask, hatch gap in px)."""
    d0 = np.array(H[0][1]) - np.array(H[0][0]); nrm = np.array([-d0[1], d0[0]]) / np.hypot(*d0)
    tang = d0 / np.hypot(*d0)
    lines = []
    for a_, b_ in H:                                        # orient consistently, sort by offset
        a_, b_ = np.array(a_), np.array(b_)
        if (b_ - a_) @ tang < 0:
            a_, b_ = b_, a_
        lines.append((((a_ + b_) / 2) @ nrm, a_, b_))
    lines.sort(key=lambda x: x[0])
    merged = []
    for o, a_, b_ in lines:
        if merged and o - merged[-1][0] < 2.0:            # same hatch line, another piece of it
            mo, ma, mb, n = merged[-1]
            ts = [ma @ tang, mb @ tang, a_ @ tang, b_ @ tang]
            oo = (mo * n + o) / (n + 1)
            merged[-1] = (oo, oo * nrm + min(ts) * tang, oo * nrm + max(ts) * tang, n + 1)
        else:
            merged.append((o, a_, b_, 1))
    lines = [(o, a_, b_) for o, a_, b_, _ in merged]
    offs = np.array([l[0] for l in lines]); gaps = np.diff(offs); gaps = gaps[gaps > PX * 0.3]
    gap = float(np.median(gaps)) if len(gaps) else PX * 3
    band = np.zeros(shape, np.uint8)
    for (o1, a1, b1), (o2, a2, b2) in zip(lines, lines[1:]):
        if o2 - o1 < gap * 1.6:                             # neighbours: fill the strip between them
            cv2.fillPoly(band, [cv2.convexHull(np.array([a1, b1, b2, a2], np.int32))], 1)   # hull: never a bow-tie
    for _, a_, b_ in lines:
        cv2.line(band, tuple(int(v) for v in a_), tuple(int(v) for v in b_), 1, 3)
    k = max(3, int(gap * 0.55)) | 1
    return cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), gap


def trace(page):
    core = core_of(page)
    if core is None:
        return None
    r = core['rect']; m = max(r.width, r.height) * 1.25
    win = fitz.Rect(r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m) & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(PX, PX), clip=win, alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(np.int16)
    g = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    black = (g < 90) & (sat < 30)                        # outline and partition lines
    hatch = (g >= 110) & (g <= 175) & (sat < 30)         # grey hatch strokes (0.57 grey)
    coreM = np.zeros_like(black)
    cpts = [(it[1].x, it[1].y) for it in core['items'] if it[0] == 'l']
    to_px = lambda p: ((p[0] - win.x0) * PX, (p[1] - win.y0) * PX)
    core_poly = np.array([to_px(p) for p in cpts], np.int32)
    cv2.fillPoly(coreM.view(np.uint8), [core_poly], 1)
    coreM = coreM.astype(bool) | ((np.abs(g - 191) < 12) & (sat < 30))
    walls = cv2.dilate(black.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # floor outline: the black-line component whose filled interior contains the core centre
    cy, cx = int(core_poly[:, 1].mean()), int(core_poly[:, 0].mean())
    lab, n = ndi.label(walls)
    floor = None
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 50:
            continue
        filled = ndi.binary_fill_holes(comp)
        if filled[cy, cx] and (floor is None or filled.sum() > floor.sum()):
            floor = filled
    if floor is None:
        return None

    # apartment: the area between neighbouring hatch lines (exact vector ends sit on its edge),
    # widened half a hatch gap and cut by walls/core so wall-bounded sides follow the wall exactly
    H = []
    for d in page.get_drawings():
        if d.get('fill') is not None or (d.get('width') or 0) > 1.1 or not d['rect'].intersects(win):
            continue
        for it in d['items']:
            if it[0] == 'l':
                a_, b_ = to_px((it[1].x, it[1].y)), to_px((it[2].x, it[2].y))
                dx, dy = b_[0] - a_[0], b_[1] - a_[1]
                ang = np.degrees(np.arctan2(dy, dx)) % 180          # direction regardless of stored order
                if 25 < ang < 65 or 115 < ang < 155:              # hatch = diagonal strokes (outline is ~0/90 deg)
                    H.append((a_, b_, ang < 90))
    if not H:
        return None
    # some sheets cross-hatch (both diagonals): trace each direction on its own and combine
    groups = [[(a_, b_) for a_, b_, d in H if d == flag] for flag in (True, False)]
    groups = [g for g in groups if len(g) >= 3] or [max(groups, key=len)]
    band = np.zeros(floor.shape, np.uint8)
    gaps_all = []
    for G in groups:
        bnd, gp = hatch_band(G, floor.shape)
        band |= bnd; gaps_all.append(gp)
    gap = float(np.median(gaps_all))
    # black hatch strokes (some sheets) are not walls: take them out of the wall mask before cutting
    hm = np.zeros(floor.shape, np.uint8)
    for G in groups:
        for a_, b_ in G:
            cv2.line(hm, tuple(int(v) for v in a_), tuple(int(v) for v in b_), 1, 7)
    walls = walls & ~hm.astype(bool)
    apt = band.astype(bool) & ~walls & ~coreM                # may extend past the outline (private gardens)
    apt = cv2.morphologyEx(apt.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # straighten: work in the building's own axes, open/close with a square ~ one hatch gap
    fc, _ = cv2.findContours(floor.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ang = cv2.minAreaRect(max(fc, key=cv2.contourArea))[2]
    ang = ((ang + 45) % 90) - 45                            # small rotation off the page axes
    hgt, wid = apt.shape
    M = cv2.getRotationMatrix2D((wid / 2, hgt / 2), ang, 1.0)
    Mi = cv2.invertAffineTransform(M)
    rot = cv2.warpAffine(apt * 255, M, (wid, hgt), flags=cv2.INTER_NEAREST)
    sq = np.ones((int(gap * 0.9) | 1, int(gap * 0.9) | 1), np.uint8)
    rot = cv2.morphologyEx(cv2.morphologyEx(rot, cv2.MORPH_OPEN, sq), cv2.MORPH_CLOSE, sq)
    rot = squared(rot > 127, gap).astype(np.uint8) * 255
    apt = (cv2.warpAffine(rot, Mi, (wid, hgt), flags=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    apt &= (~walls & ~coreM).astype(np.uint8)
    apt = cv2.morphologyEx(apt, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def polys(mask, eps):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cs:
            if cv2.contourArea(c) < (PX * 2) ** 2:
                continue
            ap = cv2.approxPolyDP(c, eps, True)[:, 0, :]
            out.append([(x / PX + win.x0, y / PX + win.y0) for x, y in ap])
        return out
    fl = polys(floor, PX * 0.5)
    return dict(floor=max(fl, key=lambda p: cv2.contourArea(np.array(p, np.float32))),
                core=cpts, apt=polys(apt, PX * 0.7))


def path(pts):
    return 'M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in pts) + 'Z'


def svg_of(t):
    xs = [p[0] for p in t['floor']] + [p[0] for poly in t['apt'] for p in poly]
    ys = [p[1] for p in t['floor']] + [p[1] for poly in t['apt'] for p in poly]
    pad = 2.5
    x0, y0, x1, y1 = min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {y0:.1f} {x1 - x0:.1f} {y1 - y0:.1f}">'
            f'<path d="{path(t["floor"])}" fill="#fff" stroke="#3f3f46" stroke-width="1.3" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>'
            f'<path d="{path(t["core"])}" fill="#d4d4d8" stroke="#a1a1aa" stroke-width="0.4"/>'
            + ''.join(f'<path d="{path(p)}" fill="#e53935" fill-opacity="0.25" stroke="#e53935" stroke-width="2.4" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>' for p in t['apt'])
            + '</svg>')


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(f'{OUT}/*'):
        os.remove(f)
    report = {}
    for pdf in sorted(glob.glob(f'{SRC}/*/*.pdf')):
        name = os.path.relpath(pdf, SRC)[:-4].replace(os.sep, '_')
        t = trace(fitz.open(pdf)[0])
        if not t or not t['apt']:
            print('FAILED', name); report[name] = None; continue
        open(f'{OUT}/{name}.svg', 'w').write(svg_of(t))
        report[name] = dict(floor_pts=len(t['floor']), apt_parts=len(t['apt']), apt_pts=[len(p) for p in t['apt']])
        print(name, report[name])
    json.dump(report, open('tools/keyplans_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
