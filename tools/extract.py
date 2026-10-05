"""Extract wall geometry + floor texture from Grofit plan PDFs into viewer JS files."""
import base64, glob, io, json, os, re, sys
import numpy as np
import pymupdf as fitz
from PIL import Image

SRC = 'plans'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
CELL_M = 0.02          # wall raster resolution (2 cm)
TEX_PX_PER_M = 110     # floor texture resolution

CONCRETE, BLOCK = 0.5, 0.75


def replay(shape, items):
    for it in items:
        k = it[0]
        if k == 'l':
            shape.draw_line(it[1], it[2])
        elif k == 're':
            shape.draw_rect(it[1])
        elif k == 'qu':
            shape.draw_quad(it[1])
        elif k == 'c':
            shape.draw_bezier(it[1], it[2], it[3], it[4])


def boxes_from_mask(mask):
    """Greedy merge of horizontal runs into rectangles. Returns [x0, y0, x1, y1] in cells."""
    h, w = mask.shape
    active, out = {}, []
    for y in range(h + 1):
        runs = set()
        if y < h:
            row = mask[y]
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            runs = set(zip(starts.tolist(), ends.tolist()))
        nxt = {}
        for r, y0 in active.items():
            if r in runs:
                nxt[r] = y0
            else:
                out.append([r[0], y0, r[1], y])
        for r in runs:
            if r not in nxt:
                nxt[r] = y
        active = nxt
    return out


def scale_of(page):
    m = re.search(r'1\s*:\s*(\d+)', page.get_text())
    return int(m.group(1)) if m else 50


def process(path):
    key = os.path.basename(path)[:-4]
    page = fitz.open(path)[0]
    W, H = page.rect.width, page.rect.height
    scale = scale_of(page)
    m_per_pt = 0.0254 / 72 * scale

    paths = [d for d in page.get_drawings()
             if d.get('fill') is not None and d['rect'].x1 < W * 0.655 and d['rect'].y0 > H * 0.22]
    walls = [d for d in paths if round(d['fill'][0], 2) in (CONCRETE, BLOCK)]
    bb = fitz.Rect(walls[0]['rect'])
    for d in walls:
        bb |= d['rect']
    bb = fitz.Rect(bb.x0 - 4, bb.y0 - 4, bb.x1 + 4, bb.y1 + 4)

    # Clean page with only wall fills (red = concrete, green = block), white fills erase (window openings).
    doc = fitz.open()
    pg = doc.new_page(width=W, height=H)
    for d in paths:
        c = round(d['fill'][0], 2)
        col = {CONCRETE: (1, 0, 0), BLOCK: (0, 1, 0), 1.0: (1, 1, 1)}.get(c)
        if col is None:
            continue
        sh = pg.new_shape()
        replay(sh, d['items'])
        sh.finish(fill=col, color=None, closePath=d.get('closePath', True), even_odd=d.get('even_odd', False))
        sh.commit()
    zoom = m_per_pt / CELL_M
    pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bb, alpha=False)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(int)
    conc = (a[:, :, 0] > 170) & (a[:, :, 1] < 110) & (a[:, :, 2] < 110)
    blk = (a[:, :, 1] > 170) & (a[:, :, 0] < 110) & (a[:, :, 2] < 110)

    def to_m(bs):
        return [[round(x0 * CELL_M, 3), round(y0 * CELL_M, 3), round(x1 * CELL_M, 3), round(y1 * CELL_M, 3)]
                for x0, y0, x1, y1 in bs if (x1 - x0) * (y1 - y0) >= 4]

    # Floor texture: the original plan cropped to the same box.
    tz = TEX_PX_PER_M * m_per_pt
    tex = page.get_pixmap(matrix=fitz.Matrix(tz, tz), clip=bb, alpha=False)
    img = Image.frombytes('RGB', (tex.width, tex.height), tex.samples)
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=62, optimize=True)

    model = {
        'key': key, 'scale': scale,
        'w': round(bb.width * m_per_pt, 3), 'd': round(bb.height * m_per_pt, 3),
        'concrete': to_m(boxes_from_mask(conc)), 'block': to_m(boxes_from_mask(blk)),
        'floor': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(),
    }
    with open(os.path.join(OUT, key + '.js'), 'w') as f:
        f.write('(window.MODELS=window.MODELS||{})[%s]=%s;\n' % (json.dumps(key), json.dumps(model, separators=(',', ':'))))
    print(key, 'scale 1:%d' % scale, '%.1fx%.1fm' % (model['w'], model['d']),
          'concrete', len(model['concrete']), 'block', len(model['block']), 'tex %dKB' % (len(buf.getvalue()) // 1024))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(SRC, '*.pdf'))):
        if '-' in os.path.basename(p):   # per-apartment variants; the base type plan is used
            continue
        process(p)
