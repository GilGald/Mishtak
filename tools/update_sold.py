"""Refresh the 'sold' flags from the developer's price list and save them locally.

Run from the site folder:  python3 tools/update_sold.py
Downloads https://mishtaken.grofit.co.il/ once, saves a dated copy in source/, reads which rows carry the
'sale-property-strip' class (sold), and updates data.json and the SOLD list in index.html.
"""
import datetime, json, re, sys, urllib.request

URL = 'https://mishtaken.grofit.co.il/'
today = datetime.date.today()
label = today.strftime('%-d %b %Y')
import time
req = urllib.request.Request(f'{URL}?nocache={int(time.time())}',      # bypass any page cache on the site
                             headers={'User-Agent': 'Mozilla/5.0', 'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})
html = urllib.request.urlopen(req, timeout=60).read().decode('utf-8')
open(f'source/mishtaken-price-list-{today.isoformat()}.html', 'w', encoding='utf-8').write(html)

rows = list(re.finditer(r'<tr class="([^"]*)" data-rooms="[^"]*" data-price="([^"]*)"[^>]*>(.*?)</tr>', html, re.S))
if len(rows) < 200:
    sys.exit(f'Only {len(rows)} apartment rows found: the site may have changed. Nothing updated.')
sold, prices = [], {}
for m in rows:
    c = [re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', x)).strip() for x in re.findall(r'<td[^>]*>(.*?)</td>', m.group(3), re.S)]
    k = c[1] + '|' + c[0]
    prices[k] = int(m.group(2)) or None
    if 'sale-property-strip' in m.group(1):
        sold.append(k)

data = json.load(open('data.json'))
key = lambda a: a['bld'] + '|' + str(a['apt'])
old = {key(a) for a in data if a.get('sold')}
new = set(sold)
price_changes = [(key(a), a['price'], prices.get(key(a))) for a in data if prices.get(key(a)) != a['price']]
for a in data:
    a['sold'] = key(a) in new
json.dump(data, open('data.json', 'w'), ensure_ascii=False)

t = open('index.html', encoding='utf-8').read()
t = re.sub(r"const SOLD_AS_OF = '[^']*';", f"const SOLD_AS_OF = '{label}';", t)
t = re.sub(r"const SOLD = new Set\(\[.*?\]\);", 'const SOLD = new Set(' + json.dumps(sold, ensure_ascii=False) + ');', t, flags=re.S)
t = re.sub(r"saved \d+ \w+ \d{4}\nconst SOLD_AS_OF", f"saved {label}\nconst SOLD_AS_OF", t)
t = re.sub(r"<label>Sold \(developer site, [^)]*\)</label>", f"<label>Sold (developer site, {label})</label>", t)
open('index.html', 'w', encoding='utf-8').write(t)

order = lambda k: (k.split('|')[0], int(k.split('|')[1]))
print(f'{today}: {len(rows)} apartments, sold now {len(new)} (was {len(old)})')
print('newly sold:', sorted(new - old, key=order))
print('no longer sold:', sorted(old - new, key=order))
print('price changes:', price_changes)
