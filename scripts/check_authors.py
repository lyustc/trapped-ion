import json, unicodedata, re
from collections import defaultdict, Counter
from src.lit_digest import _generate_preference_from_zotero_records

recs = json.load(open('zotero-export.json', encoding='utf-8'))
pref = _generate_preference_from_zotero_records(recs, top_k_keywords=50)
authors = pref['likes'][0].get('authors', [])
weights = pref['likes'][0].get('author_weights', {})

print('authors_count:', len(authors))
print('top_authors:', authors[:30])

# compute duplicate families
def fam(a):
    s = unicodedata.normalize('NFKD', a.split(',',1)[0])
    return re.sub(r'[^A-Za-z]','',s).lower()

m = defaultdict(list)
for a in authors:
    m[fam(a)].append(a)

dup = {k: v for k, v in m.items() if len(v) > 1}
print('duplicate_families:', len(dup))
for k, v in list(dup.items())[:20]:
    print(k, '->', v)

cnt2 = Counter(weights.values())
print('weights_counts:', dict(cnt2))
print('authors[20:40]:', authors[20:40])
