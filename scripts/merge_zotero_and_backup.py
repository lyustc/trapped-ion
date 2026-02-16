import shutil, json, time, os
from src.lit_digest import update_preferences_from_zotero_export

ZOTERO_EXPORT = 'zotero-export.json'
HISTORY = 'preferences.json'

timestamp = time.strftime('%Y%m%dT%H%M%S')
backup_name = f'preferences.json.bak.{timestamp}'

if os.path.exists(HISTORY):
    shutil.copy2(HISTORY, backup_name)
    print('backed up', HISTORY, '->', backup_name)
else:
    print('no existing', HISTORY, 'found; skipping backup')

# perform merge (this will write HISTORY)
try:
    update_preferences_from_zotero_export(ZOTERO_EXPORT, history_path=HISTORY, top_k_keywords=50)
    print('merge completed into', HISTORY)
except Exception as e:
    print('merge failed:', e)
    raise

# report summary
with open(HISTORY, 'r', encoding='utf-8') as f:
    data = json.load(f)
likes = data.get('likes', [])
if not likes:
    print('no likes block in', HISTORY)
else:
    primary = likes[0]
    kws = primary.get('keywords', [])
    authors = primary.get('authors', [])
    aw = primary.get('author_weights', {})
    print('keywords_count:', len(kws))
    print('authors_count:', len(authors))
    print('author_weights_top_counts:', sum(1 for v in aw.values() if v==2.0), sum(1 for v in aw.values() if v==1.0))
    print('sample_authors_top20:', authors[:20])

# also write a copy preferences.test_postmerge.json for review
shutil.copy2(HISTORY, 'preferences.test_postmerge.json')
print('wrote preferences.test_postmerge.json')
