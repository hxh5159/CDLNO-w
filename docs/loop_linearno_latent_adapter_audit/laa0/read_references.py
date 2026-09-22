"""Archive research texts and explicit source availability, never execute them."""
from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request,subprocess
from bs4 import BeautifulSoup
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
DEST=OUT/'references';DEST.mkdir(exist_ok=True)
old=json.loads((ROOT/'docs/loop_linearno_audit/ll0/reference-ledger.json').read_text())
refs=[(x['name'],x['url'],x['path']) for x in old['papers']]
refs += [('lora','https://arxiv.org/html/2106.09685',None),('relaxed_recursive_v3','https://arxiv.org/html/2410.20672v3',None),('timestep_encoding','https://proceedings.mlr.press/v267/xu25x.html',None),('latent_neural_operator','https://arxiv.org/html/2406.03923v3',None)]

def get(row):
 name,url,local=row
 result={'name':name,'url':url,'local_source':local}
 try:
  if local and Path(local).exists():content=Path(local).read_bytes();result['source']='existing pinned artifact'
  else:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as f:content=f.read()
   result['source']='network'
  suffix='.pdf' if content.startswith(b'%PDF') else '.html';p=DEST/(name+suffix);p.write_bytes(content)
  txt=DEST/(name+'.txt')
  if suffix=='.pdf':subprocess.run(['pdftotext','-layout',str(p),str(txt)],check=True)
  else:txt.write_text(BeautifulSoup(content,'html.parser').get_text(' ',strip=True))
  result.update(status='fetched',path=str(p),text=str(txt),bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
 except Exception as e:result.update(status='unavailable',error=str(e))
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(get,refs))
# Reverify fixed official repository snapshots; include only relevant source inventory.
repos=[]
for repo in old['repositories']:
 if repo['name'] not in ('linearno','attnres','transolver'):continue
 rr={'name':repo['name'],'commit':repo['commit'],'files':[]}
 for r in repo['files']:
  p=Path(r['path'])
  if p.suffix not in ('.py','.md'):continue
  exists=p.exists();entry={'path':str(p),'exists':exists}
  if exists:entry['hash_matches']=hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
  rr['files'].append(entry)
 repos.append(rr)
(OUT/'reference-sources.json').write_text(json.dumps({'papers':rows,'pinned_repositories':repos},indent=2)+'\n')
print(json.dumps(rows,indent=2))
