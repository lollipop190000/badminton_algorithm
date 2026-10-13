"""Download both public ShuttleSet trees at the recorded source commit."""
import hashlib,json,urllib.request
from pathlib import Path
from urllib.parse import quote

REPO='wywyWang/CoachAI-Projects'
REF='45517f7d4cb936b03f3eabf939cc7959d39226fe'
PREFIXES={'ShuttleSet':'ShuttleSet/','ShuttleSet22':'CoachAI-Challenge-IJCAI2023/ShuttleSet22/'}


def main():
 req=urllib.request.Request(f'https://api.github.com/repos/{REPO}/git/trees/{REF}?recursive=1',headers={'User-Agent':'badminton-research'})
 with urllib.request.urlopen(req,timeout=60) as r:tree=json.load(r)
 if tree.get('truncated'):raise ValueError('Upstream tree is truncated; cannot guarantee full source coverage')
 for name,prefix in PREFIXES.items():
  root=Path('raw')/name;root.mkdir(parents=True,exist_ok=True);manifest=[]
  files=[f for f in tree['tree'] if f['type']=='blob' and f['path'].startswith(prefix)]
  if not files:raise ValueError('Verified source prefix missing: '+prefix)
  for f in files:
   rel=f['path'][len(prefix):]
   if '..' in Path(rel).parts:raise ValueError('Invalid upstream relative path')
   dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():
    url=f'https://raw.githubusercontent.com/{REPO}/{REF}/'+quote(f['path'],safe='/')
    with urllib.request.urlopen(url,timeout=60) as r:dest.write_bytes(r.read())
   data=dest.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
   if blob!=f['sha']:raise ValueError('Source byte hash mismatch: '+rel)
   manifest.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha':blob,'commit':REF})
  (root/'download_audit.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
  print(name,len(manifest),'files, verified pinned Git blob hashes')


if __name__=='__main__':main()
