"""Download the public BFMD annotations; select singles via the real video index."""
from __future__ import annotations
import argparse
import codecs
import concurrent.futures
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import gdown
import pandas as pd

ROOT_ID = '1wQr4DpMbx-e8jFnvOH9WUJXxfXgTC1IJ'
CATEGORIES = ['metadata', 'court', 'court_perframe', 'player_bbox', 'pose', 'hit_inferred', 'shot_type', 'shuttle']


def folder_listing(folder_id):
    with urllib.request.urlopen('https://drive.google.com/drive/folders/' + folder_id, timeout=90) as response:
        text = response.read().decode('utf-8')
    match = re.search(r"window\['_DRIVE_ivd'\] = '(.*?)';", text)
    if not match:
        raise RuntimeError('Public Drive listing schema changed; do not guess file IDs.')
    # These escaped arrays were inspected in the real public folder response.
    escaped = match.group(1).replace(r'\/', '/')
    data = json.loads(codecs.decode(escaped, 'unicode_escape'))
    return [{'id': row[0], 'name': row[2], 'mime': row[3]} for row in data[0]]


def download(file, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size == 0:
        temporary = destination.with_name(destination.name + '.download')
        try:
            gdown.download(id=file['id'], output=str(temporary), quiet=True, timeout=120, retries=3, use_cookies=False)
        except Exception:
            # Same public download endpoint, without gdown's HTML link parser.
            url = 'https://drive.usercontent.google.com/download?id=' + file['id'] + '&export=download&confirm=t'
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError('Download failed: ' + file['name'])
        temporary.replace(destination)
    if destination.suffix == '.json':
        with destination.open('rb') as stream:
            if stream.read(1) not in (b'{', b'['):
                raise RuntimeError('Downloaded payload is not JSON: ' + str(destination))
    digest = hashlib.sha256()
    with destination.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': digest.hexdigest(), 'drive_id': file['id']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('raw/BFMD'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    root = folder_listing(ROOT_ID)
    for file in root:
        if file['mime'] != 'application/vnd.google-apps.folder':
            download(file, args.root / file['name'])
    index = pd.read_csv(args.root / 'Badminton_video_list.csv')
    names = set(index.loc[index.type.eq('singles'), 'match_name'])
    folders = {f['name']: f['id'] for f in root if f['mime'] == 'application/vnd.google-apps.folder'}
    category_folders = {f['name']: f['id'] for f in folder_listing(folders['annotations'])}
    manifest = {category: folder_listing(category_folders[category]) for category in CATEGORIES}
    (args.root / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    tasks = []
    for category, files in manifest.items():
        for file in files:
            name = file['name']
            match_name = name.removesuffix('_court.json') if category == 'court' else name.removesuffix('.json')
            if name.endswith('.json') and match_name in names:
                tasks.append((file, args.root / 'annotations' / category / name))
    if len(tasks) != len(names) * len(CATEGORIES):
        raise RuntimeError(f'Annotation coverage changed: {len(tasks)} files for {len(names)} singles matches.')
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, file, path): path for file, path in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
                print(f'Downloaded {len(results)}/{len(tasks)}: {futures[future].parent.name}/{futures[future].name}', flush=True)
            except Exception as error:
                failures.append({'path': str(futures[future]), 'error': str(error)})
    (args.root / 'download_audit.json').write_text(json.dumps({'files': results, 'failures': failures}, ensure_ascii=False, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} downloads failed; see download_audit.json.')


if __name__ == '__main__':
    main()
