# Lossless local grid archives

The completed valid M0 grid is preserved as byte-exact gzip archives. SHA256 of every decompressed stream was verified before removing its uncompressed copy. Published comparison outcomes, replay and checkpoints are unchanged. Archives stay in local runtime.

Use gzip.open(archive, rb) and shutil.copyfileobj into a new original-path file, then verify the original SHA256 from archives.json. Never load the incomplete write as a full grid.
