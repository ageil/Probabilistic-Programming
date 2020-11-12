# Guide

Manage Reddit submissions data:

1. Download raw data from pushshift.io
2. Stream compressed data line by line, extracting only necessary fields via helper files

Manage fast text word vectors:

1. Download fasttext word vectors
2. Split file into 8 chunks using `split -l 250000 crawl_300d_2M.vec crawl_300d_2M` in bash/zsh
3. Run vec2asterix.py on each of the split files in parallel (enable --init on first run)

Note: Cannot bulk load word vectors due to illegal words used in vectors.

