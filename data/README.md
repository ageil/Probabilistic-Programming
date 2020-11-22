# Data Collection & Processing

Prerequisites:
- A working installation of AsterixDB running at localhost:19002
- Reddit developer credentials
- Google FactCheck Explorer credentials cookie

Steps:
1. Collect ClaimReviews from Google FactCheck Explorer (collect_reviews.py)
2. Download Reddit submissions from https://files.pushshift.io/reddit/submissions/
3. Process downloaded Reddit submissions (StreamZST/XZ/BZ2.py)
4. Create AsterixDB tables, insert Reddit posts + ClaimReviews, join tables (process_asterix.py)
5. Process review ratings (process_ratings.py)
6. Collect corresponding comments via PushShift live API (collect_comments.py)

Note: The resulting datasets can be found in the /data/results folder.
