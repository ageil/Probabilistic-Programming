# Data Collection & Processing

Prerequisites:
- A working installation of AsterixDB running at localhost:19002
- Reddit developer credentials
- Google FactCheck Explorer credentials

Steps:
1. Data collection
    1. Collect ClaimReviews via 1. Data collection (data_collection.py)
    2. Download Reddit submissions from https://files.pushshift.io/reddit/submissions/
2. Data processing
    1. Process downloaded Reddit submissions (StreamZST/XZ/BZ2.py)
    2. Create AsterixDB tables, insert Reddit posts + ClaimReviews, join tables (AsterixDB_data_wrangling.py)
    3. Collect corresponding comments via PushShift live API (comments_api.py)

Note: The resulting datasets can be found in the /data/results folder. 