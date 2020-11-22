import json
import lzma
import sys

file = str(sys.argv[1])

with lzma.open(file, mode="rt") as fh:
    target_file = file.split(".xz")[0] + ".json"

    for i, line in enumerate(fh):
        object = json.loads(line)

        # do something with the object here
        vars = [
            "id",
            "created_utc",
            "subreddit_id",
            "title",
            "url",
            "author",
            "subreddit",
            "domain",
            "score",
            "num_comments",
        ]
        try:
            post = dict()
            for var in vars:
                try:
                    post[var] = object[var]
                except Exception:
                    pass

            if "id" in post.keys():
                with open(target_file, "a") as tf:
                    json.dump(post, tf)
                    tf.write("\n")
            else:
                print("Missing ID:")
                print(file)
                print("Line {}: {}".format(i, line))
                print("Skipping...\n")
        except Exception:
            print("Error in:")
            print(file)
            print("Line {}: {}".format(i, line))
            print("Skipping...\n")
