import json
import zstandard as zstd
import sys

file = str(sys.argv[1])

with open(file, 'rb') as fh:
    target_file = file.split('.zst')[0] + '.json'
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        previous_line = ""
        while True:
            chunk = reader.read(128000000)  # 256MB chunks
            if not chunk:
                break

            string_data = chunk.decode('utf-8')
            lines = string_data.split("\n")
            for i, line in enumerate(lines[:-1]):
                if i == 0:
                    line = previous_line + line
                object = json.loads(line)

                # do something with the object here
                vars = ['id', 'created_utc', 'subreddit_id', 'title', 'url',
                        'author', 'subreddit', 'domain', 'score', 'num_comments']
                try:
                    post = dict()
                    for var in vars:
                        try:
                            post[var] = object[var]
                        except:
                            pass

                    if 'id' in post.keys():
                        with open(target_file, 'a') as tf:
                            json.dump(post, tf)
                            tf.write('\n')
                    else:
                        print('Missing ID:')
                        print(file)
                        print("Line {}: {}".format(i, line))
                        print("Skipping...\n")
                except:
                    print('Error in:')
                    print(file)
                    print("Line {}: {}".format(i, line))
                    print("Skipping...\n")

            previous_line = lines[-1]