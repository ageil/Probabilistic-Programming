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
                post = dict()
                post['id'] = object['id']
                post['created_utc'] = object['created_utc']
                post['subreddit_id'] = object['subreddit_id']
                post['title'] = object['title']
                post['url'] = object['url']
                post['author'] = object['author']
                post['subreddit'] = object['subreddit']
                post['domain'] = object['domain']
                post['score'] = object['score']
                post['num_comments'] = object['num_comments']

                with open(target_file, 'a') as tf:
                    json.dump(post, tf)
                    tf.write('\n')

            previous_line = lines[-1]