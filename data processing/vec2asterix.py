from asterixdb.asterixdb import AsterixConnection
import io
import argparse
import unicodedata


parser = argparse.ArgumentParser()
parser.add_argument("inp", type=str, help="name of the input file (str)")
parser.add_argument("--init", type=bool, default=False, help="initialize asterix dataset (bool)")
args = parser.parse_args()

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

# Connect to Asterix
print("Connecting to Asterix...")
con = AsterixConnection(server='http://localhost', port=19002)

# Prepare dataset in Asterix
if args.init:
    print("Initializing dataset in Asterix...")
    response = con.query('''
        USE FactMap;

        DROP DATASET fasttext IF EXISTS;
        DROP TYPE EmbeddingType IF EXISTS;

        CREATE TYPE EmbeddingType AS {
            word: string,
            vector: [float]
        };

        CREATE DATASET fasttext(EmbeddingType)
            PRIMARY KEY word;
        ''')


# Load to AsterixDB
print("Pushing to Asterix...")

path = '/Users/ageil/Github/FactMap/Data/fasttext/'
fin = io.open(path + args.inp, 'r', encoding='utf-8', newline='\n', errors='ignore')
outfile = open(path + args.inp.split('.')[0] + '.json', 'a')

for num, line in enumerate(fin):
    tokens = line.rstrip().split(' ')
    vec = [float(x) for x in tokens[1:]]
    w = tokens[0]
    w = remove_control_characters(w)
    w = w.replace('\\', '\\\\')
    w = w.replace('"', '\\"')

    if num % 10000 == 0:
        print(str(num))

    try:
        q = '''USE FactMap;
                
                INSERT INTO fasttext
                ([{{"word":"{0}", "vector":{1}}}])
            '''.format(w, list(vec))
        response = con.query(q)
    except:
        outfile.write('''{{"word":"{0}", "vector":{1}}}\n'''.format(w, list(vec)))

outfile.close()

print("Completed!")