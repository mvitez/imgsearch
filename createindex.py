import os
from PIL import Image
from sentence_transformers import SentenceTransformer
import argparse
import faiss
import pickle
import numpy as np

parser = argparse.ArgumentParser("catimages")
parser.add_argument('srcdir', help='Images directory')
parser.add_argument('--dstdir', default='.', help='Index directory')
args = parser.parse_args()

model = SentenceTransformer('clip-ViT-B-32')

paths = []
index = faiss.IndexFlat(512, faiss.METRIC_INNER_PRODUCT)
def process(path):
    for filename in os.listdir(path):
        fullpath = path + '/' + filename
        if os.path.isdir(fullpath):
            print('Processing', fullpath)
            process(fullpath)
        else:
            _, ext = os.path.splitext(fullpath)
            if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
                print(fullpath)
                img = Image.open(fullpath)
                emb = model.encode(img)
                vector = np.array([emb], dtype=np.float32)
                vector = vector / np.linalg.norm(vector)
                index.add(vector)
                paths.append(fullpath)
    
process(args.srcdir)
faiss.write_index(index, args.dstdir + '/index.faiss')
with open(args.dstdir + '/index.pkl', 'wb') as f:
    pickle.dump(paths, f)
