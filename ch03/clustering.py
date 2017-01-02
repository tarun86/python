import sys
import os
import scipy as sp

from sklearn.feature_extraction.text import CountVectorizer

from utils import DATA_DIR

TOY_DATA = os.path.join(DATA_DIR, "toy")
posts = [open(os.path.join(TOY_DATA, f)).read() for f in os.listdir(TOY_DATA)]

import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

from sklearn.feature_extraction.text import TfidfVectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

# vectorizer = CountVectorizer(min_df=1, stop_words='english',
# preprocessor=stemmer)
vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
X = vectorizer.fit_transform(posts)
print vectorizer.get_feature_names()

tfid_vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')
Y = tfid_vectorizer.fit_transform(posts)
print tfid_vectorizer.get_feature_names()

new_post = ["imaging databases imaging databases"]
nx = vectorizer.transform(new_post)
ny = tfid_vectorizer.transform(new_post)
print nx.toarray()
print ny.toarray()


num_samples, num_features = X.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

dist = dist_norm
best_dist = sys.maxsize
best_i = None

for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X.getrow(i)
    d = dist(post_vec, nx)

    print("=== Post %i with dist=%.2f: %s" % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i
print("Best post is %i with dist=%.2f" % (best_i, best_dist))

dist = dist_norm
best_dist = sys.maxsize
best_i = None
for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = Y.getrow(i)
    d = dist(post_vec, ny)

    print("=== Post %i with dist=%.2f: %s" % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i
print("Best post is %i with dist=%.2f" % (best_i, best_dist))

