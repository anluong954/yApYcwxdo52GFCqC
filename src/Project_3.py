
import pandas as pd
import numpy as np
import re
import math
from collections import Counter

# Processing Tools
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# embedding models
from sklearn.feature_extraction.text import TfidfVectorizer #TFiDF
import torchtext #GloVe
# from gensim.models import Word2Vec # Word2Vec
import transformers #BERT and SBERT
import torch

# metrics
from sklearn.metrics.pairwise import cosine_similarity

# Converty URL from google sheet into SCV
# The export URL for CSV is: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={GID}
# Extracted from your URL:
# - Spreadsheet ID: 117X6i53dKiO7w6kuA1g1TpdTlv1173h_dPlJt5cNNMU
# - gid: 113676374

data = "https://docs.google.com/spreadsheets/d/117X6i53dKiO7w6kuA1g1TpdTlv1173h_dPlJt5cNNMU/export?format=csv&gid=113676374"
df =pd.read_csv(data)
print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
df_copy = df.copy()

# Bag of words, Tf idf, word2vec, Glove, FastText, Bert

df_copy['job_title'].value_counts()

abbreviations = {
    'GPHR': 'Global Professional in Human Resources',
    'CSR': 'Corporate Social Responsibility',
    'MES': 'Manufacturing Execution Systems',
    'SPHR': 'Senior Professional in Human Resources',
    'SVP': 'Senior Vice President',
    'GIS': 'Geographic Information System',
    'RRP': 'Reduced Risk Products',
    'CHRO': 'Chief Human Resources Officer',
    'HRIS': 'Human resources information system',
    'HR': 'Human resources',
}
def replace_abbreviations(title):
    for k, v in abbreviations.items():
        regex = r'\b{}\b'.format(re.escape(k))
        title = re.sub(regex, v, title, flags=re.IGNORECASE)
        return title

def clean_title(title):
    title = replace_abbreviations(title) #replace abbreviations
    words = word_tokenize(title.lower()) # tokenize words
    # stemming
    ps = PorterStemmer()
    stems = []
    for words in words:
        stem = ps.stem(words)
        stems.append(stem)

    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_stems = []
    engl_stopwords = stopwords.words('english')
    for stem in stems:
        if stem not in engl_stopwords:
            lemma = lemmatizer.lemmatize(stem)
            lemmatized_stems.append(lemma)
    return '_'.join(lemmatized_stems)

df_copy['job_title'] = df_copy['job_title'].apply(clean_title)

df_copy.head()

df_copy.drop('fit', axis=1, inplace=True)
keywords = ['aspiring human resources']
data_master = df_copy.copy()

# CBOW (Continuous Bag of Words
word = re.compile(r'\w+')
def str_to_vec(str):
    return Counter(word.findall(str))

def get_cosine(v1, v2):
    intersect = set(v1.keys()) & set(v2.keys())
    numerator = sum([v1[i] * v2[i] for i in intersect])
    sum_v1 = sum([v1[j] ** 2 for j in v1.keys()])
    sum_v2 = sum([v2[k] ** 2 for k in v2.keys()])
    denominator = math.sqrt(sum_v1) * math.sqrt(sum_v2)
    if not denominator:
        return 0.0
    else:
        cosine = float(numerator / denominator)
        return cosine

cbow_data = data_master.copy()
cbow_title_embeddings = [str_to_vec(str) for str in cbow_data['job_title']]
cbow_keywords_embeddings = [str_to_vec(str) for str in keywords]

cbow_cosine = [get_cosine(key_emb, title_emb) for key_emb in cbow_keywords_embeddings for title_emb in cbow_title_embeddings]
cbow_data['cbow_fit'] = cbow_cosine

data = df.merge(cbow_data['cbow_fit'], how='left', left_index=True, right_index=True)
data.sort_values('cbow_fit', ascending=False, inplace=True)
data.head(10)

# Vectorizer
vectorizer = TfidfVectorizer()

tfidf_data = data_master.copy()
titles = tfidf_data['job_title'].tolist()

tfidf_title_embs = vectorizer.fit_transform(titles)
tfidf_keyword_embs = vectorizer.transform(keywords)

tfidf_cosine = [cosine_similarity(tfidf_keyword_embs, tfidf_title_emb) for tfidf_title_emb in tfidf_title_embs]
cosine_list = []
for i in tfidf_cosine:
  cosine_list.append(i.item())

tfidf_data['tfidf_fit'] = cosine_list

data = df.merge(tfidf_data['tfidf_fit'], how='left', left_index=True, right_index=True)
data.sort_values('tfidf_fit', ascending=False, inplace=True)
data.head(10)

# GloVe
glove = torchtext.vocab.GloVe(name='6B', dim=100)

def str_to_glove(text):
    # Split by underscore because clean_title joins with '_'
    tokens = text.split('_')
    ind = [glove.stoi[token] for token in tokens if token in glove.stoi]
    if not ind:
        return np.zeros(100)
    vecs = glove.vectors[ind]
    return vecs.numpy().mean(axis=0)

glove_data = data_master.copy()
glove_titles = glove_data['job_title'].apply(str_to_glove)
cleaned_keywords = [clean_title(k) for k in keywords]
glove_title_embeddings = np.stack(glove_titles.values)
glove_keywords_embeddings = str_to_glove(cleaned_keywords[0]).reshape(1, -1)

glove_cosines = cosine_similarity(glove_title_embeddings, glove_keywords_embeddings).flatten()
glove_data['gloVe_fit'] = glove_cosines

final_data = df.merge(glove_data[['gloVe_fit']], left_index=True, right_index=True)
final_data.sort_values('gloVe_fit', ascending=False).head(10)

# BERT
bert = transformers.BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def str_to_bert_embedding(str):
  ids = bert_tokenizer.encode_plus(str, add_special_tokens=True, return_tensors='pt')
  out = bert(**ids)
  embeddings = torch.mean(out.last_hidden_state, dim=1)
  return embeddings

bert_data = data_master.copy()
bert_title_embeddings = [emb.detach().numpy() for emb in bert_data['job_title'].apply(str_to_bert_embedding)]
bert_keywords_embeddings = str_to_bert_embedding(keywords[0]).detach().numpy()

bert_cosine = [cosine_similarity(bert_keywords_embeddings, bert_title_embedding).item() for bert_title_embedding in bert_title_embeddings]
bert_data['bert_fit'] = bert_cosine

data = df.merge(bert_data['bert_fit'], how='left', left_index=True, right_index=True)
data.sort_values('bert_fit', ascending=False, inplace=True)
data.head(10)

# SBERT
sbert = transformers.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sbert_tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def str_to_sbert_embedding(str):
  ids = sbert_tokenizer.encode_plus(str, add_special_tokens=True, return_tensors='pt')
  out = sbert(**ids)
  embeddings = torch.mean(out.last_hidden_state, dim=1)
  return embeddings

sbert_data = data_master.copy()
sbert_title_embeddings = [emb.detach().numpy() for emb in sbert_data['job_title'].apply(str_to_sbert_embedding)]
sbert_keywords_embeddings = str_to_sbert_embedding(keywords[0]).detach().numpy()

sbert_cosine = [cosine_similarity(sbert_keywords_embeddings, sbert_title_embedding).item() for sbert_title_embedding in sbert_title_embeddings]
sbert_data['sbert_fit'] = sbert_cosine

data = df.merge(sbert_data['sbert_fit'], how='left', left_index=True, right_index=True)
data.sort_values('sbert_fit', ascending=False, inplace=True)
data.head(10)

# Best method looks to be BERT
# Reranking by update keyword string
def update_keywords(keywords, candidate_ids, df):
  for i in candidate_ids:
    keywords_join = ' '.join(keywords)
    keywords_l = keywords_join.lower().split()
    job_titles = df.loc[i]['job_title'].lower().split()
    for title in job_titles:
      if title not in keywords_l:
        keywords_l.append(title)
        keywords = ' '.join(keywords_l)
  return keywords

rerank_data = data_master.copy()
candidate_id = rerank_data.index.tolist()[0:10]
updated_keywords = update_keywords(keywords, candidate_id, rerank_data)

bert_updated_keywords_embeddings = str_to_bert_embedding(updated_keywords).detach().numpy()

bert_cosine_reranked = [cosine_similarity(bert_updated_keywords_embeddings, bert_title_embedding).item() for bert_title_embedding in bert_title_embeddings]

rerank_data['rerank_bert_fit'] = bert_cosine_reranked

data = data.merge(rerank_data['rerank_bert_fit'], how='left', left_index=True, right_index=True)
data.sort_values('rerank_bert_fit', ascending=False, inplace=True)
print(data.head(10))

# Reranking via embedding averages
def get_candidate_keywords(candidate_id, df):
  candidate_keywords_l = []
  for i in candidate_id:
    candidate_title = df.loc[i]['job_title'].lower().split()
    for word in candidate_title:
      if word not in candidate_keywords_l:
        candidate_keywords_l.append(word)
  candidate_keywords = ' '.join(candidate_keywords_l)
  return candidate_keywords

def averaged_bert_emb(keywords, candidate_id, df):
  bert_keywords_embeddings = str_to_bert_embedding(keywords)

  candidate_keywords = get_candidate_keywords(candidate_id, df)
  bert_candidate_keywords = str_to_bert_embedding(candidate_keywords)

  avg_bert_emb = (bert_keywords_embeddings + bert_candidate_keywords)/2
  return avg_bert_emb

rerank2_data = data_master.copy()
candidate_id = rerank_data.index.tolist()[0:10]

updated_avg_keywords = averaged_bert_emb(keywords, candidate_id, rerank2_data).detach().numpy()

bert_cosine_reranked_2 = [cosine_similarity(updated_avg_keywords, bert_title_embedding).item() for bert_title_embedding in bert_title_embeddings]

rerank2_data['rerank2_bert_fit'] = bert_cosine_reranked_2

data = data.merge(rerank2_data['rerank2_bert_fit'], how='left', left_index=True, right_index=True)
data.sort_values('rerank2_bert_fit', ascending=False, inplace=True)
print(data.head(10))
