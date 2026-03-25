# services/text_processor.py
import pandas as pd
import numpy as np
import re
import random
import spacy
import yake
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from collections import Counter

# Загружаем модель spaCy один раз
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    print("Модель de_core_news_sm не найдена. Установите её командой: python -m spacy download de_core_news_sm")
    raise

def find_snippet_in_book(book_path, first_five, last_five):
    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()

    first_pos = text.find(first_five)
    if first_pos == -1:
        return None

    last_pos = text.find(last_five, first_pos)
    if last_pos == -1:
        return None

    return text[first_pos:last_pos + len(last_five)]


def read_file(file_name):
  with open(file_name, "r", encoding="utf-8") as f:
    file = f.read().replace("\n", " ").lower()
  text = re.sub(r"[^\w\s]", "", file)
  return text

def lemmatization(text):
  spacy_text = nlp(text)
  no_sw = [token.text for token in spacy_text if not (token.is_stop or token.is_space)]
  filtered = nlp(" ".join(no_sw))
  lemmas_pos = []
  for token in filtered:
      lemmas_pos.append((token.lemma_.lower(), token.pos_))
  lemmas_pos_df = pd.DataFrame(lemmas_pos, columns=["lemma", "pos"])

  return lemmas_pos_df  

def diff_pos(lemmas_pos_df, pos):
  pos_df = lemmas_pos_df[lemmas_pos_df["pos"] == pos]
  return pos_df

def to_text(df):
    text = " ".join(token for token in df["lemma"])
    return text

def extract_kws(text):
  kw_extractor = yake.KeywordExtractor(lan="de",n=1, top=3, dedupLim=0.7, dedupFunc='seqm')
  keywords = pd.DataFrame(kw_extractor.extract_keywords(text))
  return list(keywords[0])

def extract_unique_kws(text, n_to_extract, n_result):
    kw_extractor = yake.KeywordExtractor(lan="de",n=1, top=n_to_extract, dedupLim=0.7,
                                         dedupFunc='seqm')
    keywords = list(pd.DataFrame(kw_extractor.extract_keywords(text))[0])

    total = len(keywords)
    start = int(total * 0.5)
    end = int(total * 0.7)
    mid_words = keywords[start:end]
    return random.sample(mid_words, min(n_result, len(mid_words)))

def final_wordlist(text):
  text = read_file("amoklaufer.txt")
  lemmas_pos_df = lemmatization(text)

  verbs_df = diff_pos(lemmas_pos_df, "VERB")
  nouns_df = diff_pos(lemmas_pos_df, "NOUN")

  verbs_text = to_text(verbs_df)
  nouns_text = to_text(nouns_df)

  key_verbs = extract_kws(verbs_text)
  key_nouns = extract_kws(nouns_text)

  unique_verbs = extract_unique_kws(verbs_text, 500, 3)
  unique_nouns = extract_unique_kws(nouns_text, 500, 3)

  result_verbs = key_verbs + unique_verbs
  result_nouns = key_nouns + unique_nouns

  return result_verbs, result_nouns

model_path = hf_hub_download(
    repo_id="Word2vec/german_model",
    filename="german.model")
similarity_model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True,
    unicode_errors="ignore")

def related_words(word, n=3):
  similar_words = pd.DataFrame(similarity_model.most_similar(word, topn=n))[0]
  return list(similar_words)
