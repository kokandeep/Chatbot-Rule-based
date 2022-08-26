import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def predict (words):

      df_1 = pd.read_csv('updated_required_csv_file.txt', header=None, delimiter='\t')

      vec = TfidfVectorizer(lowercase=True)
      X = vec.fit_transform(df_1[0])
      question = X.toarray()

      exit_list = ['exit', 'see you later', 'bye', 'quit', 'break', 'stop']

      text_vec = vec.transform([words])
      text_vec = text_vec.toarray()
      text_vec = text_vec[0].reshape(1, -1)
      similarity = cosine_similarity(text_vec, question)
      answer_index = np.argmax(similarity[0])
      answers = df_1[1].iloc[answer_index]
      # print('Moogle: ', answer)
      return (answers)