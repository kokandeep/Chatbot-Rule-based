import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,request,render_template
import chatbot

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat', methods = ['POST'])
def result():
    text = request.form['inpu'].lower()
    exit_list = ['exit', 'see you later', 'bye', 'quit', 'break', 'stop']
    while True:
      #text = request.args.get('inpu')

      if text in exit_list:
          prediction='bye'
          #return render_template('home.html', prediction='Chat with you later, stay safe!!')
          break

      else:

        answer = chatbot.predict(text)
        return render_template("home.html", prediction=answer)


if __name__ =='__main__':
    app.run(debug=True)