from flask import Flask,render_template,request
import pickle
import numpy as np

import pandas as pd

popular_df = pd.read_pickle('popular.pkl')
pt = pd.read_pickle('pt.pkl')
books = pd.read_pickle('books.pkl')
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input').strip()
    
    # Check if book exists in index (case-insensitive)
    lower_index = [title.lower() for title in pt.index]
    if user_input.lower() not in lower_index:
        return render_template('recommend.html', data=None, user_input=user_input)
        
    idx = lower_index.index(user_input.lower())
    similar_items = sorted(list(enumerate(similarity_scores[idx])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = []
        # Get the original title for display
        original_title = pt.index[i[0]]
        temp_df = books[books['Book-Title'] == original_title]
        
        # Calculate similarity percentage (multiplied by 100 for percentage display)
        similarity_percent = round(i[1] * 100, 2)
        
        # Clean unique info
        clean_df = temp_df.drop_duplicates('Book-Title')
        
        item.append(clean_df['Book-Title'].values[0])
        item.append(clean_df['Book-Author'].values[0])
        item.append(clean_df['Image-URL-M'].values[0])
        item.append(similarity_percent)  # Added similarity score as the 4th element

        data.append(item)

    return render_template('recommend.html', data=data, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

    