import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train():
    print("🚀 Starting training process...")
    
    # 1. Load Data
    print("📥 Loading CSV files...")
    try:
        books = pd.read_csv('Books.csv', low_memory=False)
        users = pd.read_csv('Users.csv', low_memory=False)
        ratings = pd.read_csv('Ratings.csv', low_memory=False)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure Books.csv, Users.csv, and Ratings.csv are in the current directory.")
        return

    print(f"✅ Data loaded: {len(books)} books, {len(users)} users, {len(ratings)} ratings.")

    # 2. Preprocessing & Merging
    print("🛠️ Merging datasets...")
    ratings_with_name = ratings.merge(books, on='ISBN')

    # 3. Popularity Based Recommender System (Top 50)
    print("📊 Generating Popularity-based Top 50...")
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

    avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    
    # Filter books with more than 250 ratings and sort by average rating
    popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
    
    # Add Author and Image URL to the Top 50
    popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

    # 4. Collaborative Filtering Based Recommender System
    print("🧠 Building Collaborative Filtering model...")
    # Filter for active users (more than 200 ratings given)
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    active_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]

    # Filter for famous books (at least 50 ratings received)
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

    # Create Pivot Table
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)

    # Compute Similarity Scores
    print("📐 Computing Cosine Similarity scores...")
    similarity_scores = cosine_similarity(pt)

    # 5. Exporting Model Files
    print("💾 Exporting pickle files...")
    pickle.dump(popular_df, open('popular.pkl', 'wb'))
    pickle.dump(pt, open('pt.pkl', 'wb'))
    pickle.dump(books, open('books.pkl', 'wb'))
    pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))

    print("✨ Training complete! All model files generated successfully.")
    
    # 6. Accuracy Evaluation
    print("\n🧪 Evaluating model accuracy...")
    try:
        # Evaluate on random samples from the pivot table
        row_idx, col_idx = np.where(pt.values > 0)
        total_interactions = len(row_idx)
        sample_size = min(500, total_interactions)
        indices = np.random.choice(range(total_interactions), sample_size, replace=False)
        
        actual_ratings = []
        predicted_ratings = []

        for idx in indices:
            r, c = row_idx[idx], col_idx[idx]
            actual_rating = pt.iloc[r, c]
            
            user_ratings = pt.iloc[:, c].values
            book_similarities = similarity_scores[r]
            
            mask = (user_ratings > 0) & (np.arange(len(user_ratings)) != r)
            rel_sims, rel_ratings = book_similarities[mask], user_ratings[mask]
            
            if len(rel_sims) > 0 and np.sum(np.abs(rel_sims)) > 0:
                pred = np.sum(rel_sims * rel_ratings) / np.sum(np.abs(rel_sims))
                actual_ratings.append(actual_rating)
                predicted_ratings.append(pred)

        if len(actual_ratings) > 0:
            mae = mean_absolute_error(actual_ratings, predicted_ratings)
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            accuracy = max(0, 100 * (1 - mae/10))
            
            print("\n" + "="*35)
            print("📈 FINAL TRAINING REPORT")
            print("="*35)
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
            print(f"Confidence/Accuracy Score: {accuracy:.2f}%")
            print("="*35)
    except Exception as e:
        print(f"⚠️ Could not complete evaluation: {e}")

if __name__ == '__main__':
    train()
