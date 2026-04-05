import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model():
    print("📋 Starting model evaluation...")
    
    try:
        # Load necessary files
        pt = pd.read_pickle('pt.pkl')
        similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
        print("✅ Model files loaded.")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # 1. Prepare evaluation data
    # Find indices where ratings are non-zero (actual interactions)
    row_idx, col_idx = np.where(pt.values > 0)
    total_interactions = len(row_idx)
    
    if total_interactions == 0:
        print("⚠️ No interactions found in the pivot table to evaluate.")
        return

    # Sample a manageable number of interactions for evaluation (e.g., 500)
    sample_size = min(500, total_interactions)
    indices = np.random.choice(range(total_interactions), sample_size, replace=False)
    
    actual_ratings = []
    predicted_ratings = []

    print(f"🧪 Evaluating on {sample_size} random user-book interactions...")

    for idx in indices:
        r = row_idx[idx]
        c = col_idx[idx]
        
        actual_rating = pt.iloc[r, c]
        
        # Predict rating for book r by user c
        # Rating = sum(similarity * rating) / sum(similarity)
        
        # Get all ratings by this user
        user_ratings = pt.iloc[:, c].values
        
        # Get similarities of this book with all other books
        book_similarities = similarity_scores[r]
        
        # Filter out the book itself and non-rated books
        mask = (user_ratings > 0) & (np.arange(len(user_ratings)) != r)
        
        relevant_similarities = book_similarities[mask]
        relevant_ratings = user_ratings[mask]
        
        if len(relevant_similarities) > 0 and np.sum(np.abs(relevant_similarities)) > 0:
            predicted_rating = np.sum(relevant_similarities * relevant_ratings) / np.sum(np.abs(relevant_similarities))
            
            actual_ratings.append(actual_rating)
            predicted_ratings.append(predicted_rating)

    if len(actual_ratings) > 0:
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        
        # Simple "Accuracy" metric for user friendly display
        # (Assuming max rating is 10, accuracy = 1 - error/10)
        accuracy = max(0, 100 * (1 - mae/10))
        
        print("\n" + "="*30)
        print("📈 EVALUATION RESULTS")
        print("="*30)
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
        print(f"Confidence/Accuracy Score: {accuracy:.2f}%")
        print("="*30)
    else:
        print("⚠️ Could not generate any predictions for evaluation.")

if __name__ == "__main__":
    evaluate_model()
