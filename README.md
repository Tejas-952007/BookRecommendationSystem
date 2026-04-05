# 📚 NexusReads AI - Book Recommendation System

A premium, AI-powered book recommendation engine built using **Collaborative Filtering** and **Flask**. This system analyzes user ratings on over 1.1 million interactions to provide highly accurate and personalized book suggestions with a modern, glassmorphism-themed user interface.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Recommender](https://img.shields.io/badge/AI-Collaborative_Filtering-A855F7?style=for-the-badge)

## 🚀 Features

- **Collaborative Filtering Engine**: Uses Cosine Similarity to find books that are statistically preferred by similar users.
- **Top 50 Popular Books**: Dynamic display of the top-rated books with at least 250 ratings.
- **Dynamic Match Accuracy**: Every recommendation comes with a "Match Percentage" score (Similarity Score) to help users gauge the relevance.
- **Modern UI/UX**: Premium dark-mode interface featuring glassmorphism, floating blobs, and responsive design.
- **Integrated Evaluation**: Built-in script to calculate MAE, RMSE, and model confidence scores.

## 📈 Model Performance Metrics

The model was trained on the Arashnic Book Dataset and achieved the following scores:

| Metric | Score |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | **1.19** |
| **Root Mean Square Error (RMSE)** | **1.62** |
| **Model Accuracy/Confidence** | **88.08%** |

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), Google Fonts (Outfit)
- **Deployment**: Pickle (Model Serialization)

## 📂 Project Structure

```text
├── app.py                # Main Flask Application
├── train.py              # ML Training & Pickling logic
├── evaluate.py           # Logic for Accuracy & Performance testing
├── book-recommender.ipynb # Initial research and EDA
├── templates/            # HTML views (Home & Recommend)
├── static/               # CSS and Assets
└── *.pkl                 # Pre-trained Similarity Models (cached)
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tejas-952007/BookRecommendationSystem.git
   cd BookRecommendationSystem
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional)**:
   Ensure `Books.csv`, `Users.csv`, and `Ratings.csv` are in the folder, then run:
   ```bash
   python train.py
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Tejas-952007/BookRecommendationSystem/issues).

---
Developed with ❤️ by [Tejas-952007](https://github.com/Tejas-952007)
