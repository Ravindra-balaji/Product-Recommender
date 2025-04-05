from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained components
df_cleaned = joblib.load("data_cleaned.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
knn = joblib.load("model_knn.pkl")

# Home route - show dropdowns
@app.route('/')
def index():
    categories = sorted(df_cleaned['Category'].unique())
    subcategories = sorted(df_cleaned['Subcategory'].unique())
    brands = sorted(df_cleaned['Brand'].unique())
    return render_template('index.html', categories=categories, subcategories=subcategories, brands=brands)

# KNN Recommendation function
def recommend_products_knn(product_id, top_n=5):
    # Rebuild feature matrix
    text_features = vectorizer.transform(df_cleaned["Combined_Features"])
    num_features = scaler.transform(df_cleaned[["Price", "Product_Rating", "Customer_Review_Sentiment_Score"]])
    final_features = hstack([text_features, num_features])
    final_features = csr_matrix(final_features)

    # Get recommendations
    idx = df_cleaned[df_cleaned["Product_ID"] == product_id].index[0]
    distances, indices = knn.kneighbors(final_features[idx], n_neighbors=top_n + 1)
    recommendations = df_cleaned.iloc[indices[0][1:]][["Product_ID", "Category", "Subcategory", "Brand"]]
    return recommendations

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    category = request.form['category'].strip()
    subcategory = request.form['subcategory'].strip()
    brand = request.form['brand'].strip()

    # Find product matching selected filters
    filtered = df_cleaned[
        (df_cleaned['Category'] == category) &
        (df_cleaned['Subcategory'] == subcategory) &
        (df_cleaned['Brand'] == brand)
    ]

    if filtered.empty:
        return jsonify({'error': 'No matching product found for the selected criteria.'})

    try:
        product_id = filtered.iloc[0]['Product_ID']
        recommendations = recommend_products_knn(product_id).to_dict(orient='records')
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': f"Recommendation error: {str(e)}"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
