# Sports Prediction Model

## Overview
This project uses historical game data to predict the outcome (Win/Loss) of sports games. A Random Forest Classifier is trained on features like team performance, points scored, and venue type, with the model evaluated for accuracy and feature importance.

## Features
- Data preprocessing (encoding categorical data)
- Random Forest Classifier for predictions
- Model evaluation (accuracy, classification report, confusion matrix)
- Feature importance analysis
- Model saving for future use

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation
1. Clone the repository:
git clone https://github.com/your-username/sports-prediction-model.git

2. Install dependencies:
pip install -r requirements.txt

## Dataset
The dataset (`sports_data_mock.csv`) includes columns like `HomeTeam`, `AwayTeam`, `HomeTeamPoints`, `AwayTeamPoints`, `GameResult`, and more.

## Running the Project
1. Load and explore data.
2. Preprocess data (encode categorical variables, transform target).
3. Train a Random Forest model and evaluate its performance.
4. Save the trained model as `sports_winner_prediction_model.pkl`.

## Example Output
- **Model Accuracy**: e.g., `80%`
- **Classification Report**: Precision, recall, F1-score for Win/Loss.
- **Confusion Matrix**: True/False positives and negatives.
- **Feature Importance**: Table and bar chart showing feature influence.

## Future Improvements
- Experiment with different models.
- Add more features (e.g., player stats, weather).
- Hyperparameter tuning.

## License
Apache License 2.0
