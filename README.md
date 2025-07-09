SMS Spam Classification: Traditional Machine Learning Approach
This project demonstrates a robust pipeline for classifying SMS messages as "spam" or "ham" (not spam) using traditional machine learning techniques. It showcases essential steps in a machine learning workflow, including data preparation, text preprocessing, feature engineering, model training, evaluation, and optimization.

Table of Contents
Project Goal

Dataset

Features

Methodology

Installation

Usage

Results

Conclusion & Real-World Applications

Future Work

Contributing

License

Acknowledgements

Project Goal
The primary goal of this project is to build a high-performance supervised machine learning model capable of accurately classifying SMS messages. The focus is on leveraging classic machine learning algorithms and text processing techniques, without resorting to deep learning, to achieve optimal accuracy.

Dataset
The project utilizes the "SMS Spam Collection Dataset" available on Kaggle. This dataset comprises a collection of SMS messages tagged as either "ham" (legitimate) or "spam".

Data Overview
Source: UCI Machine Learning Repository / Kaggle

Format: CSV

Columns: v1 (label: 'ham' or 'spam'), v2 (message text), and several unnamed columns which were dropped during preprocessing.

Label Distribution: The dataset is imbalanced, with a significantly higher number of "ham" messages compared to "spam" messages.

Features
TF-IDF (Term Frequency-Inverse Document Frequency): This technique was used to convert the text messages into a numerical representation (vectors) that machine learning models can understand. TF-IDF assigns a weight to each word, indicating its importance in a document relative to the entire corpus.

Methodology
The project follows a structured machine learning pipeline:

Environment Setup: Import necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, nltk).

Dataset Preparation:

Load the spam.csv dataset.

Clean the data by dropping irrelevant columns and renaming the primary columns to label and message.

Convert text labels ("ham", "spam") to numerical labels (0, 1) for model training.

Perform initial data exploration to understand its structure and class distribution.

Text Preprocessing:

Convert all message text to lowercase.

Remove punctuation.

Tokenize messages into individual words.

Remove common English stopwords (e.g., "the", "a", "is") using NLTK's stopwords list.

Apply Porter Stemming to reduce words to their root form (e.g., "running", "runs" -> "run").

Feature Extraction:

Utilize TfidfVectorizer from scikit-learn to transform the preprocessed text messages into numerical TF-IDF features. max_features was set to 5000 to limit the dimensionality.

Data Splitting:

Split the dataset into training and testing sets (80% training, 20% testing) using train_test_split with stratify=y to maintain the original class distribution in both sets.

Model Training and Evaluation:

Multinomial Naive Bayes: A probabilistic classifier often effective for text classification.

Logistic Regression: A linear model widely used for binary classification.

Support Vector Machine (LinearSVC): A powerful linear classifier known for its effectiveness in high-dimensional spaces.

Each model was trained on the X_train (TF-IDF features) and y_train (encoded labels).

Model performance was evaluated using:

Accuracy: Overall correctness of predictions.

Precision: Proportion of correctly predicted positive observations out of all positive predictions.

Recall (Sensitivity): Proportion of correctly predicted positive observations out of all actual positives.

F1-Score: Harmonic mean of Precision and Recall.

Confusion Matrix: Visual representation of true positives, true negatives, false positives, and false negatives.

(Optional) Optimization:

Hyperparameter Tuning with GridSearchCV: GridSearchCV was applied to LinearSVC and LogisticRegression to find the best hyperparameters (e.g., C for regularization, solver for LR) that maximize accuracy using cross-validation.

VotingClassifier: An ensemble method to combine the predictions of multiple models (Multinomial Naive Bayes, Logistic Regression, LinearSVC) using 'hard' voting. This can sometimes lead to more robust predictions.

Installation
To run this project, you'll need Python and the following libraries. You can install them using pip:

Bash

pip install numpy pandas scikit-learn matplotlib seaborn nltk
You will also need to download NLTK's stopwords and punkt tokenizer data:

Python

import nltk
nltk.download('stopwords')
nltk.download('punkt')
Usage
Download the dataset:
Download the spam.csv file from the Kaggle SMS Spam Collection Dataset and place it in your project directory.
(Note: The provided notebook assumes the dataset is provided as SMS Spam Classification.zip and extracts spam.csv from it. Ensure the path is correct or directly use spam.csv).

Run the Jupyter Notebook:
You can run the SMS_Spam_Classification.ipynb notebook in Google Colab, Kaggle Notebooks, or any local Jupyter environment.

Google Colab: Upload the SMS_Spam_Classification.ipynb file to your Google Drive and open it with Colab. Upload the SMS Spam Classification.zip file to your Colab environment or adjust the data loading path to access it from Kaggle.

Kaggle Notebooks: Create a new notebook on Kaggle, add the "SMS Spam Collection Dataset" as an input, and then upload/copy the code from SMS_Spam_Classification.ipynb into your Kaggle notebook.

Local Jupyter: Ensure all dependencies are installed, then run jupyter notebook in your terminal and navigate to the SMS_Spam_Classification.ipynb file.

Execute cells sequentially:
Run all cells in the notebook from top to bottom. The output of each step, including data exploration, preprocessing examples, model performance metrics, and visualizations, will be displayed.

Results
After running the notebook, you will observe the performance metrics (Accuracy, Precision, Recall, F1-Score) and confusion matrices for each trained model: Multinomial Naive Bayes, Logistic Regression, and LinearSVC (both before and after hyperparameter tuning), and the VotingClassifier.

A performance comparison table and bar plot will summarize the results, clearly indicating which model performed best based on accuracy and other metrics.

Example Performance Summary (Your exact results may vary slightly):

--- Model Performance Comparison (Including Optimized Models) ---
                               Accuracy  Precision    Recall  F1-Score
Optimized LinearSVC            0.988341   0.992754  0.919463  0.954704
LinearSVC                      0.987444   1.000000  0.906040  0.950704
Optimized Logistic Regression  0.983857   1.000000  0.879195  0.935714
VotingClassifier (Hard)        0.973991   1.000000  0.805369  0.892193
Multinomial Naive Bayes        0.966816   0.991228  0.758389  0.859316
Logistic Regression            0.957848   0.990385  0.691275  0.814229
The LinearSVC model, especially after hyperparameter tuning, generally achieves the highest accuracy and F1-score, demonstrating its strong capability in handling high-dimensional sparse data like TF-IDF features for text classification.

Conclusion & Real-World Applications
This project successfully demonstrates the effectiveness of traditional machine learning models combined with robust text preprocessing for SMS spam classification.

Key Insights:

Text Preprocessing is Key: Cleaning text, removing stopwords, and stemming are crucial steps that significantly improve model performance by reducing noise and standardizing the text data.

TF-IDF Effectiveness: TF-IDF proved to be a highly effective feature engineering technique for converting raw text into meaningful numerical features that capture word importance within the corpus.

Linear Models Shine: LinearSVC and Logistic Regression performed exceptionally well, which is common for text classification tasks where linear separability in a high-dimensional feature space is often present.

Hyperparameter Tuning Matters: Optimizing model hyperparameters (e.g., C in SVC/Logistic Regression) further boosted performance, highlighting the importance of this step.

Ensemble Potential: While the individual optimized SVC was very strong, ensemble methods like VotingClassifier can offer marginal improvements or increased robustness in some scenarios.

This approach can be applied in various real-world spam detection systems, including:

Email Spam Filters: Filtering unwanted emails from user inboxes.

SMS Spam Blocking: Implementing on mobile devices or carrier networks to prevent spam messages from reaching users.

Online Content Moderation: Identifying and flagging spammy comments, reviews, or forum posts on websites.

Phishing Detection: Recognizing patterns in messages that indicate phishing attempts.

Future Work
Advanced Text Preprocessing: Explore lemmatization, N-grams (beyond what TfidfVectorizer implicitly handles), and more sophisticated text normalization techniques.

Feature Engineering: Incorporate additional features like message length, presence of URLs, specific character patterns, or emojis.

Deep Learning Models: Investigate deep learning approaches such as Recurrent Neural Networks (RNNs) like LSTMs or GRUs, and Transformer-based models (e.g., BERT) for potentially higher accuracy, especially on larger and more complex datasets.

Imbalanced Data Handling: Implement advanced techniques for imbalanced datasets, such as SMOTE (Synthetic Minority Over-sampling Technique) or different cost-sensitive learning algorithms, to potentially improve recall for the minority class (spam).

Model Deployment: Develop a basic API or web application to demonstrate real-time spam classification.

Contributing
Feel free to fork this repository, make improvements, and submit pull requests.

License
This project is open-source and available under the MIT License.

Acknowledgements
UCI Machine Learning Repository for providing the SMS Spam Collection Dataset.

scikit-learn and NLTK communities for their invaluable libraries.

