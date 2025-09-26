# Smart Resume Screener

This project is a Streamlit-based application for automated resume screening using NLP and machine learning. It compares a candidate's resume with a job description (JD) and computes a fit score based on semantic similarity, skill overlap, and years of experience.

## Features

- Upload resume (PDF or text) and paste job description
- Extracts skills from resume and JD using an internal skill list
- Computes semantic similarity using sentence-transformer embeddings
- Calculates skill overlap and experience match
- Predicts resume profile category using a trained classifier
- Provides suggestions for improving resume fit

## Files

- `streamlit_app.py`: Main Streamlit application
- `Resume_Screening.ipynb`: Jupyter notebook for model training and data exploration
- `clf.pkl`, `tfidf.pkl`, `encoder.pkl`: Pre-trained model and encoders
- `requirements.txt`: Python dependencies
- `clf.pkl Download Link`: https://www.kaggle.com/datasets/noorsaeed/resume-trained-save-model


## Getting Started

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:
    ```sh
    streamlit run streamlit_app.py
    ```

3. Open the app in your browser, upload a resume and paste a job description to compute the fit score.

## Model Training

See [`Resume_Screening.ipynb`](Resume_Screening.ipynb) for data preprocessing, feature extraction, and model training steps.