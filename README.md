# 📊 Grade Forecaster

A Streamlit web app that predicts a student's final maths grade (G3) from their profile.

- **Data:** UCI Student Performance (maths), `student-mat.csv`
- **Model:** best of Ridge / Random Forest / Gradient Boosting (chosen by cross-validation)
- **App:** `app.py` (Streamlit)

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy
Push to GitHub, then create an app on https://share.streamlit.io pointing at `app.py`.
