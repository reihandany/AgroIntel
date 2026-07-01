import warnings
import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.exceptions import InconsistentVersionWarning
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# DOWNLOAD NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# LOAD MODEL & DATA
try:
    with open('model_dt_tfidf1.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

try:
    with open('tfidf_vectorizer_1gram.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    vectorizer = None

try:
    df_sample = pd.read_csv('final_preprocessed_data.csv')
except FileNotFoundError:
    df_sample = None

# PREPROCESSING
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):

    text = re.sub(r'[^a-zA-Z\s]', '', str(text))

    tokens = word_tokenize(text.lower())

    stemmed = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return ' '.join(stemmed)


def predict_single(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed])
    return model.predict(vectorized)[0], processed


def predict_batch(texts):
    cleaned_texts = [preprocess(text) for text in texts]
    vectorized = vectorizer.transform(cleaned_texts)
    predictions = model.predict(vectorized)
    return predictions, cleaned_texts

# PAGE CONFIG
st.set_page_config(
    page_title="AgroIntel Classifier",
    page_icon="🌱",
    layout="wide"
)

with st.sidebar:

    st.markdown("""
    <h1 style='color:white;'>🌱 AgroIntel</h1>
    <p style='color:#9CA3AF;'>
    NLP Agriculture Dashboard
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🧠 NLP Playground",
            "⚙️ Preprocessing",
            "📊 Model Performance",
            "📁 History"
        ]
    )

    st.markdown("---")

    st.success("✅ Model Active")
    st.caption("TF-IDF + Decision Tree")

# CUSTOM CSS
st.markdown("""
<style>

/* ===== IMPORT FONT ===== */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ===== GLOBAL ===== */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #071A12;
    color: white;
}

/* ===== MAIN CONTAINER ===== */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===== HERO SECTION ===== */
.hero-box {
    background: linear-gradient(
        135deg,
        #0B3D2E 0%,
        #14532D 50%,
        #1F7A4C 100%
    );

    border-radius: 30px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 4rem;
    font-weight: 800;
    color: #f6b72d;
    margin-bottom: 10px;
    letter-spacing: -2px;
}

.hero-desc {
    font-size: 1.2rem;
    color: #efe7db;
    margin-bottom: 24px;
}

.hero-badge {
    display: inline-block;
    padding: 10px 18px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    color: #f5f5f5;
    margin-right: 10px;
    font-size: 0.95rem;
    margin-top: 10px;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 14px;
    background: white;
    padding: 10px;
    border-radius: 20px;
    border: 1px solid #e6dcc8;
}

.stTabs [data-baseweb="tab"] {
    height: 58px;
    padding: 0 28px;
    background: transparent;
    border-radius: 14px;
    color: #6b7280;
    font-size: 1.05rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #355e3b !important;
    color: white !important;
}

/* ===== METRIC CARD ===== */
.metric-card {
    background: white;
    border: 1px solid #eadfcf;
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}

.metric-title {
    color: #7c7c7c;
    font-size: 0.9rem;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 1px;
}

.metric-value {
    font-size: 3rem;
    font-weight: 800;
    color: #1f1408;
    margin-top: 10px;
}

.metric-desc {
    color: #355e3b;
    font-size: 1rem;
}

/* ===== INPUT AREA ===== */
.input-box {
    background: white;
    border-radius: 24px;
    border: 1px solid #eadfcf;
    padding: 25px;
}

.stTextArea textarea {
    border-radius: 18px !important;
    border: 2px solid #e6dcc8 !important;
    padding: 22px !important;
    font-size: 1.2rem !important;
    background: #0F172A !important;
    color: white !important;
}

.stTextArea textarea:focus {
    border: 2px solid #355e3b !important;
    box-shadow: none !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #355e3b 0%, #4d7c53 100%);
    color: white;
    border: none;
    border-radius: 18px;
    padding: 16px;
    font-size: 1.15rem;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(90deg, #4d7c53 0%, #355e3b 100%);
}

/* ===== RESULT BOX ===== */
.result-box {
    background: linear-gradient(
        135deg,
        #0B3D2E 0%,
        #14532D 100%
    );

    border-radius: 28px;

    padding: 40px;

    text-align: center;

    margin-top: 24px;

    color: white;

    border: 1px solid rgba(255,255,255,0.1);

    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}

/* ===== EXAMPLE CARD ===== */
.example-card {
    background: white;
    border: 1px solid #eadfcf;
    border-left: 6px solid #355e3b;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 16px;
}

.example-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1f1408;
}

.example-desc {
    font-size: 1rem;
    color: #4b5563;
    margin-top: 6px;
}

/* ===== CHART CONTAINER ===== */
.chart-box {
    background: white;
    border-radius: 24px;
    padding: 20px;
    border: 1px solid #eadfcf;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: white;
    border-radius: 14px;
    border: 1px solid #eadfcf;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

/* ===== HIDE STREAMLIT ===== */
footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# HERO SECTION
st.markdown("""
<div class="hero-box">

<div class="hero-title">
🌾 AgroIntel Classifier
</div>
            
<div style="
font-size:1.1rem;
color:#bbf7d0;
margin-top:-10px;
margin-bottom:20px;
">
Intelligent Agricultural Text Classification Platform</div>

<div class="hero-desc">
Powered by Natural Language Processing and Machine Learning
</div>

<div>
<span class="hero-badge">📄 Dataset NLP Agriculture</span>
<span class="hero-badge">🧠 Decision Tree</span>
<span class="hero-badge">⚡ Accuracy 98%</span>
<span class="hero-badge">📊 Text Classification</span>
</div>

</div>
""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs([
    "📌 Prediction",
    "📊 Data Analysis",
    "ℹ️ Information"
])

# TAB PREDICTION
with tab1:

    st.markdown("**📌 Agriculture Text Input**")

    if model is None or vectorizer is None:
        st.error("⚠️ Model file tidak ditemukan!")
    else:
        col_input, col_examples = st.columns([1.3, 1], gap="small")
        
        with col_input:
            input_mode = st.radio(
                "Mode Input",
                ["Single Text", "Batch Text", "Upload CSV"],
                horizontal=True
            )

            st.markdown("""
            <div style="
                background: rgba(34,197,94,0.08);
                border: 1px solid rgba(34,197,94,0.2);
                padding: 14px 18px;
                border-radius: 14px;
                margin-bottom: 15px;
            ">
                <p style="
                    margin:0;
                    color:#86efac;
                    font-weight:600;
                    font-size:0.95rem;
                ">
                ✅ Active Model: Decision Tree + TF-IDF 1-Gram
                </p>
            </div>
            """, unsafe_allow_html=True)

            if input_mode == "Single Text":
                input_text = st.text_area(
                    "Masukkan teks",
                    "How to control aphid infestation in mustard crop?",
                    height=120,
                    label_visibility="collapsed"
                )

                if st.button("Analyze Text", width="stretch"):
                    if input_text.strip():
                        with st.spinner("⏳"):
                            prediction, processed = predict_single(input_text)

                        st.markdown(f"""
                        <div class="result-box">
                            <h2>Kategori Terdeteksi</h2>
                            <h1>{prediction}</h1>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("## ⚙️ NLP Pipeline")

                        colp1, colp2, colp3, colp4 = st.columns(4)

                        with colp1:
                            st.success("Cleaning")

                        with colp2:
                            st.success("Tokenization")

                        with colp3:
                            st.success("Stopword Removal")

                        with colp4:
                            st.success("Stemming")

                        with st.expander("🔍 Preprocessing Result"):
                            st.markdown("### Original Text")
                            st.code(input_text)

                            st.markdown("### Cleaned Text")
                            st.code(processed)
                    else:
                        st.warning("Enter text!")

            elif input_mode == "Batch Text":
                batch_text = st.text_area(
                    "Masukkan beberapa teks (satu teks per baris)",
                    "How to control aphid infestation in mustard crop?\nBest pesticide for aphid control?\nHow to improve crop production?",
                    height=180,
                    label_visibility="collapsed"
                )

                if st.button("Analyze Batch", width="stretch"):
                    texts = [line.strip() for line in batch_text.splitlines() if line.strip()]

                    if texts:
                        with st.spinner("⏳"):
                            predictions, cleaned_texts = predict_batch(texts)

                        result_df = pd.DataFrame({
                            "text": texts,
                            "cleaned_text": cleaned_texts,
                            "predicted_category": predictions
                        })

                        st.metric("Jumlah Teks", len(result_df))
                        st.dataframe(result_df, use_container_width=True)

                        csv_data = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Hasil Batch",
                            data=csv_data,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Masukkan minimal satu teks untuk diproses!")

            else:
                uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

                if uploaded_file is not None:
                    df_upload = pd.read_csv(uploaded_file)
                    text_columns = [col for col in df_upload.columns if pd.api.types.is_object_dtype(df_upload[col]) or pd.api.types.is_string_dtype(df_upload[col])]

                    if text_columns:
                        selected_col = st.selectbox("Pilih kolom teks", text_columns)

                        if st.button("Analyze Uploaded CSV", width="stretch"):
                            texts = [str(text).strip() for text in df_upload[selected_col].fillna("").tolist() if str(text).strip()]

                            if texts:
                                with st.spinner("⏳"):
                                    predictions, cleaned_texts = predict_batch(texts)

                                result_df = pd.DataFrame({
                                    "text": texts,
                                    "cleaned_text": cleaned_texts,
                                    "predicted_category": predictions
                                })

                                st.metric("Jumlah Baris Diproses", len(result_df))
                                st.dataframe(result_df, use_container_width=True)

                                csv_data = result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download Hasil CSV",
                                    data=csv_data,
                                    file_name="uploaded_predictions.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("Tidak ada teks yang valid dalam file CSV.")
                    else:
                        st.warning("File CSV tidak memiliki kolom teks yang bisa diproses.")
                else:
                    st.info("Unggah file CSV untuk memprediksi banyak data sekaligus.")
        
        with col_examples:

            st.markdown("## 📚 Example")

            st.info("🐛 Pest Control\n\nBest pesticide for aphid control?")

            st.info("🌱 Fertilizer\n\nPotassium fertilizer for crops")

            st.info("🚜 Crop Management\n\nHow to improve crop production?")
            
            st.info("🐄 Livestock\n\nBest feed for dairy cows")

            st.info("🌾 Agriculture Support\n\nAgriculture loan support")

            st.info("🐟 Aquaculture\n\nFish farming water quality management")

            st.info("📦 Other\n\nWeather today in Indonesia")

# TAB DATA ANALYSIS
with tab2:

    if df_sample is None:
        st.error("⚠️ File data tidak ditemukan!")

    else:

        st.markdown("## 📊 Overview Dataset")

        # ===== METRIC CARD =====
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Data</div>
                <div class="metric-value">{len(df_sample):,}</div>
                <div class="metric-desc">Dataset Rows</div>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Model</div>
                <div class="metric-value">DT</div>
                <div class="metric-desc">Decision Tree</div>
            </div>
            """, unsafe_allow_html=True)

        with col_s3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">98%</div>
                <div class="metric-desc">Classification Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col_s4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Feature</div>
                <div class="metric-value">TF-IDF 1-Gram</div>
                <div class="metric-desc">1-Gram Vectorizer</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🔥 Top Keywords")

        word_freq = {
            "answer":10058,
            "question":10000,
            "ask":6796,
            "water":4260,
            "spray":3107,
        }

        fig3, ax3 = plt.subplots(figsize=(8,4))

        sns.barplot(
            x=list(word_freq.keys()),
            y=list(word_freq.values()),
            hue=list(word_freq.keys()),
            palette="Greens",
            legend=False,
            ax=ax3
        )

        ax3.set_facecolor("#ffffff")
        fig3.patch.set_facecolor("#ffffff")

        st.pyplot(fig3)

        # ===== CHART =====
        col1, col2 = st.columns(2)

        with col1:

            st.markdown('<div class="chart-box">', unsafe_allow_html=True)

            st.markdown("### 📊 Category Distribution")

            fig1, ax1 = plt.subplots(figsize=(6,4))

            counts = df_sample['category'].value_counts()

            sns.barplot(
                x=counts.index,
                y=counts.values,
                hue=counts.index,
                palette="Greens",
                legend=False,
                ax=ax1
            )

            ax1.set_facecolor("#ffffff")
            fig1.patch.set_facecolor('#ffffff')

            ax1.set_xlabel("Category", fontsize=12)
            ax1.set_ylabel("Count", fontsize=12)

            plt.xticks(rotation=25)

            st.pyplot(fig1)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:

            st.markdown('<div class="chart-box">', unsafe_allow_html=True)

            st.markdown("### 🎯 Confusion Matrix")

            st.image(
                "confusion_matrix_terbaik.png",
                use_container_width=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("## 🧠 Model Comparison")

        model_df = pd.DataFrame({
            "Model":[
                "TF-IDF 1-gram",
                "BoW 1-gram",
                "TF-IDF 2-gram",
                "BoW 2-gram"
            ],
            "Accuracy":[0.9825,0.9785,0.8875,0.8840]
        })

        st.dataframe(
            model_df,
            use_container_width=True
        )

        # ===== DETAIL =====
        with st.expander("📈 Detailed Statistics"):

            col_a1, col_a2 = st.columns(2)

            with col_a1:
                st.markdown("### 📊 Dataset Statistics")

                st.json({
                    "Total Records": len(df_sample),
                    "Number of Categories": int(df_sample['category'].nunique()),
                    "Missing Value": int(df_sample.isnull().sum().sum())
                })

            with col_a2:
                st.markdown("### 🎯 Model Performance")

                st.json({
                    "Accuracy": "98.25%",
                    "Precision": "97.85%",
                    "Recall": "88.75%",
                    "Model": "Decision Tree"
                })

# TAB INFORMASI
with tab3:

    st.markdown("**ℹ️ About & Guide**")
    
    col_info1, col_info2 = st.columns(2, gap="small")
    
    with col_info1:
        with st.expander("🎯 About AgroIntel", expanded=True):
            st.markdown("""
            Machine learning-based agricultural text classification system that identifies agricultural information categories using Decision Tree and TF-IDF feature extraction.            """)
        
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Input Text** - Enter an agricultural question
            2. **Click Analyze** - Process the text automatically
            3. **View Results** - The category will be displayed
            
            **💡 Tips:** Use English for the best results
            """)
    
    with col_info2:
        with st.expander("🔧 Technology"):
            st.markdown("""
            - **Model:** Decision Tree Classifier
            - **Features:** TF-IDF 1-Gram
            - **Stack:** Scikit-Learn, NLTK, Streamlit
            - **Processing:** Tokenization, Stemming, Stopwords
            """)
        
        with st.expander("❓ FAQ"):
            st.markdown("""
            **Q: What is TF-IDF?**  
            A: A text representation technique that converts words into numerical features for machine learning models, considering both term frequency and inverse document frequency.
            
            **Q: What languages are supported?**  
            A: English is recommended for optimal performance.
            
            **Q: How accurate is the model?**  
            A: The model achieved approximately 98% accuracy on the training dataset.
            """)
    
    st.markdown("---")
    
    with st.expander("📊 Category Distribution"):
        cols = st.columns(2)
        categories = {
            "🌱 Fertilizer": "Fertilizer information",
            "🐛 Pest Control": "Pest control methods",
            "🚜 Crop Management": "Crop management strategies",
            "🐄 Livestock": "Livestock care",
            "🌾 Agriculture Support": "Agricultural support services",
            "🐟 Aquaculture": "Aquaculture practices",
            "📦 Other": "Other agricultural topics"
        }
        
        for idx, (cat, desc) in enumerate(categories.items()):
            with cols[idx % 2]:
                st.markdown(f"**{cat}** - {desc}")

# FOOTER
st.markdown("""
<div style="text-align: center; padding: 15px; margin-top: 20px;
            border-top: 1px solid #30363d;">
<p style="margin: 0; font-size: 12px; color: #8b949e;">
🌱 <strong>AgroIntel</strong> • Powered by Streamlit • Agricultural Text Classification
</p>
</div>
""", unsafe_allow_html=True)