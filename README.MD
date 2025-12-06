# 🩺 Medical Diagnoser

An AI-powered medical diagnosis tool for predicting **Diabetes** and **Hypertension** using machine learning models and LLM-based data extraction from lab reports.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/me-Afzal/medical-diagnoser-Diabetics-hypertension.git)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)

---

## 📋 Overview

**Medical Diagnoser** is designed for hospitals and healthcare professionals to streamline the diagnosis process by automating medical report analysis. The application eliminates manual screening of lab reports by:

- **Extracting** key medical parameters from uploaded reports (PDF, DOCX, TXT, or images)
- **Processing** data using LLM-powered extraction with Google's Gemini API
- **Predicting** diabetes and hypertension risk using trained machine learning models

This tool reduces diagnostic time, minimizes human error, and provides instant predictions based on clinical data.

---

## APP Link: [Click Here](https://medical-diagnoser-diabetics-hypertension.streamlit.app/)

## ✨ Features

### 🔍 Intelligent Data Extraction
- **Multi-format support**: PDF (with OCR for scanned documents), DOCX, TXT, and image files (PNG, JPG, JPEG)
- **LLM-powered extraction**: Uses Google Gemini 2.5 Flash Lite for accurate parameter extraction
- **Automatic unit conversion**: Converts lab values to standardized units (mg/dL → mmol/L, etc.)

### 🩸 Diabetes Prediction
- **Input methods**: Upload lab report or manual entry
- **Key parameters**: Gender, Age, Urea, Creatinine, HbA1c, Cholesterol, Triglycerides, HDL, LDL, VLDL, BMI
- **Prediction classes**: No Diabetes | Close to Diabetes | Diabetes
- **Model**: Random Forest Classifier

### 💓 Hypertension Prediction
- **Input methods**: Upload lab report or manual entry
- **Key parameters**: Age, Salt Intake, Stress Score, Blood Pressure History, Sleep Duration, BMI, Medication, Family History, Exercise Level, Smoking Status
- **Prediction classes**: Hypertension | No Hypertension
- **Model**: XGBoost Classifier

### 🎨 User Interface
- Clean, dark-themed Streamlit interface
- Tabbed navigation for different conditions
- Real-time extraction and prediction
- JSON visualization of extracted data

---

## 🏗️ Project Structure

```
medical-diagnoser/
│
├── main.py                              # Main Streamlit application
├── preprocess.py                       # Text extraction and cleaning utilities
├── inference.py                        # ML model inference functions
│
├── llm_extractors/
│   ├── diabetics_llm_extractor.py     # Diabetes data extractor
│   └── hypertension_llm_extractor.py  # Hypertension data extractor
│
├── models/
│   ├── diabetic_rf_model.pkl          # Trained Random Forest model
│   ├── diabetic_scaler.pkl            # Scaler for diabetes features
│   ├── hypertension_xgb_model.pkl     # Trained XGBoost model
│   ├── hypertension_scaler.pkl        # Scaler for hypertension features
│   ├── hypertension_o_encoder_bp.pkl  # Ordinal encoder for BP history
│   └── hypertension_o_encoder_exercise.pkl  # Ordinal encoder for exercise level
│
├── Diabetics-POC.ipynb                # Model development notebook for Diabetes
├── HyperTension-POC.ipynb             # Model development notebook for Hypertension
│
├── requirements.txt                    # Python dependencies
├── .streamlit/
│   └── secrets.toml                   # API keys (not in repo)
└── README.md                          # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/me-Afzal/medical-diagnoser-Diabetics-hypertension.git
   cd medical-diagnoser-Diabetics-hypertension
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   
   Create `.streamlit/secrets.toml` file:
   ```toml
   api_key = "YOUR_GEMINI_API_KEY_HERE"
   ```

5. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

---

## 📦 Dependencies

```
streamlit
numpy
pandas
scikit-learn
xgboost
pymupdf (fitz)
easyocr
python-docx
requests
pickle
```

Install all dependencies:
```bash
pip install streamlit numpy pandas scikit-learn xgboost pymupdf easyocr python-docx requests
```

---

## 💻 Usage

### Upload Lab Report Mode

1. Navigate to the desired prediction tab (Diabetes or Hypertension)
2. Select "Upload Lab Report" as input method
3. Upload your medical report (PDF, DOCX, TXT, or image)
4. Click "Extract & Predict"
5. View extracted parameters and prediction result

### Manual Entry Mode

1. Select "Manual Entry" as input method
2. Fill in all required medical parameters
3. Click "Predict"
4. View prediction result

---

## 🧪 Model Evaluation

Detailed model development, training, and evaluation processes are documented in Jupyter notebooks:

### Diabetes Model
- **Notebook**: [Diabetics-POC.ipynb](./Diabetics-POC.ipynb)
- **Contents**: Data preprocessing, feature engineering, model selection, hyperparameter tuning, performance metrics
- **Algorithm**: Random Forest Classifier
- **Features**: 11 clinical parameters including metabolic markers and demographics

### Hypertension Model
- **Notebook**: [HyperTension-POC.ipynb](./HyperTension-POC.ipynb)
- **Contents**: Data analysis, feature encoding, model comparison, cross-validation results
- **Algorithm**: XGBoost Classifier
- **Features**: 10 lifestyle and clinical factors including BP history and behavioral metrics

Both notebooks include:
- Exploratory Data Analysis (EDA)
- Feature importance analysis
- Model performance metrics (accuracy, precision, recall, F1-score)
- Confusion matrices
- ROC curves and AUC scores

---

## 🔧 Technical Details

### Data Extraction Pipeline

1. **File Upload**: Accepts multiple formats (PDF, DOCX, TXT, images)
2. **Text Extraction**: 
   - Plain text from TXT/DOCX
   - PyMuPDF for PDF text extraction
   - EasyOCR for scanned PDFs and images
3. **Text Cleaning**: Removes noise, normalizes whitespace
4. **LLM Extraction**: Gemini API extracts structured data
5. **Unit Conversion**: Standardizes measurements to required units

### Prediction Pipeline

1. **Data Validation**: Fills missing values with defaults
2. **Feature Engineering**: 
   - Age grouping for diabetes
   - Ordinal encoding for categorical features
3. **Scaling**: Standardizes numerical features
4. **Inference**: Trained models generate predictions
5. **Output**: User-friendly prediction labels

---

## 🎯 Use Cases

- **Hospitals**: Rapid screening of patient lab reports
- **Clinics**: Automated preliminary diagnosis support
- **Telemedicine**: Remote patient monitoring and risk assessment
- **Research**: Clinical data analysis and pattern recognition
- **Healthcare Analytics**: Population health screening programs

---

## ⚠️ Disclaimer

This application is designed as a **diagnostic support tool** and should not replace professional medical judgment. All predictions should be reviewed by qualified healthcare professionals before making clinical decisions.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Afzal**

- GitHub: [@me-Afzal](https://github.com/me-Afzal)
- Repository: [medical-diagnoser-Diabetics-hypertension](https://github.com/me-Afzal/medical-diagnoser-Diabetics-hypertension.git)

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- Google Gemini API for LLM-powered data extraction
- Streamlit for the web application framework
- scikit-learn and XGBoost for machine learning models
- EasyOCR for optical character recognition

---

**Made with ❤️ for better healthcare**