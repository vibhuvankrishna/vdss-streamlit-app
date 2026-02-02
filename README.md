# VDss Prediction Web Application

A deep learning-powered web application for predicting pharmacokinetic Volume of Distribution (VDss) from molecular SMILES using RDKit descriptors and CNN models.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser:**
   - The app will automatically open at `http://localhost:8501`

## 📋 Project Information

**Author:** KONATALA VIBHUVAN KRISHNA

**Supervisor:** Dr. A. R. Jac Fredo (Associate Professor, IITBHU)

### About the Project

Building an end-to-end ADMET prediction pipeline to compare and find best among CNN, RNN, LSTM models across 7 descriptors and 8 datasets making 168 deep learning models.

Optimised the pipeline for GPU rendering by leveraging WSL 2 Linux virtual machines with CUDA cuDNN, ensuring stable and XLA-safe execution. Used MLflow for analysis and uploaded results to centralized server via Dagshub.

**Co-Authoring a research paper** about this work.

## 🏗️ Architecture

```
vdss_streamlit_app/
├── app.py                  # Streamlit entry point
├── config.py               # Model configuration
├── requirements.txt        # Dependencies
├── models/
│   └── rdkit_maccs_cnn.keras
└── features/
    ├── rdkit.py           # RDKit descriptors
    ├── maccs.py           # MACCS fingerprints
    └── builder.py         # Feature pipeline
```

## 🧪 Usage

1. Enter a SMILES string in the input field
2. Click "Predict VDss"
3. View the predicted Volume of Distribution (L/kg)

### Example SMILES:
- `CCO` - Ethanol
- `CC(=O)O` - Acetic acid
- `c1ccccc1` - Benzene

## 📦 Dependencies

- streamlit
- tensorflow==2.15.1
- numpy<2.0,>=1.26.0
- pandas
- scikit-learn
- rdkit-pypi
- protobuf<5.0,>=3.20

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push this repository to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path: `app.py`
7. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## 📝 License

Research project - IITBHU
