# Deployment Guide - Streamlit Cloud

## 🚀 Quick Deployment Steps

### 1. Prepare Your Repository

Make sure your GitHub repository has:
- ✅ `app.py` (main Streamlit file)
- ✅ `requirements.txt` (dependencies)
- ✅ `models/rdkit_maccs_cnn.keras` (your model file)
- ✅ `features/` directory (all feature modules)
- ✅ `config.py`

### 2. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: VDss prediction app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vdss-streamlit-app.git
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. **Go to:** https://streamlit.io/cloud
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in:**
   - **Repository:** Select your repository
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. **Click "Deploy"**

### 4. Your App Will Be Live At:

`https://YOUR-APP-NAME.streamlit.app`

## 📝 Important Notes

- **Model File Size:** If your model is >100MB, you may need to use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.keras"
  git add .gitattributes
  git add models/*.keras
  git commit -m "Add model with LFS"
  ```

- **First Load:** The app may take 1-2 minutes to load on first deployment as it installs dependencies and loads the model.

- **Custom Domain:** You can customize your app URL in Streamlit Cloud settings.

## ✅ Checklist Before Deployment

- [ ] All files are committed to GitHub
- [ ] `requirements.txt` is up to date
- [ ] Model file is included in repository
- [ ] Tested locally with `streamlit run app.py`
- [ ] Author and project info is displayed correctly

## 🔗 Add to Resume

Once deployed, add this to your resume:

**"Developed and deployed a deep-learning-powered web application for predicting pharmacokinetic VDss from molecular SMILES using RDKit descriptors and CNN models. [Live Demo: https://YOUR-APP-NAME.streamlit.app]"**
