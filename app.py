# import streamlit as st
# import tensorflow as tf
# from features.builder import build_features
# from config import MODELS

# st.set_page_config(page_title="VDss Predictor", layout="centered", page_icon="🧪")

# st.title("🧪 VDss Prediction (Deep Learning)")
# st.write("Predict Volume of Distribution (L/kg) from SMILES")

# # Sidebar with project information
# with st.sidebar:
#     st.header("📋 Project Information")
#     st.markdown("""
#     **Author:** KONATALA VIBHUVAN KRISHNA
    
#     **Supervisor:** Dr. A. R. Jac Fredo  
#     *(Associate Professor, IITBHU)*
    
#     ---
    
#     **About the Project:**
    
#     Building an end-to-end ADMET prediction pipeline to compare and find best among CNN, RNN, LSTM models across 7 descriptors and 8 datasets making 168 deep learning models.
    
#     Optimised the pipeline for GPU rendering by leveraging WSL 2 Linux virtual machines with CUDA cuDNN, ensuring stable and XLA-safe execution. Used MLflow for analysis and uploaded results to centralized server via Dagshub.
    
#     **Co-Authoring a research paper** about this work.
#     """)
    
#     st.markdown("---")
#     st.markdown("**Model:** RDKit + MACCS + CNN")
#     st.markdown("**Input:** SMILES string")
#     st.markdown("**Output:** Volume of Distribution (L/kg)")

# model_name = st.selectbox("Select Model", list(MODELS.keys()))
# model_path = MODELS[model_name]["path"]

# @st.cache_resource
# def load_model(path):
#     import json
#     import os
#     import tempfile
#     import zipfile
#     import shutil
    
#     # SOLUTION 1: Try standard loading first
#     try:
#         return tf.keras.models.load_model(path, compile=False)
#     except (ValueError, TypeError) as e:
#         error_str = str(e)
#         if 'batch_shape' in error_str or 'DTypePolicy' in error_str:
#             # SOLUTION 1: Copy-then-modify approach (safest)
#             # Copy entire archive, then modify only JSON files in-place
#             temp_dir = tempfile.mkdtemp()
#             fixed_path = path.replace('.keras', '_fixed.keras')
#             try:
#                 # Copy the entire archive first
#                 shutil.copy2(path, fixed_path)
                
#                 # Extract to temp directory
#                 with zipfile.ZipFile(fixed_path, 'r') as zip_ref:
#                     zip_ref.extractall(temp_dir)
                
#                 # Recursively fix compatibility issues in config
#                 def fix_compatibility(obj, depth=0):
#                     if depth > 50:  # Prevent infinite recursion
#                         return
                    
#                     if isinstance(obj, dict):
#                         # Fix batch_shape -> input_shape
#                         if 'batch_shape' in obj:
#                             batch_shape = obj.pop('batch_shape')
#                             if batch_shape and len(batch_shape) > 1:
#                                 obj['input_shape'] = batch_shape[1:]
                        
#                         # Fix DTypePolicy -> string dtype or remove it
#                         if 'dtype' in obj:
#                             dtype_val = obj['dtype']
#                             if isinstance(dtype_val, dict):
#                                 class_name = dtype_val.get('class_name', '')
#                                 module = dtype_val.get('module', '')
                                
#                                 if class_name == 'DTypePolicy' or (module == 'keras' and class_name == 'DTypePolicy'):
#                                     dtype_config = dtype_val.get('config', {})
#                                     dtype_name = dtype_config.get('name', 'float32')
#                                     obj['dtype'] = dtype_name
#                                 elif 'class_name' in dtype_val:
#                                     obj.pop('dtype')
#                                 else:
#                                     obj.pop('dtype')
                        
#                         # Recursively process all values
#                         for k, v in list(obj.items()):
#                             if k != 'dtype' or not isinstance(v, str):
#                                 fix_compatibility(v, depth + 1)
#                     elif isinstance(obj, list):
#                         for item in obj:
#                             fix_compatibility(item, depth + 1)
                
#                 # Fix all JSON files in the archive
#                 for root, dirs, files in os.walk(temp_dir):
#                     for file in files:
#                         if file.endswith('.json'):
#                             json_path = os.path.join(root, file)
#                             try:
#                                 with open(json_path, 'r', encoding='utf-8') as f:
#                                     config = json.load(f)
                                
#                                 fix_compatibility(config)
                                
#                                 with open(json_path, 'w', encoding='utf-8') as f:
#                                     json.dump(config, f, indent=2)
#                             except Exception:
#                                 pass
                
#                 # SOLUTION 1: Repackage by copying ALL files from original, only replacing JSON
#                 with zipfile.ZipFile(path, 'r') as orig_zip, zipfile.ZipFile(fixed_path, 'w', zipfile.ZIP_DEFLATED) as fixed_zip:
#                     # First, copy ALL files from original (preserves weights, metadata, etc.)
#                     for item in orig_zip.infolist():
#                         if not item.filename.endswith('.json'):
#                             # Copy non-JSON files exactly as they are
#                             fixed_zip.writestr(item, orig_zip.read(item.filename))
                    
#                     # Then, add/modify JSON files from temp directory
#                     for root, dirs, files in os.walk(temp_dir):
#                         for file in files:
#                             if file.endswith('.json'):
#                                 file_path = os.path.join(root, file)
#                                 arc_name = os.path.relpath(file_path, temp_dir).replace('\\', '/')
#                                 fixed_zip.write(file_path, arc_name)
                
#                 # Try to load the fixed model
#                 try:
#                     model = tf.keras.models.load_model(fixed_path, compile=False)
#                     os.remove(fixed_path)
#                     return model
#                 except Exception as e2:
#                     # SOLUTION 2: Try loading architecture and weights separately
#                     try:
#                         # Load model architecture from config
#                         with zipfile.ZipFile(fixed_path, 'r') as zip_ref:
#                             config_str = zip_ref.read('config.json').decode('utf-8')
#                             model_config = json.loads(config_str)
                        
#                         # Reconstruct model from config (try both methods)
#                         try:
#                             model = tf.keras.models.model_from_json(json.dumps(model_config), custom_objects={})
#                         except:
#                             # Fallback to model_from_config for newer Keras
#                             from keras.src.saving.saving_lib import model_from_config
#                             model = model_from_config(model_config, custom_objects={})
                        
#                         # Try to load weights from the fixed archive
#                         # Weights in .keras files are typically in variables/ directory or as .weights.h5
#                         with zipfile.ZipFile(fixed_path, 'r') as zip_ref:
#                             all_files = zip_ref.namelist()
#                             # Look for weight files
#                             weight_files = [f for f in all_files if 
#                                           ('variables' in f and (f.endswith('.index') or f.endswith('.data-00000-of-00001'))) or
#                                           f.endswith('.weights.h5') or
#                                           ('weight' in f.lower() and f.endswith('.h5'))]
                            
#                             if weight_files:
#                                 # Extract to temp and load
#                                 weights_temp = tempfile.mkdtemp()
#                                 try:
#                                     zip_ref.extractall(weights_temp)
#                                     # Try loading from variables directory (SavedModel format)
#                                     variables_dir = os.path.join(weights_temp, 'variables')
#                                     if os.path.exists(variables_dir):
#                                         model.load_weights(os.path.join(variables_dir, 'variables'))
#                                     else:
#                                         # Try finding .h5 weight file
#                                         for root, dirs, files in os.walk(weights_temp):
#                                             for file in files:
#                                                 if file.endswith('.h5') and 'weight' in file.lower():
#                                                     model.load_weights(os.path.join(root, file))
#                                                     break
#                                 finally:
#                                     shutil.rmtree(weights_temp, ignore_errors=True)
                        
#                         os.remove(fixed_path)
#                         return model
#                     except Exception as e3:
#                         try:
#                             os.remove(fixed_path)
#                         except:
#                             pass
#                         raise e2
#             finally:
#                 shutil.rmtree(temp_dir, ignore_errors=True)
        
#         # If all else fails, raise the original error
#         raise e

# model = load_model(model_path)

# smiles = st.text_input("Enter SMILES")

# if st.button("Predict VDss"):
#     if smiles.strip() == "":
#         st.warning("Please enter a SMILES string")
#     else:
#         try:
#             X = build_features(smiles)
#             pred = model.predict(X, verbose=0)[0][0]
#             st.success(f"**Predicted VDss:** {pred:.3f} L/kg")
#         except Exception as e:
#             st.error(str(e))

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load model ONCE
# ----------------------------
MODEL_PATH = "models/rdkit_maccs_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = StandardScaler()

app = FastAPI(title="VDss Predictor API")

# ----------------------------
# Request schema
# ----------------------------
class SMILESInput(BaseModel):
    smiles: str

# ----------------------------
# Feature extraction
# ----------------------------
def featurize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # RDKit descriptors
    rdkit_desc = {name: func(mol) for name, func in Descriptors._descList}

    # MACCS
    maccs = list(MACCSkeys.GenMACCSKeys(mol))
    maccs_dict = {f"MACCS_{i}": bit for i, bit in enumerate(maccs)}

    X = pd.DataFrame([{**rdkit_desc, **maccs_dict}])
    X = X.select_dtypes(include=[np.number])

    X_scaled = scaler.fit_transform(X.values)
    return X_scaled.reshape(1, X_scaled.shape[1], 1)

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def root():
    return {"status": "VDss API is running"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_vdss(data: SMILESInput):
    X = featurize(data.smiles)
    if X is None:
        return {"error": "Invalid SMILES"}

    pred = model.predict(X, verbose=0)[0][0]
    return {"vdss": float(pred)}
