# ================================
# Smart Healthcare Assistant (Full Cloud-ready)
# ================================

import os, json, random
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Directories
MODELS_DIR = "models"
PROGRESS_DIR = "progress"
HISTORY_CSV = os.path.join(PROGRESS_DIR, "history.csv")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

# Lazy flags
REPORTLAB_AVAILABLE = False
TRANSLATION_AVAILABLE = False
GPT4ALL_AVAILABLE = False
GPT_INSTANCE = None

# ========================
# Lazy imports
# ========================
def lazy_reportlab():
    global REPORTLAB_AVAILABLE, canvas
    if REPORTLAB_AVAILABLE: return
    try:
        from reportlab.pdfgen import canvas
        REPORTLAB_AVAILABLE = True
    except: REPORTLAB_AVAILABLE = False

def lazy_translator():
    global TRANSLATION_AVAILABLE, GoogleTranslator
    if TRANSLATION_AVAILABLE: return
    try:
        from deep_translator import GoogleTranslator
        TRANSLATION_AVAILABLE = True
    except: TRANSLATION_AVAILABLE = False

def lazy_gpt():
    global GPT4ALL_AVAILABLE, GPT_INSTANCE
    if GPT4ALL_AVAILABLE: return
    try:
        from gpt4all import GPT4All
        GPT_INSTANCE = GPT4All()
        GPT4ALL_AVAILABLE = True
    except: GPT4ALL_AVAILABLE = False; GPT_INSTANCE=None

# ========================
# Model loading on-demand
# ========================
@st.cache_resource
def load_model(name):
    import joblib
    mdl, sc = None, None
    model_path = os.path.join(MODELS_DIR,f"{name}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR,f"{name}_scaler.pkl")
    if os.path.exists(model_path):
        try: mdl = joblib.load(model_path)
        except: mdl=None
    if os.path.exists(scaler_path):
        try: sc = joblib.load(scaler_path)
        except: sc=None
    return {"model": mdl, "scaler": sc}

def apply_scaler(X, scaler):
    if scaler is None: return X
    try: return scaler.transform(X)
    except: return X

def predict(name, X_list):
    entry = load_model(name)
    model = entry["model"]
    scaler = entry["scaler"]
    if model is None: return None, None
    X = np.array(X_list).reshape(1,-1)
    Xs = apply_scaler(X, scaler)
    pred = int(model.predict(Xs)[0])
    prob = None
    try:
        pp = model.predict_proba(Xs)
        if pp.shape[1]==2: prob = float(pp[0,1])
        else: prob = float(pp[0,pred]) if pred<pp.shape[1] else None
    except: prob=None
    return pred, prob

# ========================
# Symptom mapping
# ========================
SYMPTOM_CHOICES = {
    "diabetes":["Frequent urination","Excessive thirst","Fatigue","Slow healing wounds","Blurred vision"],
    "heart":["Chest pain","Shortness of breath","Dizziness","Swelling in legs","Irregular heartbeat"],
    "liver":["Yellowing of eyes/skin","Abdominal pain","Nausea","Fatigue","Loss of appetite"],
    "kidney":["Swelling in feet","Foamy urine","Fatigue","Back pain","Frequent urination at night"]
}

def symptoms_to_values_original(disease, symptoms):
    # Your original mapping code here
    # Copy from your existing 650-line app
    # Must return list of numeric values in correct feature order
    return [...]  # replace with full original logic

@st.cache_data
def symptoms_to_values(disease, symptoms):
    return symptoms_to_values_original(disease, symptoms)

# ========================
# History helpers
# ========================
@st.cache_data
def load_history():
    if os.path.exists(HISTORY_CSV):
        try: return pd.read_csv(HISTORY_CSV)
        except: return pd.DataFrame()
    return pd.DataFrame()

def save_history(record):
    df=pd.DataFrame([record])
    if os.path.exists(HISTORY_CSV): df.to_csv(HISTORY_CSV,mode="a",header=False,index=False)
    else: df.to_csv(HISTORY_CSV,index=False)

# ========================
# Health tips & doctor map
# ========================
TIPS = {
    "diabetes":{"high":["Maintain low sugar diet","Exercise","Monitor sugar","Visit endocrinologist"],
                "low":["Maintain healthy weight","Routine checkups"]},
    "heart":{"high":["Avoid smoking","Control BP","Immediate attention for chest pain"],
             "low":["Stay active and balanced diet"]},
    "liver":{"high":["Avoid alcohol","Avoid unnecessary meds","Consult hepatologist"],
             "low":["Limit alcohol, balanced diet"]},
    "kidney":{"high":["Control sugar & BP","Limit salt & protein","Consult nephrologist"],
              "low":["Keep hydrated","Avoid excess painkillers"]}
}

DOCTOR_MAP={"diabetes":"Endocrinologist","heart":"Cardiologist","liver":"Hepatologist","kidney":"Nephrologist"}

# ========================
# PDF helper
# ========================
def generate_pdf(record, out_path):
    lazy_reportlab()
    if not REPORTLAB_AVAILABLE: return False,"reportlab not available"
    try:
        c=canvas.Canvas(out_path)
        c.drawString(50,800,"Healthcare Assistant Report")
        y=780
        for k,v in record.items():
            c.drawString(50,y,f"{k}: {v}")
            y-=20
            if y<50: c.showPage(); y=780
        c.save(); return True,None
    except Exception as e: return False,str(e)

# ========================
# Chatbot
# ========================
def chatbot_answer(user_q):
    q=user_q.lower().strip()
    if any(w in q for w in ["diet","eat","food"]):
        return "Balanced diet recommended."
    if any(w in q for w in ["exercise","workout","activity"]):
        return "30 min moderate exercise most days."
    if any(w in q for w in ["doctor","visit"]):
        return "Seek medical professional if severe."
    lazy_gpt()
    if GPT_INSTANCE:
        try:
            with GPT_INSTANCE.chat_session() as sess:
                prompt=f"You are a helpful health assistant. Question: {user_q}"
                return sess.generate(prompt,max_tokens=200)
        except: return None
    return "General health guidance only."

# ========================
# Translation
# ========================
def translate_if_available(text,target_lang):
    lazy_translator()
    if not TRANSLATION_AVAILABLE: return None
    try: return GoogleTranslator(source="auto",target=target_lang).translate(text)
    except: return None

# ========================
# UI
# ========================
st.set_page_config(page_title="Smart Healthcare Assistant",layout="wide")
st.title("🩺 Smart Healthcare Assistant")

section=st.sidebar.selectbox("Section",["Prediction","History","Chatbot","Upload CSV","Settings"])
selected_disease=st.sidebar.selectbox("Disease",["diabetes","heart","liver","kidney"])
input_mode=st.sidebar.radio("Input Mode",["Lab Reports (exact values)","Symptoms (I only know symptoms)"])
lang_choice=st.sidebar.selectbox("Display language",["auto","en","ur","es","fr"])
FEATURES_MAP={
    "diabetes":["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
    "heart":["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"],
    "liver":["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"],
    "kidney":["Age","BloodPressure","SpecificGravity","Albumin","Sugar","RedBloodCells","PusCell","PusCellClumps","Bacteria","BloodGlucoseRandom","SerumCreatinine","Sodium","Potassium","Hemoglobin","PackedCellVolume","WhiteBloodCellCount","RedBloodCellCount"]
}

# ========================
# Prediction Section
# ========================
if section=="Prediction":
    st.header("Prediction")
    features=FEATURES_MAP[selected_disease]
    use_csv=st.checkbox("Use CSV first row")
    uploaded_csv=None; csv_vals=None
    if use_csv: uploaded_csv=st.file_uploader("Upload CSV",type=["csv"])
    if uploaded_csv: csv_vals=extract_inputs_from_csv(uploaded_csv,features)
    inputs=csv_vals if csv_vals else None
    symptoms=[]
    if input_mode=="Symptoms (I only know symptoms)" and not use_csv:
        symptoms=st.multiselect("Select symptoms",SYMPTOM_CHOICES[selected_disease])
        if symptoms: inputs=symptoms_to_values(selected_disease,symptoms); st.write(dict(zip(features,inputs)))
    elif input_mode=="Lab Reports (exact values)" and not use_csv:
        cols=st.columns(2); vals=[]
        for i,f in enumerate(features):
            with cols[i%2]: val=st.number_input(f"{f}",value=0.0)
            vals.append(val)
        inputs=vals
    if st.button("Run Prediction") and inputs:
        pred, prob=predict(selected_disease,inputs)
        rec={"timestamp":datetime.utcnow().isoformat(),"disease":selected_disease,
             "mode":"csv" if use_csv else ("lab" if input_mode.startswith("Lab") else "symptoms"),
             "symptoms":json.dumps(symptoms),"inputs":json.dumps(inputs),
             "prediction":int(pred) if pred is not None else None,
             "probability":float(prob) if prob is not None else None,"bookmark":False}
        save_history(rec)
        if pred==1: st.error(f"⚠️ High Risk {selected_disease.upper()}")
        else: st.success(f"✅ Low Risk {selected_disease.upper()}")
        if prob is not None: st.write(f"Probability: {prob*100:.2f}%")

# ========================
# History, Chatbot, CSV, Settings
# Add lazy-load logic as needed
# ========================

# ========================
# History Section
# ========================
elif section == "History":
    st.header("Prediction History")
    df_hist = load_history()
    if df_hist.empty:
        st.info("No history records yet.")
    else:
        st.dataframe(df_hist.sort_values("timestamp", ascending=False).reset_index(drop=True))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export History as CSV"):
                csv_out = df_hist.to_csv(index=False).encode("utf-8")
                st.download_button("Download history.csv", data=csv_out, file_name="history_export.csv")
        with col2:
            if st.button("Clear History"):
                try:
                    os.remove(HISTORY_CSV)
                    st.success("History cleared.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error("Failed to clear history: " + str(e))

# ========================
# Chatbot Section
# ========================
elif section == "Chatbot":
    st.header("Chatbot Guidance")
    lazy_gpt()  # Load GPT only if Chatbot section selected
    q = st.text_input("Ask a health question (e.g., 'What should I eat to reduce sugar?')","")
    if st.button("Ask"):
        if not q.strip():
            st.warning("Type a question first.")
        else:
            answer = chatbot_answer(q)
            st.markdown("**Bot:** " + str(answer))
            # Translation
            lazy_translator()
            if TRANSLATION_AVAILABLE:
                tgt = st.selectbox("Translate to", ["none","ur","es","fr"], index=0)
                if tgt != "none":
                    tr = translate_if_available(answer, tgt)
                    if tr:
                        st.markdown(f"**Translated ({tgt}):** {tr}")
                    else:
                        st.info("Translation failed or not available.")
            else:
                st.info("Translation package not installed. Optional: install `deep_translator`.")

# ========================
# Upload CSV Section
# ========================
elif section == "Upload CSV":
    st.header("Upload CSV for lab results")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        features = FEATURES_MAP[selected_disease]
        def extract_inputs_from_csv(uploaded_file, features):
            try:
                df = pd.read_csv(uploaded_file)
                if df.shape[0]==0: raise ValueError("CSV has no rows")
                first = df.iloc[0]
                vals=[]
                for f in features:
                    match = next((col for col in df.columns if col.strip().lower()==f.strip().lower()), f)
                    vals.append(first.get(match,0))
                return vals
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
                return None
        vals = extract_inputs_from_csv(uploaded, features)
        if vals is not None:
            st.write("Using these values for prediction (first row):")
            st.write(dict(zip(features, vals)))
            if st.button("Predict from uploaded CSV"):
                pred, prob = predict(selected_disease, vals)
                st.write("Prediction:", "High risk" if pred==1 else "Low risk")
                if prob is not None:
                    st.write(f"Probability: {prob*100:.2f}%")
                # Save
                rec = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "disease": selected_disease,
                    "mode": "csv_upload",
                    "symptoms": json.dumps([]),
                    "inputs": json.dumps(vals),
                    "prediction": int(pred),
                    "probability": float(prob) if prob is not None else None,
                    "bookmark": False
                }
                save_history(rec)

# ========================
# Settings Section
# ========================
elif section == "Settings":
    st.header("Settings & Troubleshooting")
    lazy_reportlab(); lazy_translator(); lazy_gpt()
    st.write(f"Translation available: {TRANSLATION_AVAILABLE}")
    st.write(f"ReportLab available (PDF): {REPORTLAB_AVAILABLE}")
    st.write(f"GPT4All available: {GPT4ALL_AVAILABLE}")
    if st.button("Reload models"):
        try:
            load_model.clear()
        except:
            pass
        st.experimental_rerun()

# ========================
# End of App
# ========================
