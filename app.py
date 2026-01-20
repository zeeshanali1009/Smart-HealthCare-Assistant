# Part 1/4 — Imports, config, model loading, utilities
# app.py — Part 1/4
# Full-feature Smart Healthcare Assistant
import os
import json
import random
import joblib
from datetime import datetime
from deep_translator import GoogleTranslator


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional libraries (safe imports)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except Exception:
    TRANSLATION_AVAILABLE = False

try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except Exception:
    GPT4ALL_AVAILABLE = False

# --------------------
# Directories & files
# --------------------
MODELS_DIR = "models"
DATA_DIR = "data"
PROGRESS_DIR = "progress"
HISTORY_CSV = os.path.join(PROGRESS_DIR, "history.csv")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

# --------------------
# Load models (safe)
# --------------------
@st.cache_resource
def load_models():
    """Load models and optional scalers if present. Returns dict: name -> {'model':..., 'scaler':...}"""
    names = ["diabetes", "heart", "liver", "kidney"]
    loaded = {}
    for n in names:
        model_path = os.path.join(MODELS_DIR, f"{n}_model.pkl")
        scaler_path = os.path.join(MODELS_DIR, f"{n}_scaler.pkl")
        mdl = None
        sc = None
        if os.path.exists(model_path):
            try:
                mdl = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load model {model_path}: {e}")
        else:
            st.warning(f"Model file not found: {model_path} (train {n} model first).")
        if os.path.exists(scaler_path):
            try:
                sc = joblib.load(scaler_path)
            except Exception:
                st.info(f"Scaler load failed for {scaler_path}; continuing without scaler.")
        loaded[n] = {"model": mdl, "scaler": sc}
    return loaded

ALL_MODELS = load_models()

def apply_scaler(name, X_arr):
    sc = ALL_MODELS.get(name, {}).get("scaler")
    if sc is not None:
        try:
            return sc.transform(X_arr)
        except Exception:
            return X_arr
    return X_arr

def predict_model(name, X_list):
    """Return (pred, prob) where prob may be None if predict_proba unavailable."""
    entry = ALL_MODELS.get(name)
    if not entry or entry["model"] is None:
        raise RuntimeError(f"Model for {name} not available.")
    model = entry["model"]
    X = np.array(X_list).reshape(1, -1)
    Xs = apply_scaler(name, X)
    pred = int(model.predict(Xs)[0])
    prob = None
    try:
        pp = model.predict_proba(Xs)
        # for binary, probability of positive class is pp[0,1] if shape (1,2)
        if pp.shape[1] == 2:
            prob = float(pp[0,1])
        else:
            # fallback: predicted class probability
            prob = float(pp[0, pred]) if pred < pp.shape[1] else None
    except Exception:
        prob = None
    return pred, prob

# --------------------
# History utilities
# --------------------
def save_history_record(record):
    """Append a record dict to history CSV."""
    df = pd.DataFrame([record])
    if os.path.exists(HISTORY_CSV):
        df.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_CSV, index=False)

def load_history_df():
    if os.path.exists(HISTORY_CSV):
        try:
            return pd.read_csv(HISTORY_CSV)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()
# Part 2/4 — Symptom -> value generation, tips, doctor map, PDF, plotting
# app.py — Part 2/4
# Symptom mappings, tips, doctor map, PDF and plotting helpers

# --------------------
# Symptom options for UI (human-friendly)
# --------------------
SYMPTOM_CHOICES = {
    "diabetes": ["Frequent urination", "Excessive thirst", "Fatigue", "Slow healing wounds", "Blurred vision"],
    "heart": ["Chest pain", "Shortness of breath", "Dizziness", "Swelling in legs", "Irregular heartbeat"],
    "liver": ["Yellowing of eyes/skin", "Abdominal pain", "Nausea", "Fatigue", "Loss of appetite"],
    "kidney": ["Swelling in feet", "Foamy urine", "Fatigue", "Back pain", "Frequent urination at night"]
}

# --------------------
# Convert symptoms to realistic-ish numeric inputs (randomized ranges)
# Must return list in same feature order as used to train models.
# --------------------
def symptoms_to_values(disease, symptoms):
    r = random.Random(sum(len(s) for s in symptoms) + len(symptoms))
    if disease == "diabetes":
        base = {
            "Pregnancies": r.randint(0, 6),
            "Glucose": r.randint(90, 125),
            "BloodPressure": r.randint(65, 85),
            "SkinThickness": r.randint(18, 30),
            "Insulin": r.randint(40, 140),
            "BMI": round(r.uniform(18.5, 30.0), 1),
            "DiabetesPedigreeFunction": round(r.uniform(0.1, 1.2), 2),
            "Age": r.randint(20, 65)
        }
        if "Frequent urination" in symptoms:
            base["Glucose"] = r.randint(150, 185)
        if "Excessive thirst" in symptoms:
            base["Glucose"] = max(base["Glucose"], r.randint(160, 210))
        if "Fatigue" in symptoms:
            base["BMI"] = round(r.uniform(27, 36), 1)
        if "Slow healing wounds" in symptoms:
            base["Insulin"] = r.randint(20, 80)
        if "Blurred vision" in symptoms:
            base["Glucose"] = r.randint(170, 230)
        return [base[k] for k in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

    if disease == "heart":
        base = {
            "Age": r.randint(30, 75),
            "Sex": r.randint(0,1),
            "ChestPainType": 0,
            "RestingBP": r.randint(100, 140),
            "Cholesterol": r.randint(150, 230),
            "FastingBS": 0,
            "RestingECG": 0,
            "MaxHR": r.randint(110, 170),
            "ExerciseAngina": 0,
            "Oldpeak": round(r.uniform(0.0, 2.5), 1),
            "ST_Slope": r.randint(0,2)
        }
        if "Chest pain" in symptoms:
            base["ChestPainType"] = r.choice([1,2,3])
            base["Cholesterol"] = r.randint(220, 320)
        if "Shortness of breath" in symptoms:
            base["ExerciseAngina"] = 1
            base["RestingBP"] = r.randint(140, 170)
        if "Dizziness" in symptoms:
            base["MaxHR"] = r.randint(80, 120)
        if "Swelling in legs" in symptoms:
            base["Oldpeak"] = round(r.uniform(1.0, 3.5), 1)
        return [base[k] for k in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]]

    if disease == "liver":
        base = {
            "Age": r.randint(20, 70),
            "Gender": r.randint(0,1),
            "Total_Bilirubin": round(r.uniform(0.3, 1.2), 2),
            "Direct_Bilirubin": round(r.uniform(0.1, 0.4), 2),
            "Alkaline_Phosphotase": r.randint(70, 150),
            "Alamine_Aminotransferase": r.randint(15, 45),
            "Aspartate_Aminotransferase": r.randint(15, 45),
            "Total_Protiens": round(r.uniform(6.0, 8.5), 1),
            "Albumin": round(r.uniform(3.2, 4.8), 1),
            "Albumin_and_Globulin_Ratio": round(r.uniform(0.8, 1.8), 2)
        }
        if "Yellowing of eyes/skin" in symptoms:
            base["Total_Bilirubin"] = round(r.uniform(2.0, 6.0), 2)
            base["Direct_Bilirubin"] = round(r.uniform(1.0, 3.5), 2)
        if "Abdominal pain" in symptoms:
            base["Alkaline_Phosphotase"] = r.randint(160, 320)
        if "Nausea" in symptoms:
            base["Alamine_Aminotransferase"] = r.randint(50, 120)
        return [base[k] for k in ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]]

    if disease == "kidney":
        base = {
            "Age": r.randint(20, 75),
            "BloodPressure": r.randint(70, 120),
            "SpecificGravity": round(r.uniform(1.005, 1.025), 3),
            "Albumin": r.randint(0, 3),
            "Sugar": r.randint(0, 3),
            "RedBloodCells": r.randint(0, 1),
            "PusCell": r.randint(0, 1),
            "PusCellClumps": r.randint(0, 1),
            "Bacteria": r.randint(0, 1),
            "BloodGlucoseRandom": r.randint(80, 150),
            "SerumCreatinine": round(r.uniform(0.6, 2.5), 2),
            "Sodium": r.randint(130, 145),
            "Potassium": round(r.uniform(3.2, 5.5), 1),
            "Hemoglobin": round(r.uniform(10.5, 15.5), 1),
            "PackedCellVolume": r.randint(30, 48),
            "WhiteBloodCellCount": r.randint(4000, 12000),
            "RedBloodCellCount": round(r.uniform(3.5, 5.2), 1)
        }
        if "Swelling in feet" in symptoms:
            base["Albumin"] = min(5, base["Albumin"] + r.randint(1,3))
        if "Foamy urine" in symptoms:
            base["Albumin"] = max(base["Albumin"], r.randint(2,4))
        if "Frequent urination at night" in symptoms:
            base["BloodGlucoseRandom"] = r.randint(140, 220)
        return [base[k] for k in ["Age","BloodPressure","SpecificGravity","Albumin","Sugar","RedBloodCells","PusCell","PusCellClumps","Bacteria","BloodGlucoseRandom","SerumCreatinine","Sodium","Potassium","Hemoglobin","PackedCellVolume","WhiteBloodCellCount","RedBloodCellCount"]]

    return []

# --------------------
# Rule-based symptom overrides (Option 1)
# If certain severe symptom combinations are present, force High Risk.
# --------------------
def symptom_override(disease, symptoms, current_pred, current_prob):
    # returns (pred, prob) possibly overridden
    pred = current_pred
    prob = current_prob
    if disease == "diabetes":
        # if >=3 key symptoms, mark high risk
        if len([s for s in symptoms if s in SYMPTOM_CHOICES["diabetes"]]) >= 3:
            return 1, 0.88
    if disease == "heart":
        if "Chest pain" in symptoms and "Shortness of breath" in symptoms:
            return 1, 0.92
    if disease == "liver":
        if ("Yellowing of eyes/skin" in symptoms or "Jaundice" in symptoms) and ("Abdominal pain" in symptoms or "Nausea" in symptoms):
            return 1, 0.9
    if disease == "kidney":
        if "Swelling in feet" in symptoms and "Foamy urine" in symptoms:
            return 1, 0.9
    return pred, prob

# --------------------
# Health tips (per disease)
# --------------------
TIPS = {
    "diabetes": {
        "high": [
            "Maintain a balanced diet (low sugar, high fiber).",
            "Exercise at least 30 minutes daily.",
            "Monitor blood sugar regularly.",
            "Visit an endocrinologist for confirmation."
        ],
        "low": [
            "Maintain healthy weight and diet.",
            "Get routine checkups and stay active."
        ]
    },
    "heart": {
        "high": [
            "Avoid smoking and limit alcohol consumption.",
            "Control blood pressure and cholesterol.",
            "Seek immediate medical attention for severe chest pain."
        ],
        "low": [
            "Stay physically active and maintain a balanced diet."
        ]
    },
    "liver": {
        "high": [
            "Avoid alcohol completely.",
            "Avoid unnecessary medications that strain the liver.",
            "Consult a hepatologist for tests."
        ],
        "low": [
            "Limit alcohol and eat a balanced diet."
        ]
    },
    "kidney": {
        "high": [
            "Control blood sugar and blood pressure.",
            "Limit salt and excessive protein intake.",
            "Consult a nephrologist."
        ],
        "low": [
            "Keep hydrated and avoid overuse of painkillers."
        ]
    }
}

# --------------------
# Doctor recommendation map
# --------------------
DOCTOR_MAP = {
    "diabetes": "Endocrinologist",
    "heart": "Cardiologist",
    "liver": "Hepatologist",
    "kidney": "Nephrologist"
}

# --------------------
# PDF report helper (ReportLab)
# --------------------
def generate_pdf(record, out_path):
    if not REPORTLAB_AVAILABLE:
        return False, "reportlab not available"
    try:
        c = canvas.Canvas(out_path, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width/2, height - 50, "Healthcare Assistant - Report")
        c.setFont("Helvetica", 10)
        y = height - 80
        for k, v in record.items():
            text = f"{k}: {v}"
            c.drawString(40, y, text)
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 40
        c.save()
        return True, None
    except Exception as e:
        return False, str(e)

# --------------------
# Simple plotting helper
# --------------------
def plot_values_vs_ranges(labels, values, ranges=None, title="Values vs Normal Range"):
    fig, ax = plt.subplots(figsize=(7,3))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_xticklabels(labels, rotation=45, ha='right')
    if ranges:
        for i, lab in enumerate(labels):
            if lab in ranges:
                low, high = ranges[lab]
                ax.plot([i], [ (low+high)/2 ], marker='o', color='red')
                ax.vlines(i, low, high, color='red', alpha=0.6, linewidth=2)
    st.pyplot(fig)
# Part 3/4 — Chatbot, translation, CSV upload helpers, explainability placeholder, UI start
# app.py — Part 3/4
# Chatbot, translation, CSV helpers and UI main (start)

# --------------------
# Chatbot (simple + optional GPT4All)
# --------------------
GPT_INSTANCE = None
if GPT4ALL_AVAILABLE:
    try:
        GPT_INSTANCE = GPT4All()  # will use system default local model if available
    except Exception:
        GPT_INSTANCE = None

def chatbot_answer(user_q):
    q = user_q.lower().strip()
    # rule-based quick answers
    if any(w in q for w in ["diet", "eat", "food"]):
        return "A balanced diet with vegetables, whole grains, lean proteins, and minimal added sugar is recommended."
    if any(w in q for w in ["exercise", "workout", "activity"]):
        return "Aim for 30 minutes of moderate exercise most days. If you have chest pain or dizziness, consult a doctor before exercise."
    if any(w in q for w in ["doctor", "see a doctor", "visit"]):
        return "If symptoms are severe or persistent, please seek a medical professional or emergency care."
    # fallback to GPT4All if available
    if GPT_INSTANCE is not None:
        try:
            with GPT_INSTANCE.chat_session() as sess:
                prompt = f"You are a helpful health assistant. Keep answers short and non-prescriptive. Question: {user_q}"
                resp = sess.generate(prompt, max_tokens=200)
                return resp
        except Exception:
            pass
    return "I can give general advice about diet, exercise, and when to consult a doctor. Please provide more details."

# --------------------
# Translation helper (deep_translator)
# --------------------
def translate_if_available(text, target_lang):
    if not TRANSLATION_AVAILABLE:
        return None
    try:
        tr = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return tr
    except Exception:
        return None

# --------------------
# CSV upload helper - read first row and map to features
# --------------------
def extract_inputs_from_csv(uploaded_file, features):
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[0] == 0:
            raise ValueError("CSV has no rows.")
        first = df.iloc[0]
        vals = []
        for f in features:
            if f in first:
                vals.append(first[f])
            else:
                # fallback: try lowercase match
                match = None
                for col in df.columns:
                    if col.strip().lower() == f.strip().lower():
                        match = first[col]
                        break
                vals.append(match if match is not None else 0)
        return vals
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return None

# --------------------
# Explainability placeholder (SHAP)
# --------------------
def explain_with_shap(name, features, inputs):
    # NOTE: computing SHAP can be slow and may require model compatibility.
    # This function is a placeholder — implement if you add shap and want to compute explanations.
    st.info("Explainability currently not active. To enable SHAP, install shap and ensure your model supports it.")
    return

# --------------------
# UI start
# --------------------
st.set_page_config(page_title="Smart Healthcare Assistant", layout="wide")
st.title("🩺 Smart Healthcare Assistant (Full)")

st.sidebar.header("Controls")
section = st.sidebar.selectbox("Section", ["Prediction","History","Chatbot","Upload CSV","Settings"])
selected_disease = st.sidebar.selectbox("Disease", ["diabetes","heart","liver","kidney"])
input_mode = st.sidebar.radio("Input Mode", ["Lab Reports (exact values)","Symptoms (I only know symptoms)"])
lang_choice = st.sidebar.selectbox("Display language (translation if available)", ["auto","en","ur","es","fr"])

# helper mapping for features (must match training order)
FEATURES_MAP = {
    "diabetes": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
    "heart": ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"],
    "liver": ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"],
    "kidney": ["Age","BloodPressure","SpecificGravity","Albumin","Sugar","RedBloodCells","PusCell","PusCellClumps","Bacteria","BloodGlucoseRandom","SerumCreatinine","Sodium","Potassium","Hemoglobin","PackedCellVolume","WhiteBloodCellCount","RedBloodCellCount"]
}
# Part 4/4 — UI logic for prediction, history, chatbot, upload, settings, and finishing touches
# app.py — Part 4/4
# UI logic (prediction flows), history display, chatbot and CSV upload sections

def display_health_tips(disease, pred, symptoms=None):
    key = "high" if pred == 1 else "low"
    st.subheader("Health Tips")
    for tip in TIPS[disease][key]:
        st.write("- " + tip)
    if symptoms:
        st.write("Symptoms considered:", ", ".join(symptoms))
    st.write("Recommendation: Specialist -", DOCTOR_MAP.get(disease, "General Physician"))

# ---------- Prediction Section ----------
if section == "Prediction":
    st.header("Prediction")
    st.write(f"Selected disease: **{selected_disease.upper()}**")
    features = FEATURES_MAP[selected_disease]

    # Allow CSV first-row usage (optional)
    use_csv = st.checkbox("Use first row from uploaded CSV (instead of manual inputs/symptoms)")
    uploaded_csv = None
    csv_inputs = None
    if use_csv:
        uploaded_csv = st.file_uploader("Upload CSV with matching columns", type=["csv"])
        if uploaded_csv:
            csv_inputs = extract_inputs_from_csv(uploaded_csv, features)
            if csv_inputs is None:
                st.error("CSV mapping failed. You can input manually or choose Symptoms mode.")
    inputs = None
    symptoms = []

    if use_csv and csv_inputs is not None:
        inputs = csv_inputs
        st.write("Inputs loaded from CSV (first row):")
        st.write(dict(zip(features, inputs)))
    else:
        if input_mode == "Lab Reports (exact values)":
            st.subheader("Enter lab/report values (exact)")
            cols = st.columns(2)
            values = []
            for i, f in enumerate(features):
                with cols[i % 2]:
                    if isinstance(f, str) and f.lower() in ["sex","gender","chestpaintype","restingecg","exercis eangina","fastingbs","redbloodcells","puscell","puscellclumps","bacteria"]:
                        val = st.number_input(f"{f}", step=1, value=0)
                    else:
                        # default float input
                        val = st.number_input(f"{f}", value=0.0)
                values.append(val)
            inputs = values
        else:
            st.subheader("Select symptoms (Symptoms -> approximate values will be generated)")
            symptoms = st.multiselect("Choose symptoms", SYMPTOM_CHOICES[selected_disease])
            inputs = symptoms_to_values(selected_disease, symptoms)
            st.write("Generated approximate inputs (estimates):")
            st.write(dict(zip(features, inputs)))

    if st.button("Run Prediction"):
        try:
            pred, prob = predict_model(selected_disease, inputs)
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            pred, prob = None, None

        # Apply rule-based overrides for Symptoms mode
        if input_mode == "Symptoms (I only know symptoms)":
            pred, prob = symptom_override(selected_disease, symptoms, pred, prob)

        # Save record
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "disease": selected_disease,
            "mode": "csv" if use_csv else ("lab" if input_mode.startswith("Lab") else "symptoms"),
            "symptoms": json.dumps(symptoms),
            "inputs": json.dumps(inputs),
            "prediction": int(pred) if pred is not None else None,
            "probability": float(prob) if prob is not None else None,
            "bookmark": False
        }
        save_history_record(record)

        # Display result
        if pred == 1:
            st.error(f"⚠️ High Risk for {selected_disease.upper()}")
        elif pred == 0:
            st.success(f"✅ Low Risk for {selected_disease.upper()}")
        else:
            st.info("Prediction unavailable.")

        if prob is not None:
            st.write(f"Estimated probability of positive class: {prob*100:.2f}%")

        # Show tips
        display_health_tips(selected_disease, pred, symptoms if symptoms else None)

        # Visualize first 6 features vs simple ranges
        try:
            labels = features[:6]
            values = [float(v) if v is not None else 0 for v in inputs[:6]]
            ranges = {}
            for lab in labels:
                if "Glucose" in lab:
                    ranges[lab] = (70, 140)
                elif "Cholesterol" in lab:
                    ranges[lab] = (120, 200)
                elif "BloodPressure" in lab or "RestingBP" in lab:
                    ranges[lab] = (80, 130)
                elif "BMI" in lab:
                    ranges[lab] = (18.5, 24.9)
                elif "SerumCreatinine" in lab:
                    ranges[lab] = (0.6, 1.3)
                else:
                    ranges[lab] = (0, max(values) * 1.2 if max(values) > 0 else 1)
            plot_values_vs_ranges(labels, values, ranges, title="First 6 input values (estimates)")
        except Exception:
            pass

        # Generate PDF if requested
        if st.checkbox("Generate downloadable PDF report"):
            rec_for_pdf = {
                "timestamp": record["timestamp"],
                "disease": selected_disease,
                "mode": record["mode"],
                "prediction": "High" if pred==1 else "Low",
                "probability": f"{prob*100:.2f}%" if prob is not None else "N/A",
                "doctor_recommendation": DOCTOR_MAP.get(selected_disease, "General Physician")
            }
            pdf_name = os.path.join(PROGRESS_DIR, f"report_{selected_disease}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf")
            ok, err = generate_pdf(rec_for_pdf, pdf_name)
            if ok:
                with open(pdf_name, "rb") as f:
                    st.download_button("📥 Download PDF", f, file_name=os.path.basename(pdf_name))
            else:
                st.error("PDF generation failed: " + str(err))

# ---------- History Section ----------
elif section == "History":
    st.header("Prediction History")
    df_hist = load_history_df()
    if df_hist.empty:
        st.info("No history records yet.")
    else:
        st.dataframe(df_hist.sort_values("timestamp", ascending=False).reset_index(drop=True))
        # allow simple operations: delete all, export CSV
        if st.button("Export History as CSV"):
            csv_out = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("Download history.csv", data=csv_out, file_name="history_export.csv")
        if st.button("Clear History"):
            try:
                os.remove(HISTORY_CSV)
                st.success("History cleared.")
                st.experimental_rerun()
            except Exception as e:
                st.error("Failed to clear history: " + str(e))

# ---------- Chatbot Section ----------
elif section == "Chatbot":
    st.header("Chatbot Guidance (General)")
    q = st.text_input("Ask a health question (e.g., 'What should I eat to reduce sugar?')", "")
    if st.button("Ask"):
        if not q.strip():
            st.warning("Type a question first.")
        else:
            answer = chatbot_answer(q)
            st.markdown("**Bot:** " + str(answer))
            # translation option
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

# ---------- Upload CSV Section ----------
elif section == "Upload CSV":
    st.header("Upload CSV to use lab results")
    st.write("Upload a CSV file where the first row contains lab/test values matching the selected disease feature names.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        features = FEATURES_MAP[selected_disease]
        vals = extract_inputs_from_csv(uploaded, features)
        if vals is not None:
            st.write("Using these values for prediction (first row):")
            st.write(dict(zip(features, vals)))
            if st.button("Predict from uploaded CSV"):
                try:
                    pred, prob = predict_model(selected_disease, vals)
                    st.write("Prediction:", "High risk" if pred==1 else "Low risk")
                    if prob is not None:
                        st.write(f"Probability: {prob*100:.2f}%")
                    # save
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
                    save_history_record(rec)
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

# ---------- Settings / fallback ----------
elif section == "Settings":
    st.header("Settings")
    st.write("Optional controls and troubleshooting")
    st.write(f"Translation available: {TRANSLATION_AVAILABLE}")
    st.write(f"ReportLab available (PDF): {REPORTLAB_AVAILABLE}")
    st.write(f"GPT4All available: {GPT4ALL_AVAILABLE}")
    if st.button("Reload models"):
        # not perfect but reload cache by clearing resource
        try:
            load_models.cache_clear()
        except Exception:
            pass
        st.experimental_rerun()

# End of app
