import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #e6edf3; }
  [data-testid="stAppViewContainer"] { background-color: #0d1117; }
  [data-testid="stHeader"] { background-color: #0d1117; }
  [data-testid="stSidebar"] { background-color: #0e1318; }
  [data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px;
  }
  [data-baseweb="select"] > div {
    background-color: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
  }
  hr { border-color: #30363d; }
  .pass-box { background:#0d2818; border:1px solid #3fb950; border-radius:10px; padding:18px 22px; font-size:15px; line-height:1.7; margin-top:8px; }
  .fail-box { background:#2d1115; border:1px solid #f85149; border-radius:10px; padding:18px 22px; font-size:15px; line-height:1.7; margin-top:8px; }
  .warn-box { background:#2b2006; border:1px solid #d29922; border-radius:10px; padding:18px 22px; font-size:15px; line-height:1.7; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

ORDINAL_COLS = [
    "Parental_Involvement", "Access_to_Resources", "Motivation_Level",
    "Family_Income", "Teacher_Quality", "Peer_Influence",
    "Parental_Education_Level", "Distance_from_Home",
]
BINARY_COLS = [
    "Extracurricular_Activities", "Internet_Access",
    "Learning_Disabilities", "School_Type", "Gender",
]
NUM_COLS = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity",
]
ALL_COLS = ORDINAL_COLS + BINARY_COLS + NUM_COLS

@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    return joblib.load("models/pipeline.joblib")

pipeline = load_pipeline()

def predict(inputs):
    df = pd.DataFrame([inputs])[ALL_COLS]
    return float(pipeline.predict(df)[0])

def find_min_hours(inputs):
    test = inputs.copy()
    current = int(inputs["Hours_Studied"])
    if predict(test) >= 60:
        for h in range(1, current + 1):
            test["Hours_Studied"] = h
            if predict(test) >= 60:
                return h
    for h in range(current + 1, 45):
        test["Hours_Studied"] = h
        if predict(test) >= 60:
            return h
    return None

st.markdown("""
<div style='text-align:center;padding:8px 0 24px'>
  <div style='font-family:monospace;font-size:12px;color:#58a6ff;letter-spacing:3px;margin-bottom:10px'>
    ML-POWERED · 6,607 STUDENTS · RANDOM FOREST
  </div>
  <h1 style='font-size:2.2rem;font-weight:800;color:#e6edf3;margin:0'>
    Will You <span style='color:#58a6ff'>Pass</span> Your Exam?
  </h1>
  <p style='color:#7d8590;font-size:14px;margin-top:8px'>
    Fill in your details — get your predicted score and minimum study hours to pass.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("### 📊 Key Performance Factors")
st.caption("These three have the highest impact on your score.")

col1, col2 = st.columns(2)
with col1:
    attendance       = st.slider("🏫 Attendance Rate (%) ★ #1", 60, 100, 80)
    previous_scores  = st.slider("📝 Previous Exam Score ★ #3", 50, 100, 75)
with col2:
    hours_studied    = st.slider("⏰ Hours Studied / Week ★ #2", 1, 44, 20)
    sleep_hours      = st.slider("😴 Sleep Hours / Night", 4, 10, 7)

st.divider()
st.markdown("### 🤝 Support & Environment")

col3, col4, col5 = st.columns(3)
with col3:
    motivation           = st.selectbox("💪 Motivation Level", ["Low","Medium","High"], index=1)
    peer_influence       = st.selectbox("👥 Peer Influence", ["Negative","Neutral","Positive"], index=1)
with col4:
    access_to_resources  = st.selectbox("📚 Access to Resources", ["Low","Medium","High"], index=1)
    parental_involvement = st.selectbox("👨‍👩‍👧 Parental Involvement", ["Low","Medium","High"], index=1)
with col5:
    family_income        = st.selectbox("💰 Family Income", ["Low","Medium","High"], index=1)
    teacher_quality      = st.selectbox("👩‍🏫 Teacher Quality", ["Low","Medium","High"], index=1)

col6, col7 = st.columns(2)
with col6:
    tutoring_sessions = st.slider("🧑‍🏫 Tutoring Sessions / Month", 0, 8, 1)
with col7:
    physical_activity = st.slider("🏃 Physical Activity (days/week)", 0, 6, 3)

st.divider()
st.markdown("### 🏠 Background Info")

col8, col9, col10 = st.columns(3)
with col8:
    school_type           = st.radio("🏫 School Type", ["Public","Private"])
    internet_access       = st.radio("🌐 Internet Access", ["Yes","No"])
with col9:
    extracurricular       = st.radio("⚽ Extracurricular", ["No","Yes"])
    learning_disabilities = st.radio("♿ Learning Disabilities", ["No","Yes"])
with col10:
    distance_from_home  = st.selectbox("📍 Distance from School", ["Far","Moderate","Near"], index=2)
    parental_education  = st.selectbox("🎓 Parental Education", ["High School","College","Postgraduate"])
    gender              = st.radio("👤 Gender", ["Male","Female"])

st.divider()

inputs = {
    "Hours_Studied": hours_studied, "Attendance": attendance,
    "Sleep_Hours": sleep_hours, "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions, "Physical_Activity": physical_activity,
    "Parental_Involvement": parental_involvement, "Access_to_Resources": access_to_resources,
    "Motivation_Level": motivation, "Family_Income": family_income,
    "Teacher_Quality": teacher_quality, "Peer_Influence": peer_influence,
    "Parental_Education_Level": parental_education, "Distance_from_Home": distance_from_home,
    "Extracurricular_Activities": extracurricular, "Internet_Access": internet_access,
    "Learning_Disabilities": learning_disabilities, "School_Type": school_type, "Gender": gender,
}

st.markdown("### 🎯 Your Results")

score     = round(predict(inputs))
passing   = score >= 60
min_hours = find_min_hours(inputs)

m1, m2, m3 = st.columns(3)
m1.metric("Predicted Score", f"{score} / 100")
m2.metric("Status", "✅ PASS" if passing else "❌ FAIL")

if passing:
    m3.metric("Min Hours to Pass", f"{min_hours} hrs/week", delta="You're already there!")
elif min_hours:
    gap = min_hours - hours_studied
    m3.metric("Min Hours to Pass", f"{min_hours} hrs/week", delta=f"+{gap} more hrs needed", delta_color="inverse")
else:
    m3.metric("Min Hours to Pass", "44+ hrs", delta="Focus on attendance!", delta_color="inverse")

pct = max(0.0, min(1.0, (score - 55) / 46))
st.markdown(f"**Score: {score} / 100**")
st.progress(pct)

if passing:
    quality = "top-tier 🌟" if score >= 80 else "solid 👍" if score >= 70 else "passing ✅"
    extra = (f" You could pass with just **{min_hours} hrs/week** — your extra effort builds a safety margin."
             if min_hours and min_hours < hours_studied else " Keep it up!")
    st.markdown(f'<div class="pass-box">✅ <strong>Predicted to PASS</strong> with <strong>{score}</strong> — {quality}!{extra}</div>', unsafe_allow_html=True)
elif min_hours:
    gap = min_hours - hours_studied
    st.markdown(f'<div class="warn-box">⚠️ <strong>Currently failing</strong> (score: {score}).<br><br>Study <strong>{gap} more hr{"s" if gap>1 else ""}/week</strong> — reaching <strong>{min_hours} hrs/week total</strong> — to pass.<br><br>💡 Your attendance ({attendance}%) and previous scores ({previous_scores}) are your biggest levers.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="fail-box">🚨 <strong>High risk of failing</strong> even at max hours (score: {score}).<br><br>Your attendance ({attendance}%) is the single biggest factor. Improve that first, then add tutoring.</div>', unsafe_allow_html=True)

st.divider()
st.markdown("### 📈 What Affects Your Score the Most?")
st.caption("Feature importances from the trained Random Forest model.")

importance_df = pd.DataFrame({
    "Factor": ["Attendance","Hours Studied","Previous Scores","Tutoring Sessions",
               "Access to Resources","Parental Involvement","Sleep Hours",
               "Physical Activity","Family Income","Peer Influence"],
    "Impact (%)": [38.5, 24.7, 8.8, 3.6, 3.0, 2.9, 2.7, 2.6, 1.8, 1.6]
}).sort_values("Impact (%)", ascending=True)
st.bar_chart(importance_df.set_index("Factor"))

st.divider()
st.markdown(
    "<div style='text-align:center;font-size:12px;color:#484f58;font-family:monospace'>"
    "Random Forest (200 trees) · 6,607 students · "
    "Built by <a href='https://ipycharmer.github.io' style='color:#58a6ff'>Ameer Hamza Nasir</a>"
    "</div>", unsafe_allow_html=True
)