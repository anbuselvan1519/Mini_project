import streamlit as st
from PIL import Image

# Local imports
from ai_engine.disease_classifier import predict_disease
from ai_engine.conversation_ai import chat_with_ai as ask_ai

# ---------------------------
# Translation dictionary
# ---------------------------
translations = {
    "en": {
        "title": "🌿 Edge AI for Plant Disease Detection",
        "subtitle": "Upload a leaf image to detect plant disease, get severity details, and ask prevention questions — all **offline**.",
        "upload": "Upload a leaf image",
        "analyzing": "Analyzing image...",
        "ask_title": "👁️‍🗨️ Ask AI About This Disease",
        "ask_input": "Ask a question about the disease (press Enter to submit):",
        "ai_answer": "**AI Answer:**"
    },
    "ta": {
        "title": "🌿 தாவர நோய் கண்டறிதல் AI",
        "subtitle": "இலை படத்தை பதிவேற்றி நோயை கண்டறியவும், அதன் தீவிரத்தைக் காணவும், தடுப்பு கேள்விகளை கேட்கவும் — அனைத்தும் **ஆஃப்லைனில்**.",
        "upload": "இலை படத்தை பதிவேற்றவும்",
        "analyzing": "படம் பகுப்பாய்வு செய்யப்படுகிறது...",
        "ask_title": "👁️‍🗨️ இந்த நோயைப் பற்றி AI-யிடம் கேளுங்கள்",
        "ask_input": "நோய் குறித்து கேள்வி கேளுங்கள் (Enter அழுத்தவும்):",
        "ai_answer": "**AI பதில்:**"
    },
    "hi": {
        "title": "🌿 पौधों की बीमारी पहचान AI",
        "subtitle": "पत्ते की तस्वीर अपलोड करें, बीमारी की पहचान करें, गंभीरता देखें और रोकथाम से जुड़े सवाल पूछें — सब कुछ **ऑफलाइन**.",
        "upload": "पत्ते की तस्वीर अपलोड करें",
        "analyzing": "चित्र का विश्लेषण हो रहा है...",
        "ask_title": "👁️‍🗨️ इस बीमारी के बारे में AI से पूछें",
        "ask_input": "बीमारी के बारे में प्रश्न पूछें (Enter दबाएँ):",
        "ai_answer": "**AI उत्तर:**"
    }
}

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="🌱 Plant Disease Detection AI (Offline)",
    layout="wide"
)

# Sidebar: Language selection only
lang = st.sidebar.selectbox("🌐 Choose Language", ["en", "ta", "hi"])

# Title & Subtitle
st.title(translations[lang]["title"])
st.write(translations[lang]["subtitle"])

# ---------------------------
# Image Upload Section
# ---------------------------
uploaded_file = st.file_uploader(translations[lang]["upload"], type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Run Prediction
    with st.spinner(translations[lang]["analyzing"]):
        prediction, confidence = predict_disease(image)

    # Show results
    st.success(f"✅ Predicted: {prediction} (Confidence: {confidence*100:.2f}%)")

    # ---------------------------
    # AI Answer Section
    # ---------------------------
    st.subheader(translations[lang]["ask_title"])

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    def ask_ai_callback():
        if st.session_state.user_question.strip():
            with st.spinner("🤖 Thinking..."):
                lang_instruction = {
                    "en": "Answer in English.",
                    "ta": "பதில் தமிழில் கொடு.",
                    "hi": "उत्तर हिंदी में दें।"
                }[lang]

                context_question = (
                    f"The detected plant disease is: {prediction} "
                    f"(confidence: {confidence*100:.2f}%). "
                    f"Provide the following in your answer:\n"
                    f"1. Confirm the prediction (disease name)\n"
                    f"2. A short explanation about this disease\n"
                    f"3. Prevention and treatment steps\n\n"
                    f"{lang_instruction}\n\n"
                    f"User question: {st.session_state.user_question}"
                )
                st.session_state.ai_answer = ask_ai(context_question)

    st.text_input(
        translations[lang]["ask_input"],
        key="user_question",
        on_change=ask_ai_callback
    )

    if "ai_answer" in st.session_state:
        st.write(translations[lang]["ai_answer"])
        st.write(st.session_state.ai_answer)
