import streamlit as st
import pandas as pd
import openai
import requests
import PyPDF2
import io
import os
import json
import speech_recognition as sr
from datetime import datetime
import base64
from pathlib import Path
import fitz  # PyMuPDF

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ValleyHelps - HR Assistant",
    page_icon="images/valleyhelpslogo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS with Accessibility Improvements ───────────────────────────────
st.markdown("""
<style>
    /* General Styling */
    .main {
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }

    /* Accessibility: High contrast and focus styles */
    button, a, input, select, textarea {
        outline: 2px solid transparent !important;
        transition: outline 0.2s ease;
    }
    button:focus, a:focus, input:focus, select:focus, textarea:focus {
        outline: 2px solid #0068c9 !important;
    }

    /* Chat Message Styling */
    .user-bubble {
        background-color: #e6f7ff;
        border-radius: 15px;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-left: 4px solid #0068c9;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .assistant-bubble {
        background-color: #f0f7ff;
        border-radius: 15px;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    /* Voice Button Styling */
    button[title="Click to speak"] {
        border-radius: 50%;
        height: 100px; 
        width: 100px; 
        font-size: 36px;
        background-color: #0068c9;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    button[title="Click to speak"]:hover {
        background-color: #005bb5;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }

    /* Header Styling */
    .header-container {
        display: flex;
        align-items: center;
        background-color: #f0f7ff;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* KB Status Indicator */
    .kb-status {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 10px;
    }
    .kb-active {
        background-color: #d4edda;
        color: #155724;
    }
    .kb-inactive {
        background-color: #f8d7da;
        color: #721c24;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 10px;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 20px;
    }

    /* File Uploader */
    .uploadedFile {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
    }

    /* Chat Input */
    .stTextInput input {
        border-radius: 20px;
        padding: 10px 15px;
        border: 1px solid #ced4da;
    }

    /* Career Plan Styling */
    .match-analysis {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #5bc0de;
        margin-bottom: 15px;
    }

    .event-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border-left: 3px solid #0068c9;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column;
            align-items: flex-start;
        }
        .header-title {
            margin-left: 0;
            margin-top: 10px;
        }
        button[title="Click to speak"] {
            height: 80px;
            width: 80px;
            font-size: 30px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
def get_openai_client():
    return openai.OpenAI(api_key=st.session_state.openai_api_key)

# ─── Load API Key ───────────────────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY", "")
st.session_state.openai_api_key = openai_api_key

# ─── Session State Defaults ────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""
if "audio_mode" not in st.session_state:
    st.session_state.audio_mode = False
if "kb_text_1" not in st.session_state:
    st.session_state.kb_text_1 = ""
if "kb_text_2" not in st.session_state:
    st.session_state.kb_text_2 = ""
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "kb_names" not in st.session_state:
    st.session_state.kb_names = ["", ""]
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "match_analysis" not in st.session_state:
    st.session_state.match_analysis = None
if "career_goal" not in st.session_state:
    st.session_state.career_goal = None
if "events_data" not in st.session_state:
    st.session_state.events_data = None

# ─── PDF Helpers ────────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    try:
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_text = ""
        for i in range(len(pdf)):
            page = pdf.load_page(i)
            file_text += page.get_text("text")
        return file_text
    except Exception as e:
        try:
            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e2:
            st.error(f"Error extracting text from PDF: {e2}")
            return None

def download_pdf_from_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return io.BytesIO(r.content)
        return None
    except Exception as e:
        st.error(f"Error downloading PDF: {e}")
        return None

# ─── OpenAI Chat & TTS ──────────────────────────────────────────────────────────
def query_openai(prompt, history, model="gpt-4o-mini", sys_prompt=None):
    try:
        client = get_openai_client()
        if sys_prompt is None:
            sys = "You are ValleyHelps, an HR assistant chatbot for Valley Water. Be helpful, friendly, and concise. When someone brings up a job or resume, inform them of the career planning tool. If the topic is workshops or events, bring up the events exploration tool."
            if st.session_state.knowledge_base:
                sys += "\n\nKnowledge Base:\n" + st.session_state.knowledge_base
        else:
            sys = sys_prompt

        msgs = [{"role":"system","content":sys}]
        for e in history:
            msgs.append({"role":e["role"], "content":e["content"]})
        msgs.append({"role":"user","content":prompt})

        with st.spinner("ValleyHelps is thinking..."):
            resp = client.chat.completions.create(
                model=model, messages=msgs, max_tokens=500, temperature=0.7
            )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def text_to_speech(text):
    try:
        client = get_openai_client()
        with st.spinner("Converting to speech..."):
            audio = client.audio.speech.create(model="tts-1", voice="sage", input=text)
        os.makedirs("cache", exist_ok=True)
        audio_path = f"cache/response_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio.content)
        return audio_path
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# ─── Voice Recording ────────────────────────────────────────────────────────────
def record_audio():
    try:
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as src:
            r.adjust_for_ambient_noise(src, duration=1)
            r.pause_threshold = 1.0
            with st.spinner("🎙️ Listening..."):
                audio = r.listen(src, timeout=5, phrase_time_limit=15)
        os.makedirs("cache", exist_ok=True)
        audio_path = f"cache/recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio.get_wav_data())
        return audio_path
    except Exception as e:
        st.error(f"Recording Error: {e}")
        return None

def speech_to_text(path):
    try:
        client = get_openai_client()
        with open(path, "rb") as f:
            with st.spinner("Transcribing your message..."):
                t = client.audio.transcriptions.create(model="whisper-1", file=f)
        return t.text
    except Exception as e:
        st.error(f"Whisper Error: {e}")
        return ""

def get_base64_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

def get_html_player(file_path):
    audio_base64 = get_base64_audio(file_path)
    return f"""
    <audio controls>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """

# ─── Cache Management ───────────────────────────────────────────────────────────
Path("cache").mkdir(exist_ok=True)

def cleanup_cache(current_audio_path=None):
    try:
        audio_files = list(Path("cache").glob("*.mp3")) + list(Path("cache").glob("*.wav"))
        for f in audio_files:
            if current_audio_path and f.samefile(current_audio_path):
                continue
            f.unlink()
    except Exception as e:
        print(f"Error cleaning cache: {e}")

# ─── Career Planning Functions ───────────────────────────────────────────────────
def analyze_event_relevance(events, match_analysis, career_goal):
    relevant_events = []
    for _, event in events.iterrows():
        prompt = f"""
        The user has the following career goal: "{career_goal}". Based on the following match analysis:
        "{match_analysis}", assess if the following event would help the user achieve their goal:

        Event Details:
        Name: {event['Event Name']}
        Description: {event['Description']}

        Respond with "Yes" if the event is relevant, and "No" if it is not.
        """
        completion = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intelligent career planner."},
                {"role": "user", "content": prompt},
            ]
        )
        response = completion.choices[0].message.content.strip()
        if response.lower() == "yes":
            relevant_events.append(event)
    return relevant_events

# ─── Sidebar with Improved Organization ─────────────────────────────────────────
with st.sidebar:
    
    st.markdown("<h2 style='margin-top:-5px'></h2>", unsafe_allow_html=True)

    if not openai_api_key:
        st.error("⚠️ Please set the OPENAI_API_KEY environment variable.")
    else:
        st.success("✅ API Key loaded")

    st.divider()

    # Settings Section
    with st.expander("⚙️ Settings", expanded=True):
        # Theme Toggle
        theme_label, theme_toggle = st.columns([3, 1])
        with theme_label:
            st.markdown("#### Theme")
        with theme_toggle:
            if st.button("🌙" if st.session_state.theme == "light" else "☀️", key="theme_toggle"):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
                st.rerun()

        # Voice Mode Toggle
        voice_label, voice_toggle = st.columns([3, 1])
        with voice_label:
            st.markdown("#### Voice Interaction")
        with voice_toggle:
            voice_enabled = st.toggle("", st.session_state.audio_mode, key="voice_toggle")
            if voice_enabled != st.session_state.audio_mode:
                st.session_state.audio_mode = voice_enabled
                st.rerun()

        if st.session_state.audio_mode:
            st.info("💡 Click the microphone button below the chat to speak your questions.")

    st.divider()

    # Knowledge Base Management
    with st.expander("📚 Knowledge Base", expanded=True):
        kb_tabs = st.tabs(["KB 1", "KB 2"])
        for i, tab in enumerate(kb_tabs):
            with tab:
                if i == 0:
                    kb_option_key = "kb1_option"
                    pdf_url_key = "pdf_url_1"
                    load_url_key = "load_url_1"
                    file_uploader_key = "ups1"
                    proc_key = "proc1"
                    kb_text = st.session_state.kb_text_1
                else:
                    kb_option_key = "kb2_option"
                    pdf_url_key = "pdf_url_2"
                    load_url_key = "load_url_2"
                    file_uploader_key = "ups2"
                    proc_key = "proc2"
                    kb_text = st.session_state.kb_text_2

                kb_option = st.radio(
                    "Input method:",
                    ["URL", "File Upload", "None"],
                    key=kb_option_key,
                    horizontal=True
                )

                if kb_option == "URL":
                    pdf_url = st.text_input(
                        "PDF URL",
                        value="" if i > 0 else "https://s3.us-west-1.amazonaws.com/valleywater.org.us-west-1/s3fs-public/Employees%20Association%20MOU%202022-2025.docx%20%283%29.pdf",
                        key=pdf_url_key,
                        placeholder="Enter URL to PDF document"
                    )
                    if st.button("📥 Load PDF", key=load_url_key) and pdf_url:
                        with st.spinner("Downloading PDF..."):
                            f = download_pdf_from_url(pdf_url)
                        if f:
                            with st.spinner("Extracting text..."):
                                if i == 0:
                                    st.session_state.kb_text_1 = extract_text_from_pdf(f)
                                    st.session_state.kb_names[0] = pdf_url.split("/")[-1]
                                else:
                                    st.session_state.kb_text_2 = extract_text_from_pdf(f)
                                    st.session_state.kb_names[1] = pdf_url.split("/")[-1]
                                st.success("✅ PDF loaded successfully")
                        else:
                            st.error("❌ Failed to fetch PDF")
                elif kb_option == "File Upload":
                    uploaded_files = st.file_uploader(
                        "Upload PDFs (max 3)",
                        type="pdf",
                        accept_multiple_files=True,
                        key=file_uploader_key
                    )
                    if uploaded_files:
                        files = uploaded_files[:3]
                        for file in files:
                            st.markdown(f"📄 **{file.name}** ({round(file.size/1024, 1)} KB)")
                        if st.button("📥 Process Files", key=proc_key):
                            with st.spinner("Extracting text from PDFs..."):
                                text = ""
                                for p in files:
                                    text += extract_text_from_pdf(p) + "\n\n"
                                if i == 0:
                                    st.session_state.kb_text_1 = text
                                    st.session_state.kb_names[0] = ", ".join([f.name for f in files])
                                else:
                                    st.session_state.kb_text_2 = text
                                    st.session_state.kb_names[1] = ", ".join([f.name for f in files])
                                st.success(f"✅ {len(files)} file(s) processed successfully")

                if kb_text:
                    st.info(f"📚 KB {i+1} loaded ({len(kb_text)} chars)")
                    if st.button(f"❌ Clear KB {i+1}"):
                        if i == 0:
                            st.session_state.kb_text_1 = ""
                            st.session_state.kb_names[0] = ""
                        else:
                            st.session_state.kb_text_2 = ""
                            st.session_state.kb_names[1] = ""
                        st.rerun()

    # Admin Section
    with st.expander("🔒 Admin Section", expanded=False):
        uploaded_resources = st.file_uploader(
            "Upload Career Resources (PDF or Text) Please omit any personal details.",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="admin_resources"
        )
        resource_texts = []
        if uploaded_resources:
            for resource in uploaded_resources:
                if resource.type == "application/pdf":
                    resource_texts.append(extract_text_from_pdf(resource))
                elif resource.type == "text/plain":
                    resource_texts.append(resource.read().decode("utf-8"))
            st.success(f"✅ {len(uploaded_resources)} resource(s) loaded")

        uploaded_events_csv = st.file_uploader("Upload Events Data (CSV)", type=["csv"], key="events_csv")
        if uploaded_events_csv:
            try:
                st.session_state.events_data = pd.read_csv(uploaded_events_csv)
                st.success("✅ Events data uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")

    # Combine KBs
    st.session_state.knowledge_base = "\n\n".join(
        txt for txt in (st.session_state.kb_text_1, st.session_state.kb_text_2) if txt
    )

    st.divider()

    # Chat History Management
    with st.expander("💾 Chat Management", expanded=False):
        st.markdown("#### Import Chat History")
        chat_json = st.text_area(
            "Paste chat JSON here",
            placeholder='[{"role":"user","content":"What\'s the vacation policy?"},...]',
            height=100
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load History"):
                try:
                    st.session_state.chat_history = json.loads(chat_json)
                    st.success("✅ Chat history loaded")
                except:
                    st.error("❌ Invalid JSON format")

        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.success("🧹 Chat cleared")
                st.rerun()

        st.markdown("#### Export Chat")
        if st.button("📤 Export History"):
            payload = json.dumps(st.session_state.chat_history)
            st.download_button(
                "📥 Download JSON",
                payload,
                file_name=f"valleyhelps_chat_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json"
            )

# ─── Main App UI ───────────────────────────────────────────────────────────────
# Custom Header
st.image("images/valleyhelpsbannerfinalver1.png", width=800)
st.markdown("<h1 style='margin-bottom:0'>HR Assistant</h1>", unsafe_allow_html=True)

# Status Indicators
status_col1, status_col2 = st.columns(2)
with status_col1:
    if st.session_state.knowledge_base:
        kb_names = [name for name in st.session_state.kb_names if name]
        if kb_names:
            st.markdown(f"<div class='kb-status kb-active'>📚 Using: {', '.join(kb_names)}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kb-status kb-active'>📚 Knowledge Base Active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='kb-status kb-inactive'>⚠️ No Knowledge Base</div>", unsafe_allow_html=True)
with status_col2:
    if st.session_state.audio_mode:
        st.markdown("<div class='kb-status kb-active'>🎙️ Voice Mode Active</div>", unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📈 Career Planning", "🎉 Events Exploration"])

# Tab 1: Chat Assistant
with tab1:
    chat_container = st.container()
    audio_player_container = st.container()

    # Clear previous audio player before displaying a new one
    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        with audio_player_container:
            audio_player_container.empty()  # Clear previous audio players
            st.markdown(get_html_player(st.session_state.audio_path), unsafe_allow_html=True)

    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-bubble'><strong>You:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-bubble'><strong>ValleyHelps:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)

    if not st.session_state.audio_mode:
        user_input = st.chat_input("Ask ValleyHelps anything about HR...", key="text_input")
        if user_input:
            st.markdown(f"<div class='user-bubble'><strong>You:</strong><br>{user_input}</div>", unsafe_allow_html=True)
            st.session_state.chat_history.append({"role":"user","content":user_input})
            resp = query_openai(user_input, st.session_state.chat_history[:-1])
            st.session_state.last_response = resp
            st.markdown(f"<div class='assistant-bubble'><strong>ValleyHelps:</strong><br>{resp}</div>", unsafe_allow_html=True)
            st.session_state.chat_history.append({"role":"assistant","content":resp})
            st.rerun()
    else:
        if not openai_api_key:
            st.warning("⚠️ Voice mode requires OpenAI API key")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🎤", use_container_width=True, help="Click to speak", key="voice_button"):
                    with st.spinner("Listening to your question..."):
                        wav = record_audio()
                    if wav:
                        text_input = speech_to_text(wav)
                        if text_input:
                            st.markdown(f"<div class='user-bubble'><strong>You:</strong><br>{text_input}</div>", unsafe_allow_html=True)
                            st.session_state.chat_history.append({"role":"user","content":text_input})
                            resp = query_openai(text_input, st.session_state.chat_history[:-1])
                            st.session_state.last_response = resp
                            st.markdown(f"<div class='assistant-bubble'><strong>ValleyHelps:</strong><br>{resp}</div>", unsafe_allow_html=True)
                            st.session_state.chat_history.append({"role":"assistant","content":resp})
                            mp3_path = text_to_speech(resp)
                            if mp3_path:
                                # Clean up previous audio files before setting the new one
                                if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                                    cleanup_cache(mp3_path)
                                st.session_state.audio_path = mp3_path
                                with audio_player_container:
                                    audio_player_container.empty()  # Clear previous audio players
                                    st.markdown(get_html_player(mp3_path), unsafe_allow_html=True)
                                try:
                                    if os.path.exists(wav) and wav != mp3_path:
                                        os.unlink(wav)
                                except Exception as e:
                                    print(f"Error cleaning up recording: {e}")
                            else:
                                st.error("❌ Couldn't generate speech")
                        else:
                            st.error("❌ Couldn't transcribe your message")
                            try:
                                if os.path.exists(wav):
                                    os.unlink(wav)
                            except:
                                pass

# Tab 2: Career Planning
with tab2:
    st.header("Career Planning and Growth")
    resume_col, job_col = st.columns(2)
    with resume_col:
        st.image("images/resume_icon1.png", caption="Resume", width=150)
        uploaded_resume = st.file_uploader("Upload Resume (PDF or Text)", type=["pdf", "txt"], key="career_resume")
    with job_col:
        st.image("images/job_description_icon1.png", caption="Job Description", width=150)
        uploaded_job_description = st.file_uploader("Upload Desired Job Description (PDF or Text)", type=["pdf", "txt"], key="career_job")

    if uploaded_resume and uploaded_job_description and st.session_state.match_analysis is None:
        resume_text = extract_text_from_pdf(uploaded_resume)
        job_desc_text = extract_text_from_pdf(uploaded_job_description)
        if resume_text and job_desc_text:
            st.info("🔄 Analyzing Resume and Job Description...")
            prompt = f"""
            Compare the following resume to the desired job description and provide a match score (0-100). 
            Additionally, identify missing skills or qualifications and suggest ways to bridge the gap.

            Resume:
            {resume_text}

            Job Description:
            {job_desc_text}
            """
            with st.spinner("AI is analyzing your documents..."):
                client = get_openai_client()
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an HR system focusing on internal employee growth."},
                        {"role": "user", "content": prompt},
                    ]
                )
                st.session_state.match_analysis = completion.choices[0].message.content.strip()
                st.success("✅ Analysis complete!")

    if st.session_state.match_analysis:
        st.markdown("### 📊 Match Analysis and Suggestions")
        with st.container(height=300):
            st.markdown(f'<div class="match-analysis">{st.session_state.match_analysis}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload your resume and desired job description for analysis.")

    if st.session_state.match_analysis:
        explore_options = st.checkbox("I would like to explore suggestions for growth.")
        if explore_options:
            career_goal = st.selectbox(
                "🎯 Select a Career Goal:",
                ["Job Performance Improvement", "Career Advancement", "Professional Growth"],
                key="career_goal_select"
            )
            st.session_state.career_goal = career_goal
            resources_prompt = f"""
            Based on the selected career goal: {career_goal}, and the match analysis above, suggest tailored growth plans.

            Include these available resources from the company:
            {''.join(resource_texts) if resource_texts else "No resources uploaded."}
            """
            with st.spinner("Generating personalized development plan..."):
                client = get_openai_client()
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an HR system providing career development advice."},
                        {"role": "user", "content": resources_prompt},
                    ]
                )
                analysis = completion.choices[0].message.content.strip()
                st.subheader("🚀 Career Development Suggestions")
                st.write(analysis)

# Tab 3: Events Exploration
with tab3:
    st.header("Events Exploration")
    st.write("Explore professional development events to help achieve your career goals:")

    image_col1, image_col2, image_col3 = st.columns(3)
    with image_col1:
        st.image("images/workshop_image.png", caption="Workshop", use_container_width=True)
    with image_col2:
        st.image("images/seminar_image.png", caption="Seminar", use_container_width=True)
    with image_col3:
        st.image("images/conference_image.png", caption="Conference", use_container_width=True)

    if st.session_state.events_data is not None:
        if st.session_state.career_goal and st.session_state.match_analysis:
            with st.spinner("Finding relevant events for your career goals..."):
                relevant_events = analyze_event_relevance(
                    st.session_state.events_data,
                    st.session_state.match_analysis,
                    st.session_state.career_goal
                )
                if relevant_events:
                    st.subheader("🎯 Events Matching Your Career Goals")
                    for event in relevant_events:
                        with st.container():
                            st.markdown(f"""
                            <div class="event-card">
                                <h3>{event['Event Name']}</h3>
                                <p><strong>Date:</strong> {event.get('Date', 'TBD')}</p>
                                <p><strong>Location:</strong> {event.get('Location', 'TBD')}</p>
                                <p>{event['Description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    event_details = "\n".join(
                        [f"{event['Event Name']}: {event['Description']}" for event in relevant_events]
                    )
                    summary_prompt = f"""
                    Based on the user's match analysis and career goal of "{st.session_state.career_goal}", the following events were identified as relevant:

                    {event_details}

                    Generate a concise summary explaining how these events collectively support the user's career development and help them achieve their goals.
                    """
                    client = get_openai_client()
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert career planner."},
                            {"role": "user", "content": summary_prompt},
                        ]
                    )
                    summary = completion.choices[0].message.content.strip()
                    st.subheader("📝 Summary of Recommendations")
                    st.write(summary)
                else:
                    st.warning("⚠️ No matching events found for your career goals.")
        else:
            st.info("ℹ️ Please complete the Career Planning tab first to see relevant events.")
    else:
        st.warning("⚠️ No events data found. Please upload a CSV file in the admin sidebar.")

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 ValleyHelps HR Assistant</p>
</div>
""", unsafe_allow_html=True)