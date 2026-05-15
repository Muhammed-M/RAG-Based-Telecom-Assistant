import streamlit as st
import requests

# 1. UI Configuration
st.set_page_config(page_title="NileTel Support", page_icon="📞", layout="centered")

# --- INJECT RTL CSS FOR ARABIC ---
st.markdown("""
<style>
    /* Make all main text align right and read Right-to-Left */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, div {
        direction: rtl !important;
        text-align: right !important;
    }
    
    /* Fix the text input box so typing starts from the right */
    .stChatInputContainer textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    /* Force the sources and action text to stay LTR so it looks clean */
    .st-emotion-cache-1wmy9hl { 
        direction: ltr !important; 
        text-align: left !important; 
    }
</style>
""", unsafe_allow_html=True)
# ----------------------------------

st.title("📞 NileTel Customer Support Assistant")
st.markdown("مرحباً بك في خدمة عملاء نايل تيل. إزاي أقدر أساعدك النهاردة؟")

# 2. Define the FastAPI Endpoint URL
API_URL = "http://127.0.0.1:8000/chat"

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show metadata for the demo
        if message["role"] == "assistant":
            st.caption(f"🚀 Action Required: **{message.get('action', 'NO')}** | 📚 Sources: {', '.join(message.get('sources', []))}")

# 5. Handle User Input
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # Display user query in the UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. Call the FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("جاري التفكير..."):
            try:
                # Send the POST request to main.py
                response = requests.post(API_URL, json={"query": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Error in response")
                    needs_action = data.get("needs_action", "NO")
                    sources = data.get("sources", [])
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display the crucial metadata for Asya's demo
                    st.caption(f"🚀 Action Required: **{needs_action}** | 📚 Sources: {', '.join(sources) if sources else 'None'}")
                    
                    if needs_action == "YES":
                        st.success("✅ Ticket triggered via n8n webhook!")

                    # Save assistant response to memory
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "action": needs_action,
                        "sources": sources
                    })
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Failed to connect to FastAPI. Is main.py running?")