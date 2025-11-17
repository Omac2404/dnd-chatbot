"""
Streamlit Chat Arayüzü
"""
import streamlit as st # type: ignore
from rag_pipeline_hybrid import HybridRAGPipeline
from config import config
import time

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="D&D RAG Chatbot",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #2b313e;
        border-left: 4px solid #4a9eff;
    }
    .bot-message {
        background-color: #1e2127;
        border-left: 4px solid #00d4aa;
    }
    .source-box {
        background-color: #262730;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .confidence-high {
        color: #00d4aa;
    }
    .confidence-medium {
        color: #ffa500;
    }
    .confidence-low {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_pipeline' not in st.session_state:
    with st.spinner("🚀 RAG Pipeline yükleniyor..."):
        st.session_state.rag_pipeline = HybridRAGPipeline()

# Sidebar
with st.sidebar:
    st.title("🎲 D&D RAG Chatbot")
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Ayarlar")
    
    top_k = st.slider(
        "Retrieval Chunk Sayısı",
        min_value=3,
        max_value=10,
        value=5,
        help="Vector DB'den kaç chunk alınsın?"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Bu değerin altında web araması yapılır"
    )
    
    show_sources = st.checkbox("Kaynakları Göster", value=True)
    show_confidence = st.checkbox("Confidence Skorunu Göster", value=True)
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 İstatistikler")
    st.metric("Toplam Soru", len(st.session_state.chat_history))
    
    if st.session_state.chat_history:
        llama_count = sum(1 for msg in st.session_state.chat_history 
                         if msg.get('method') == 'llama')
        st.metric("Llama Kullanımı", f"{llama_count}/{len(st.session_state.chat_history)}")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat area
st.title("🎲 D&D Rules Assistant")
st.caption("D&D 5e kuralları hakkında soru sorun!")

# Display chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 Sen:</strong><br>
            {message['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        confidence_class = (
            "confidence-high" if message['confidence'] > 0.8 
            else "confidence-medium" if message['confidence'] > 0.5 
            else "confidence-low"
        )
        
        method_icon = "🦙" if message['method'] == 'llama' else "☁️"
        method_text = "Llama (Local)" if message['method'] == 'llama' else "Claude + Web"
        
        bot_message = f"""
        <div class="chat-message bot-message">
            <strong>{method_icon} Assistant ({method_text}):</strong><br>
            {message['answer']}
        """
        
        # Confidence
        if show_confidence:
            bot_message += f"""
            <div style="margin-top: 0.5rem;">
                <span class="{confidence_class}">
                    📊 Confidence: {message['confidence']:.2%}
                </span>
            </div>
            """
        
        # Sources
        if show_sources and message.get('sources'):
            bot_message += "<div class='source-box'><strong>📚 Kaynaklar:</strong><br>"
            for i, source in enumerate(message['sources'][:3], 1):
                bot_message += f"""
                {i}. {source['source']} (Chunk {source['chunk_id']}) 
                - Similarity: {source.get('similarity', 0):.2f}<br>
                """
            bot_message += "</div>"
        
        # Web sources
        if message.get('web_enhanced') and message.get('web_sources'):
            bot_message += "<div class='source-box'><strong>🌐 Web Kaynakları:</strong><br>"
            for i, web_src in enumerate(message['web_sources'][:2], 1):
                bot_message += f"{i}. {web_src['url']}<br>"
            bot_message += "</div>"
        
        bot_message += "</div>"
        st.markdown(bot_message, unsafe_allow_html=True)

# Chat input
st.markdown("---")

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Sorunuzu yazın:",
            placeholder="Örn: What are ability scores?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.form_submit_button("Gönder 🚀", use_container_width=True)

if submit_button and user_input:
    # Add user message
    with st.spinner("🤔 Düşünüyorum..."):
        # Query RAG
        start_time = time.time()
        result = st.session_state.rag_pipeline.query(user_input, top_k=top_k)
        elapsed_time = time.time() - start_time
        
        # Add to history
        st.session_state.chat_history.append({
            'question': user_input,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'method': result['method_used'],
            'sources': result.get('sources', []),
            'web_enhanced': result.get('web_enhanced', False),
            'web_sources': result.get('web_sources', []),
            'response_time': elapsed_time
        })
    
    st.rerun()

# Footer
st.markdown("---")
st.caption("🎲 Powered by Llama 3.1 8B + Claude Haiku 4 + ChromaDB")
