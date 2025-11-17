"""
Gelişmiş Streamlit Arayüzü - Ek Özellikler
Week 4 - Advanced Features
"""
import streamlit as st # type: ignore
from rag_pipeline_hybrid import HybridRAGPipeline
from config import config
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
import time
import json
from datetime import datetime
from pathlib import Path

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
    .metric-card {
        background-color: #1e2127;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #4a9eff;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_pipeline' not in st.session_state:
    with st.spinner("🚀 RAG Pipeline yükleniyor..."):
        try:
            st.session_state.rag_pipeline = HybridRAGPipeline()
        except Exception as e:
            st.error(f"❌ RAG Pipeline yüklenemedi: {e}")
            st.stop()

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
    show_response_time = st.checkbox("Response Time Göster", value=True)
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 İstatistikler")
    
    total_questions = len(st.session_state.chat_history)
    st.metric("Toplam Soru", total_questions)
    
    if st.session_state.chat_history:
        llama_count = sum(1 for msg in st.session_state.chat_history 
                         if msg.get('method') == 'llama')
        llama_percentage = (llama_count / total_questions * 100) if total_questions > 0 else 0
        
        st.metric(
            "Llama Kullanımı", 
            f"{llama_count}/{total_questions}",
            delta=f"{llama_percentage:.0f}%"
        )
        
        avg_confidence = sum(m['confidence'] for m in st.session_state.chat_history) / total_questions
        st.metric(
            "Ortalama Confidence",
            f"{avg_confidence:.2%}"
        )
        
        if 'response_time' in st.session_state.chat_history[-1]:
            avg_time = sum(m.get('response_time', 0) for m in st.session_state.chat_history) / total_questions
            st.metric(
                "Ortalama Response Time",
                f"{avg_time:.2f}s"
            )
    
    st.markdown("---")
    
    # Export Section
    st.subheader("💾 Export")
    
    if st.session_state.chat_history:
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "📥 JSON",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # CSV export
            df = pd.DataFrame([
                {
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Question': msg['question'],
                    'Method': msg['method'],
                    'Confidence': msg['confidence'],
                    'Response_Time': msg.get('response_time', 0),
                    'Answer_Length': len(msg['answer'])
                }
                for msg in st.session_state.chat_history
            ])
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 CSV",
                data=csv,
                file_name=f"chat_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Henüz export edilecek veri yok")
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
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
        
        # Response Time
        if show_response_time and 'response_time' in message:
            bot_message += f"""
            <div style="margin-top: 0.3rem;">
                <span style="color: #888;">
                    ⏱️ Response Time: {message['response_time']:.2f}s
                </span>
            </div>
            """
        
        # Sources
        if show_sources and message.get('sources'):
            bot_message += "<div class='source-box'><strong>📚 Kaynaklar:</strong><br>"
            for i, source in enumerate(message['sources'][:3], 1):
                similarity = source.get('similarity', 0)
                bot_message += f"""
                {i}. {source['source']} (Chunk {source['chunk_id']}) 
                - Similarity: {similarity:.2f}<br>
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
        try:
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
            
        except Exception as e:
            st.error(f"❌ Hata oluştu: {e}")

# Analytics Dashboard
st.markdown("---")

with st.expander("📊 Analytics Dashboard"):
    if st.session_state.chat_history:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_confidence = sum(m['confidence'] for m in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Ortalama Confidence</h4>
                <h2>{avg_confidence:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            llama_pct = sum(1 for m in st.session_state.chat_history if m['method'] == 'llama') / len(st.session_state.chat_history)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Llama Kullanım Oranı</h4>
                <h2>{llama_pct:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_time = sum(m.get('response_time', 0) for m in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Ortalama Response Time</h4>
                <h2>{avg_time:.2f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            web_enhanced = sum(1 for m in st.session_state.chat_history if m.get('web_enhanced', False))
            st.markdown(f"""
            <div class="metric-card">
                <h4>Web-Enhanced Queries</h4>
                <h2>{web_enhanced}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confidence histogram
        confidences = [m['confidence'] for m in st.session_state.chat_history]
        
        fig = go.Figure(data=[go.Histogram(
            x=confidences,
            nbinsx=10,
            marker_color='#4a9eff',
            name='Confidence'
        )])
        
        fig.update_layout(
            title="Confidence Score Dağılımı",
            xaxis_title="Confidence Score",
            yaxis_title="Soru Sayısı",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Method distribution pie chart
        methods = [m['method'] for m in st.session_state.chat_history]
        method_counts = pd.Series(methods).value_counts()
        
        fig2 = go.Figure(data=[go.Pie(
            labels=['Llama (Local)', 'Claude + Web'],
            values=[method_counts.get('llama', 0), method_counts.get('claude+web', 0)],
            marker_colors=['#00d4aa', '#4a9eff']
        )])
        
        fig2.update_layout(
            title="Method Kullanım Dağılımı",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info("Henüz soru sorulmadı. Analytics için birkaç soru sorun!")

# Footer
st.markdown("---")
st.caption("🎲 Powered by Llama 3.1 8B + Claude Haiku 4 + ChromaDB")