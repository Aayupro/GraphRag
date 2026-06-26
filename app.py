# app.py
import streamlit as st
import time
from main_initializer import setup_system

# Configure page to leverage wide real-estate layouts
st.set_page_config(page_title="RAG Benchmarking UI", layout="wide")

st.markdown("# 🔬 Graph RAG vs Traditional Vector RAG Chatbot")
st.markdown("---")

# Cache the initial startup pipeline execution so it only builds indices once
@st.cache_resource
def initialize_pipeline():
    with st.spinner("[*] Initializing system cores, embeddings, and localized subgraphs..."):
        return setup_system()

try:
    retriever = initialize_pipeline()
    st.success("[✓] Subsystem cores linked successfully. Ready for comparison profiling.")
except Exception as e:
    st.error(f"Initialization failure: {str(e)}")
    st.stop()

# Persistent session state layout array for chat histories
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input chat execution tray at the bottom of the workspace
user_query = st.chat_input("Ask a multi-hop question (e.g., 'Trace Dr. Vance's connection to Matrix Foundry')")

if user_query:
    with st.spinner("Processing dual pipeline routing loops..."):
        # Run independent evaluation passes
        v_res = retriever.execute_vector_rag(user_query)
        g_res = retriever.execute_graph_rag(user_query)
        
        # Save historical trace step to state
        st.session_state.chat_history.append({
            "query": user_query,
            "vector": v_res,
            "graph": g_res
        })

# Render full contextual trace history arrays
for session in reversed(st.session_state.chat_history):
    st.info(f"**Query:** {session['query']}")
    
    # Establish split-screen view lanes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Traditional Vector RAG")
        # Display execution metrics
        st.metric(label="Total Execution Time", value=f"{session['vector']['total_time']:.4f} s")
        st.markdown(f"**Answer:** {session['vector']['answer']}")
        
        # Diagnostics inspector panel
        with st.expander("Inspect Retrieved Vector Chunks"):
            st.caption(f"Context Payload Dimensions: {len(session['vector']['context'])} characters")
            st.text_area("Raw Text Node Block", value=session['vector']['context'], height=150, key=f"v_{session['query']}")

    with col2:
        st.subheader("🟢 Custom Graph RAG")
        # Display execution metrics
        st.metric(label="Total Execution Time", value=f"{session['graph']['total_time']:.4f} s")
        st.markdown(f"**Answer:** {session['graph']['answer']}")
        
        # Diagnostics inspector panel
        with st.expander("Inspect Traversed Subgraph"):
            st.caption(f"Context Payload Dimensions: {len(session['graph']['context'])} characters")
            st.text_area("Extracted Triples Path Context", value=session['graph']['context'], height=150, key=f"g_{session['query']}")
            
    st.markdown("---")