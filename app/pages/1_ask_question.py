import  streamlit as st
from qa_engine import generate_answer

st.set_page_config(page_title="Ask a Question:")
st.title("Ask a question about the doc")

if not st.session_state.get("index"):
    st.warning("No document has been indexed yet.")
else:
    st.success(f"Indexed document: {st.session_state['doc_name']}")
    question = st.text_input("Your question")

    if question:
        with st.spinner("Thinking..."):
            answer = generate_answer(question)
            st.markdown("Answer: ")
            st.info(answer)