from pdf_parser import chunk_text
from rag_pipeline import build_faiss_index, search_index
from qa_engine import generate_answer

def test_end_to_end():
    sample_text = """
    According to Article 13 of the GDPR, data subjects have the right to be informed.
    Organizations must provide clear information on how personal data is collected and used.
    """

    chunks = chunk_text(sample_text)
    assert len(chunks) > 0, "Chunking failed"
    build_faiss_index(chunks)
    question = "What rights does GDPR give to data subjects?"
    answer = generate_answer(question)
    assert answer and len(answer) > 5, "Answer is too short or empty"

    print("Integration test passed.")
    print(f"Sample Answer: {answer}")

if __name__ == "__main__":
    test_end_to_end()
