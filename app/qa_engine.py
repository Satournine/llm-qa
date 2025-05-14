from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from rag_pipeline import search_index

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def format_prompt(context_chunks, question):
    context = "\n".join(context_chunks)
    prompt = (
        "You are a helpful, accurate legal assistant. Your job is to answer questions based only on the provided context.\n"
        "If the context does not contain enough information, say:\n"
        "\"I'm not sure based on the current document.\"\n\n"
        "Do not make up answers or guess. Use only what is supported by the text.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

def generate_answer(question, top_k=3):
    chunks = search_index(question, top_k=top_k)
    prompt = format_prompt(chunks, question)
    result = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return result


if __name__ == "__main__":
    while True:
        q=input("Ask a question")
        if q.lower() == "exit":
            break
        answer = generate_answer(q)
        print("\nAnswer\n", answer)
        print("-" * 160)