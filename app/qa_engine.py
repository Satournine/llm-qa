from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rag_pipeline import search_index
import os

hf_token = os.getenv("HF_TOKEN")

model_name = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token,
                                             torch_dtype="auto")

qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=True, max_new_tokens=512, temperature=0.7, top_p=0.9,repetition_penalty=1.1)

def format_prompt(context_chunks, question):
    context = "\n".join(context_chunks)
    prompt = (
            "You are a legal assistant. Use only the provided context below to answer the user's question.\n"
        " – Do not use outside knowledge.\n"
        "– Do not guess. If the context does not contain the answer, say: \"I'm not sure based on the current document.\"\n"
        "– Do not repeat the question or prompt in your answer.\n"
        "– Respond clearly and concisely.\n"
        "– Use legal language only when appropriate.\n"
        "– If the answer is a list, format it as bullet points.\n"
        "– Do not summarize or rephrase the entire context.\n"
        "– If the question is unrelated to the context, say you cannot answer.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

def generate_answer(question, top_k=3):
    chunks = search_index(question,top_k = top_k)
    prompt = format_prompt(chunks, question)
    result = qa_pipeline(prompt)[0]["generated_text"]
    answer = result.split("Answer: ")[-1].strip()
    return answer