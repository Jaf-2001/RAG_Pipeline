"""Gradio web app. Also the entry point for Hugging Face Spaces."""

import os
import subprocess
import sys

import gradio as gr

from rag import INDEX_DIR, RAGAssistant

if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
    subprocess.run([sys.executable, "ingest.py"], check=True)

assistant = RAGAssistant()

EXAMPLES = [
    "What is the P50 net energy of the Thoothukudi Ridge project?",
    "How is wind speed extrapolated to hub height?",
    "Why do lenders use P90 instead of P50?",
    "What is the relation between GHI, DNI and DHI?",
]


def ask(question, hybrid):
    if not question.strip():
        return "Please enter a question.", ""
    answer, hits = assistant.answer(question, hybrid=hybrid)
    sources = "\n".join(f"[{n}] {c['source']} (page {c['page']}) | similarity {s:.2f}"
                        for n, (c, s) in enumerate(hits, 1))
    return answer, sources


with gr.Blocks(title="Wind & Solar RAG Assistant") as demo:
    gr.Markdown("## Wind & Solar Resource Assessment Assistant\n"
                "Answers are generated only from the indexed documents, with cited sources.")
    question = gr.Textbox(label="Question", placeholder="e.g. How is P90 calculated from P50?")
    hybrid = gr.Checkbox(label="Hybrid search (vector + BM25)", value=True)
    button = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Retrieved sources", lines=6)
    gr.Examples(EXAMPLES, inputs=question)
    button.click(ask, [question, hybrid], [answer, sources])
    question.submit(ask, [question, hybrid], [answer, sources])

if __name__ == "__main__":
    demo.launch()
