import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from flask import Flask, request, jsonify

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------- Step 1: Load Models and Datasets -------------------

# Load ParsBERT model and tokenizer
parsbert_model_name = "HooshvareLab/bert-base-parsbert-uncased"
parsbert_tokenizer = AutoTokenizer.from_pretrained(parsbert_model_name)
parsbert_model = AutoModelForQuestionAnswering.from_pretrained(parsbert_model_name).to(device)

# Load Llama model for Zero-Shot and Few-Shot
llama_model_name = "unsloth/llama-3-8b-bnb-4bit"
llama_pipeline = pipeline("question-answering", model=llama_model_name, tokenizer=llama_model_name, device=0 if torch.cuda.is_available() else -1)

# Load PQuAD dataset
dataset = load_dataset("Gholamreza/pquad")

# ------------------- Step 2: Data Preprocessing -------------------

def preprocess_function(examples):
    inputs = [q + " " + c for q, c in zip(examples["question"], examples["context"])]
    model_inputs = parsbert_tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    model_inputs["start_positions"] = [ans["answer_start"] for ans in examples["answers"]]
    model_inputs["end_positions"] = [ans["answer_end"] for ans in examples["answers"]]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# ------------------- Step 3: Fine-Tuning ParsBERT -------------------

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=parsbert_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

trainer.train()
parsbert_model.save_pretrained("./finetuned_parsbert")
parsbert_tokenizer.save_pretrained("./finetuned_parsbert")
print("ParsBERT Fine-Tuning Completed")

# ------------------- Step 4: Zero-Shot and Few-Shot Evaluation with Llama -------------------

def evaluate_llama(question, context, method="zero-shot"):
    if method == "zero-shot":
        result = llama_pipeline(question=question, context=context)
    else:
        result = llama_pipeline(question=question, context=context, top_k=5)
    return result

# Example usage:
question = "تهران پایتخت کجاست؟"
context = "تهران پایتخت ایران است و در شمال کشور قرار دارد."
print("Zero-Shot Result:", evaluate_llama(question, context))

# ------------------- Step 5: Retrieval-Augmented Generation (RAG) -------------------

def simple_retrieval(query, documents):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer().fit_transform(documents)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectorizer).flatten()
    best_doc_idx = similarity.argmax()
    return documents[best_doc_idx]

# Example usage for RAG:
documents = ["تهران پایتخت ایران است.", "شیراز شهر شعر و ادب است."]
query = "پایتخت ایران کجاست؟"
best_doc = simple_retrieval(query, documents)
print("Best Retrieved Document:", best_doc)

# ------------------- Step 6: Web Interface -------------------

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")
    context = data.get("context")

    # Use ParsBERT or Llama based on request
    model_type = data.get("model", "parsbert")
    if model_type == "parsbert":
        inputs = parsbert_tokenizer(question, context, return_tensors="pt").to(device)
        outputs = parsbert_model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer = parsbert_tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
    else:
        answer = evaluate_llama(question, context)["answer"]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
