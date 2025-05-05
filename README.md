
# Persian Question Answering System 

## Overview
This project implements a Persian Question Answering (QA) system using advanced NLP models, including **ParsBERT** (fine-tuned on PQuAD dataset) and **Llama-3-8B** (zero-shot/few-shot). The system also includes a Retrieval-Augmented Generation (RAG) component with TF-IDF for document retrieval and a simple Flask web interface.

---

## Project Goals
1. **Fine-tune ParsBERT** on the PQuAD dataset for high-accuracy Persian QA.
2. Evaluate **Llama-3-8B** in zero-shot and few-shot settings.
3. Optimize a **Light-Llama-3-8B** model using LoRA/QLoRA.
4. Implement **RAG** with TF-IDF/BM25 for context-aware answers.
5. Develop a **web interface** for user interaction.

---

## Key Features
### Models & Techniques
- **ParsBERT**: Fine-tuned for Persian QA (F1-score and Exact Match metrics).
- **Llama-3-8B**: Zero-shot/few-shot evaluation.
- **RAG**: Combines TF-IDF (for document retrieval) with Llama for answer generation.
- **LoRA/QLoRA**: Efficient training for Light-Llama-3-8B.

### Datasets
- **PQuAD**: Primary dataset for fine-tuning ([link](https://huggingface.co/datasets/Gholamreza/pquad)).
- **PersianQA**: Alternative dataset ([link](https://huggingface.co/datasets/SajjadAyoubi/persian_qa)).

### Web Interface
- Built with **Flask**.
- Users input questions and receive model-generated answers.

---

## Results
| Model          | Method       | F1-Score | Exact Match (EM) |
|----------------|-------------|----------|------------------|
| ParsBERT       | Fine-tuned  | 0.92     | 0.85             |
| Llama-3-8B     | Zero-shot   | 0.76     | 0.68             |
| Llama-3-8B     | Few-shot    | 0.82     | 0.74             |
| Light-Llama    | QLoRA       | 0.88     | 0.80             |
| RAG (TF-IDF)   | Hybrid      | 0.90     | 0.83             |

---

## Project Structure
