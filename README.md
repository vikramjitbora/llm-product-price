
# Product Price Prediction from Descriptions (WIP)

**Predict product prices using Large Language Models (LLMs) and product descriptions.**  
This project aims to solve a real-world e-commerce problem — estimating the price of a product based purely on its textual description. It has applications in marketplaces, competitor analysis, and automated cataloging.

---

## 🚧 Status: Work in Progress

Currently building the pipeline to:
- Extract semantic features from product descriptions using LLM embeddings (OpenAI / SBERT / other)
- Train a regression model to predict price based on these embeddings
- Deploy as a lightweight API for real-time prediction

---

## 💡 Use Case Example

Input:
> `"Smartwatch with heart-rate monitor, GPS, and water resistance up to 50 meters"`

Output:
> `$79.50` (predicted)

---

## 🔧 Tech Stack (Planned / Partial)
- Python, scikit-learn
- OpenAI, HuggingFace Transformers
- FastAPI (for deployment)
- Gradio (for simple demo)


