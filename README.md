# Single Local Document Query Assistant 📄

## Overview

This streamlit app allows users to upload documents and ask specific questions about their contents. Utilizing state-of-the-art Deepseek R1 Reasoning Language Models, this application provides intelligent, context-aware responses by analyzing the uploaded document.

## Features

- 🔍 Multi-document support
  - Upload PDF, Word, Text, and Markdown files
  - Extract and analyze document contents

- 🤖 Advanced AI Models
  - Multiple DeepSeek language models
  - Intelligent query processing
  - Context-aware response generation

- 📊 Response Evaluation
  - Grounding score assessment
  - Quality score verification
  - Detailed response analysis

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abdul-456/SLM-Fine-Tuning
cd SLM-Fine-Tuning/Agentic-RAG/Single-Local-Document-RAG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the project root
- Add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

Run the Streamlit application:
```bash
streamlit run main.py
```

### Application Workflow
1. Upload a document
2. Select a reasoning model
3. Enter your query related to the dcoument
5. Optionally evaluate model responses using Facts Evaluation

## Model Options

The application supports multiple DeepSeek models:
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-QWen-7B
- DeepSeek-R1-Distill-QWen-14B

## Advanced Features

- Raw model output debugging
- Detailed response evaluation
- CSV export of results and evaluations

## Performance Notes

- Larger models require more computational resources
- Recommended for systems with dedicated GPUs
- Processing time varies based on document size and model complexity

