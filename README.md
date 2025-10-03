# **📄 RAG-Powered Single Document Q\&A (Streamlit App)**

## **Overview**

This Streamlit application allows users to upload local documents (PDF, Word, Text, and Markdown) and ask specific questions about their contents. Utilizing state-of-the-art **DeepSeek R1 Reasoning Language Models**, this application provides intelligent, context-aware responses by analyzing the uploaded document. It delivers powerful answers complete with grounding and evaluation metrics.

## **🚀 Key Features**

* **Multi-Document Support**: Upload multiple file types (PDF, Word, Text, and Markdown files) with automatic content extraction and analysis.  
* **Advanced AI Reasoning**: Integrates multiple **DeepSeek R1** language models for intelligent query processing and context-aware natural language Q\&A over document contents.  
* **Response Evaluation**: Provides built-in metrics and tools for response analysis:  
  * Grounding score measurement  
  * Quality score verification  
  * Detailed response analysis

## **⚡ Prerequisites**

| Requirement | Detail |
| :---- | :---- |
| **Python** | Python 3.8+ |
| **GPU** | CUDA-compatible GPU (Recommended for faster inference) |
| **RAM** | Minimum 16GB RAM (Suggested for operating large models) |

## **⚙️ Installation**

Follow these steps to set up the application environment:

### **1\. Clone the Repository**

git clone \[https://github.com/muhammadharoon9319-png/RAG-Powered-Single-Document-Q-A.git] 

### **2\. Create a Virtual Environment**

python \-m venv venv  
source venv/bin/activate   \# On Windows, use \`venv\\Scripts\\activate\`

### **3\. Install Dependencies**

pip install \-r requirements.txt

### **4\. Set up Environment Variables**

Create a file named .env in the project root and add your Google API key:

GOOGLE\_API\_KEY=your\_google\_api\_key\_here

## **🚀 Usage**

Run the Streamlit application using the following command:

streamlit run main.py

### **Application Workflow (🛠)**

1. Upload one or more documents.  
2. Select a reasoning model from the available options.  
3. Enter your query related to the document(s).  
4. View AI-generated, context-aware answers.  
5. Optionally evaluate model responses using **Facts Evaluation**.

## **🤖 Supported Models**

The application supports multiple DeepSeek R1 Distill models:

* DeepSeek-R1-Distill-Llama-8B  
* DeepSeek-R1-Distill-QWen-7B  
* DeepSeek-R1-Distill-QWen-14B

## **🔬 Advanced Features**

* Raw model output debugging  
* Detailed response evaluation  
* Export results & evaluations as CSV

## **⚡ Performance Notes**

* Larger models require more computational resources (CPU/GPU).  
* The best performance is achieved on systems with dedicated GPUs.  
* Processing time varies based on the size of the uploaded document and the complexity of the selected model.
