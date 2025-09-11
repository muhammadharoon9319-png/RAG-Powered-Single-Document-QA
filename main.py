import streamlit as st
import torch
import tempfile
import os
from dotenv import load_dotenv
import pandas as pd
import re
import json
import logging
import mammoth
import gc
import docx  

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class DocumentQueryPipeline:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):

        self.model_name = model_name
        self.context_document = ""
        self.model = None
        self.tokenizer = None
        self._initialize_models()
    
    def _clear_gpu_memory(self):
        """Explicitly clear GPU memory and unload models"""
        if self.model is not None:
            del self.model
            del self.tokenizer
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force Python garbage collection
        gc.collect()
    
    def _initialize_models(self):
        """Initialize models with robust memory management"""
        self._clear_gpu_memory()
        
        # Configure Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load selected DeepSeek model
        config = AutoConfig.from_pretrained(self.model_name)
        self.eos_id = config.eos_token_id if config.eos_token_id is not None else 0
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with low memory settings when necessary
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except Exception as e:
            st.warning(f"Loading model with reduced precision due to memory constraints: {str(e)}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            st.warning("Running on CPU. This will be slow for large models.")
    
    def _extract_text_from_file(self, uploaded_file):
        """Extract text from various file types using Markitdown for PDFs"""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        try:
            # PDF processing with Markitdown
            if file_extension == '.pdf':
                from markitdown import MarkItDown
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    md_converter = MarkItDown()  
                    result = md_converter.convert(temp_file_path)
                    md_text = result.text_content
                    
                    md_text = "\n".join(
                        line for line in md_text.splitlines()
                        if line.strip() and not line.strip().startswith("----")
                    )
                    
                    return md_text
                finally:
                    os.unlink(temp_file_path)
            
            elif file_extension in ['.docx', '.doc']:
                doc = docx.Document(uploaded_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            
            elif file_extension in ['.txt', '.md']:
                return uploaded_file.getvalue().decode('utf-8')
            
            else:
                result = mammoth.convert_to_html(uploaded_file)
                text = re.sub('<[^<]+?>', '', result.value)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return ""

    def load_document_context(self, uploaded_file):
        """Convert uploaded file to text and store as context"""
        text = self._extract_text_from_file(uploaded_file)
        
        # Truncate extremely long documents to manage memory
        max_document_length = 50000  
        if len(text) > max_document_length:
            st.warning(f"Document is very long. Truncating to first {max_document_length} characters.")
            text = text[:max_document_length]
        
        self.context_document = text
        return text

    def extract_clean_answer(self, text):
        """Extract and clean the model's response"""
        # Remove the full prompt from the text
        prompt_start = text.find("Context Document:")
        if prompt_start != -1:
            text = text[prompt_start:]
        
        # Find the position of "</think>" tag
        think_end_index = text.find("</think>")
        
        if think_end_index != -1:
            # Extract text after "</think>" tag
            text = text[think_end_index + len("</think>"):].strip()
        
        # Remove any remaining markdown or special tokens
        text = re.sub(r'^\s*[#*]+\s*', '', text, flags=re.MULTILINE)
        return text.strip()    


    def process_query(self, query, progress_bar=None, status_text=None):
        """Process a single query through the pipeline"""
        if not self.context_document:
            st.error("No PDF context loaded. Please upload a PDF first.")
            return None
        
        if status_text:
            status_text.text("Generating answer...")
        
        prompt = f"""
        Context Document: {self.context_document}
        
        User Query: {query}
        
        Based strictly on the context document, provide a detailed answer to the query.
        """
    
        # Use the tokenizer to prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=2048, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        raw_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        model_response = self.extract_clean_answer(raw_response)
        
        if progress_bar:
            progress_bar.progress(1.0)
            
        if status_text:
            status_text.text("Done!")
        
        return {
            "query": query, 
            "context": self.context_document, 
            "response": model_response,
            "raw_response": raw_response
        }


class FactsEvaluator:
    def __init__(self):
        # Load evaluation prompts
        self.json_prompt = """You are a helpful and harmless AI assistant. You will be provided with a textual context and a model-generated response.\nYour task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context.\n\n**Instructions:**\n\n1. **Decompose the response into individual sentences.**\n2. **For each sentence, assign one of the following labels:**\n    * **`supported`**: The sentence is entailed by the given context.  Provide a supporting excerpt from the context. The supporting except must *fully* entail the sentence. If you need to cite multiple supporting excepts, simply concatenate them.\n    * **`unsupported`**: The sentence is not entailed by the given context. No excerpt is needed for this label.\n    * **`contradictory`**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.\n    * **`no_rad`**: The sentence does not require factual attribution (e.g., opinions, greetings, questions, disclaimers).  No excerpt is needed for this label.\n3. **For each label, provide a short rationale explaining your decision.**  The rationale should be separate from the excerpt.\n4. **Be very strict with your `supported` and `contradictory` decisions.** Unless you can find straightforward, indisputable evidence excerpts *in the context* that a sentence is `supported` or `contradictory`, consider it `unsupported`. You should not employ world knowledge unless it is truly trivial.\n\n**Input Format:**\n\nThe input will consist of two parts, clearly separated:\n\n* **Context:**  The textual context used to generate the response.\n* **Response:** The model-generated response to be analyzed.\n\n**Output Format:**\n\nFor each sentence in the response, output a JSON object with the following fields:\n\n* `"sentence"`: The sentence being analyzed.\n* `"label"`: One of `supported`, `unsupported`, `contradictory`, or `no_rad`.\n* `"rationale"`: A brief explanation for the assigned label.\n* `"excerpt"`:  A relevant excerpt from the context. Only required for `supported` and `contradictory` labels.\n\nOutput each JSON object on a new line.\n\n**Example:**\n\n**Input:**\n\n```\nContext: Apples are red fruits. Bananas are yellow fruits.\n\nResponse: Apples are red. Bananas are green. Bananas are cheaper than apples. Enjoy your fruit!\n```\n\n**Output:**\n\n{"sentence": "Apples are red.", "label": "supported", "rationale": "The context explicitly states that apples are red.", "excerpt": "Apples are red fruits."}\n{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}\n{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "The context does not mention the price of bananas or apples.", "excerpt": null}\n{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general expression and does not require factual attribution.", "excerpt": null}\n\n**Now, please analyze the following context and response:**\n\n**User Query:**\n{{user_request}}\n\n**Context:**\n{{context_document}}\n\n**Response:**\n{{response}}"""

        
        self.quality_prompt = """Your mission is to judge the response from an AI model, the *test* response, calibrating your judgement using a *baseline* response.\nPlease use the following rubric criteria to judge the responses:\n\n<START OF RUBRICS>\nYour task is to analyze the test response based on the criterion of "Instruction Following". Start your analysis with "Analysis".\n\n**Instruction Following**\nPlease first list the instructions in the user query.\nIn general, an instruction is VERY important if it is specifically asked for in the prompt and deviates from the norm. Please highlight such specific keywords.\nYou should also derive the task type from the user query and include the task-specific implied instructions.\nSometimes, no instruction is available in the user query.\nIt is your job to infer if the instruction is to autocomplete the user query or is asking the LLM for follow-ups.\nAfter listing the instructions, you should rank them in order of importance.\nAfter that, INDEPENDENTLY check if the test response and the baseline response meet each of the instructions.\nYou should itemize, for each instruction, whether the response meets, partially meets, or does not meet the requirement, using reasoning.\nYou should start reasoning first before reaching a conclusion about whether the response satisfies the requirement.\nCiting examples while reasoning is preferred.\n\nReflect on your answer and consider the possibility that you are wrong.\nIf you are wrong, explain clearly what needs to be clarified, improved, or changed in the rubric criteria and guidelines.\n\nIn the end, express your final verdict as one of the following three json objects:\n\n```json\n{{\n  "Instruction Following": "No Issues"\n}}\n```\n\n```json\n{{\n  "Instruction Following": "Minor Issue(s)"\n}}\n```\n\n```json\n{{\n  "Instruction Following": "Major Issue(s)"\n}}\n```\n\n<END OF RUBRICS>\n\n# Your task\n## User query\n<|begin_of_query|>\n{{full_prompt}}\n<|end_of_query|>\n\n## Test Response:\n<|begin_of_test_response|>\n{{response_a}}\n<|end_of_test_response|>\n\n## Baseline Response:\n<|begin_of_baseline_response|>\n{{response_b}}\n<|end_of_baseline_response|>\n\nPlease write your analysis and final verdict for the test response."""
        
        # Configure Gemini API
        google_api_key = "gemini_api_key_here"
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=google_api_key)

    def generate_gemini(self, prompt):
        """Generate a response using Gemini model"""
        response = genai.GenerativeModel('gemini-1.5-flash-002').generate_content(prompt)
        return response.text

    def parse_structured_json(self, ans):
        """Parse JSON output from grounding evaluation"""
        if '```json' in ans:
            ans = ans.split('```json')[1].split('```')[0]
        ans = ans.strip()
        ans = ans.replace('}\n', '}\n@\n@\n')
        parsed_answers = []
        for line in ans.split('\n@\n@\n'):
            try:
                line = line.replace('\n', ' ')
                line = line.replace("\\'", "'")
                parsed = json.loads(line)
                parsed_answers.append(parsed)
            except:
                pass
        if len(parsed_answers) > 0:
            bool_ans = all(d['label'] == 'supported' or d['label'] == 'no_rad' for d in parsed_answers)
        else:
            bool_ans = False
        return bool_ans, parsed_answers

    def parse_json(self, ans):
        """Parse JSON output from quality evaluation"""
        parsed = {}
        if '```json' in ans:
            ans = ans.split('```json')[1]
            ans = ans.split('```')[0]
        ans = ans.replace('\n', ' ')
        try:
            parsed = json.loads(ans)
        except Exception as e:
            pass
        if 'Instruction Following' not in parsed:
            parsed['Instruction Following'] = 'Invalid'
        elif parsed['Instruction Following'] not in ['No Issues', 'Minor Issue(s)', 'Major Issue(s)', 'Invalid']:
            parsed['Instruction Following'] = 'Invalid'
        return parsed

    def evaluate_grounding(self, user_request, context_document, response):
        """Evaluate if response is grounded in the context"""
        prompt = self.json_prompt.replace('{{user_request}}', user_request).replace('{{context_document}}', context_document).replace('{{response}}', response)
        evaluation_text = self.generate_gemini(prompt)
        evaluation, parsed = self.parse_structured_json(evaluation_text)
        return evaluation, parsed

    def evaluate_quality(self, user_request, response_a, response_b):
        """Evaluate response quality against a reference"""
        prompt = self.quality_prompt.replace('{{user_request}}', user_request).replace('{{response_a}}', response_a).replace('{{response_b}}', response_b)
        evaluation_text = self.generate_gemini(prompt)
        parsed = self.parse_json(evaluation_text)
        return "No Issues" in parsed['Instruction Following'], parsed

    def evaluate_result(self, result):
        """Evaluate a single result for both grounding and quality"""
        # Generate a reference response using Gemini for quality evaluation
        reference_prompt = f"""
        You are a medical expert. Answer the following query based on your knowledge:
        
        Query: {result['query']}
        
        Provide a detailed, well-structured answer.
        """
        reference_response = self.generate_gemini(reference_prompt)
        
        # Evaluate grounding
        grounding_result, grounding_details = self.evaluate_grounding(
            user_request=result['query'],
            context_document=result['context'],
            response=result['response']
        )
        
        # Evaluate quality
        quality_result, quality_details = self.evaluate_quality(
            user_request=result['query'],
            response_a=result['response'],
            response_b=reference_response
        )
        
        # Calculate combined score
        combined_result = grounding_result and quality_result
        
        return {
            "grounding_evaluation": grounding_result,
            "quality_evaluation": quality_result,
            "combined_evaluation": combined_result,
            "grounding_details": grounding_details,
            "quality_details": quality_details
        }

def main():
    st.set_page_config(
        page_title="Document Query Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Document Query Assistant")
    st.markdown("""
    Upload a PDF, Word document, or text file and ask queries about its contents. 
    The system uses a DeepSeek model to analyze and answer questions.
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    
    model_options = {
        "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-QWen-7B": "deepseek-ai/DeepSeek-R1-Distill-QWen-7B",
        "DeepSeek-R1-Distill-QWen-14B": "deepseek-ai/DeepSeek-R1-Distill-QWen-14B"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        show_raw_output = st.checkbox("Show Raw Model Output (for debugging)", value=False)
    
    # Initialize session state for storing results and evaluation
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    
    # Document Upload Section
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["pdf", "docx", "txt", "doc", "md", "odt"],
        help="Supported file types: PDF, Word, Text, Markdown"
    )

    # Initialize or reinitialize pipeline if model changes
    if ('pipeline' not in st.session_state or 
        st.session_state.model_name != model_options[selected_model]):
        # If a pipeline exists, clear its GPU resources first
        if 'pipeline' in st.session_state:
            try:
                st.session_state.pipeline._clear_gpu_memory()
            except Exception as e:
                st.warning(f"Error clearing previous model resources: {e}")
        
        st.session_state.model_name = model_options[selected_model]
        with st.spinner(f"Loading {selected_model}... This may take a moment."):
            st.session_state.pipeline = DocumentQueryPipeline(model_name=model_options[selected_model])
        st.success(f"Model loaded: {selected_model}")
        
    # If PDF is uploaded, process it
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            context_text = st.session_state.pipeline.load_document_context(uploaded_file)
        
        st.success("PDF Loaded Successfully!")
        
        with st.expander("PDF Content Preview"):
            st.text(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)
    
    # Query input
    query = st.text_area("Enter your query about the PDF:", height=100)
    
    # Process the query when the user clicks the button
    if st.button("Submit Query", type="primary") and query and uploaded_file is not None:
        st.markdown("---")
        
        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process the query
        try:
            result = st.session_state.pipeline.process_query(
                query, 
                progress_bar=progress_bar, 
                status_text=status_text
            )
            
            st.session_state.current_result = result         
            st.session_state.evaluation_results = None
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
        
        finally:
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

    if st.session_state.current_result is not None:
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("### Document Context")
        with st.expander("View Document Content", expanded=False):
            st.markdown(result["context"])
        
        st.markdown("### Answer")
        st.markdown(result["response"])
        
        if show_raw_output:
            with st.expander("Raw Model Output (Debug)", expanded=False):
                st.text(result["raw_response"])
        
        result_df = pd.DataFrame([{
            "Query": result["query"],
            "Document Context": result["context"],
            "Response": result["response"]
        }])
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="pdf_query_results.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        st.markdown("### Facts Evaluation")
        st.markdown("Evaluate how well the response is grounded in the uploaded document.")
        
        if st.button("Evaluate Response", type="secondary"):
            with st.spinner("Evaluating response... This may take a moment."):
                try:
                    evaluator = FactsEvaluator()
                    
                    evaluation_results = evaluator.evaluate_result(st.session_state.current_result)
                    
                    st.session_state.evaluation_results = evaluation_results
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                    st.exception(e)
        
        if st.session_state.evaluation_results is not None:
            eval_results = st.session_state.evaluation_results
            
            # Create columns for scores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Grounding Score", 
                    value="True" if eval_results["grounding_evaluation"] else "False"
                )
            
            with col2:
                st.metric(
                    label="Quality Score", 
                    value="True" if eval_results["quality_evaluation"] else "False"
                )
            
            with col3:
                st.metric(
                    label="Combined Score", 
                    value="True" if eval_results["combined_evaluation"] else "False"
                )
            
            eval_summary_df = pd.DataFrame([{
                "Query": st.session_state.current_result["query"],
                "Grounding Score": "True" if eval_results["grounding_evaluation"] else "False",
                "Quality Score": "True" if eval_results["quality_evaluation"] else "False",
                "Combined Score": "True" if eval_results["combined_evaluation"] else "False"
            }])
            
            csv = eval_summary_df.to_csv(index=False)
            st.download_button(
                label="Download Evaluation Results as CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
