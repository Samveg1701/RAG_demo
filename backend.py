import torch
import accelerate
import torch 
from flask import Flask, jsonify, request
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders.text import TextLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForQuestionAnswering, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
# from langchain.chains import vectorstores

# Initialize Flask app
app = Flask(__name__)

# Load PDF and text data
with open("data/ICMR_GuidelinesType2diabetes2018_0.pdf", "rb") as pdf_file:
    loader = PyPDFLoader("data/ICMR_GuidelinesType2diabetes2018_0.pdf")
    #data = TextLoader('data/data_description.txt').load()
    print("Text:")
    pages=loader.load()
    with open("pdf_text.txt", 'a') as f:
        for page in pages: 
            f.write(page.page_content)



# Initialize RecursiveCharacterTextSplitter
data = TextLoader('data/data_description.txt').load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# Initialize HuggingFaceEmbeddings and FAISS
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
db = FAISS.from_documents(docs, embeddings)

# Initialize model and tokenizer
HOST_IP = '127.0.0.1'
PORT = 5000

# Initialize model and tokenizer
nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("Mistral-7B-Instruct-v0.1", device_map='cuda', quantization_config=nf4_config)
tokenizer = AutoTokenizer.from_pretrained("Mistral-7B-Instruct-v0.1")

# Define question-answering pipeline
question_answerer = pipeline(return_full_text=True, task='text-generation',
                             model=model, tokenizer=tokenizer, max_new_tokens=512)

# Define endpoint for asking questions
@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        question = request.json.get('question')
        try:
            # Generate answer
            messages = [{"role": "user", "content": question}]
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            device = "cuda" if torch.cuda.is_available() else 'cpu'
            model_inputs = encodeds.to(device)
            generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True)
            result = tokenizer.batch_decode(generated_ids)
            return jsonify({"answer": result[0]})
        except Exception as e:
            return jsonify({"error": str(e)})


@app.route('/test', methods=['GET'])
def test():
    return jsonify({"testing":"test"})

if __name__ == '__main__':
    # Run the app on an internal IP and specified port
    # app.run(host=HOST_IP, port=PORT, debug=True)
    app.run(debug=False, host='0.0.0.0', port=5000)
    # http://192.168.1.4:5000/ask