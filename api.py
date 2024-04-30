from flask import Flask, render_template, request, jsonify
import pdfplumber
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain import OpenAI
import os
from flask_cors import CORS

app = Flask(__name__)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'pdf_file' not in request.files:
        return "No file part"

    pdf_file = request.files['pdf_file']
    
    if pdf_file.filename == '':
        return "No selected file"
    
    os.remove('C:/Users/queen/Desktop/Hackathon/TSEC/flask_api/data/output_text.txt')
    pdf_path = 'C:/Users/queen/Desktop/Hackathon/TSEC/flask_api/Dataset (1)/' + pdf_file.filename
    output_text_file_path = 'C:/Users/queen/Desktop/Hackathon/TSEC/flask_api/data/output_text.txt'


    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        text = page.extract_text(x_tolerance=2)

    with open(output_text_file_path, 'w') as file:
        file.write(text)

    return f"Text has been saved to: {output_text_file_path}"

os.environ["OPENAI_API_KEY"] = "API KEY"

CORS(app)

def create_vector_index(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 0.2
    
    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
    
    # Define LLM
    # Missing comment symbol here
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
    
    # Load data
    docs = SimpleDirectoryReader(path).load_data()
    
    # Create vector index
    vector_index = GPTVectorStoreIndex.from_documents(docs, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    vector_index.storage_context.persist(persist_dir='vectorIndex.json')
    return vector_index

vector_index = create_vector_index('data')

class NodeWithScore:
    def _init_(self, id, score):
        self.id = id
        self.score = score
    
    def get_info(self):
        return {'id': self.id, 'score': self.score}

def answer_me(prompt):
    storage_context = StorageContext.from_defaults(persist_dir='vectorIndex.json')
    v_index = load_index_from_storage(storage_context)
    query_engine = v_index.as_query_engine(similarity_top_k=2)
    response = query_engine.query(prompt)
    
    # Extracting relevant information from NodeWithScore
    serialized_response = response
    return serialized_response

@app.route('/ask', methods=['GET'])
def ask_question():
    dict = {}
    prompt = request.args.get('question', default='', type=str)
    if not prompt:
        return jsonify({'error': 'No question provided'}), 400

    response = answer_me(prompt)
    temp=response.response
    print(temp)
    dict['answer']=temp
    return dict


if __name__ == '__main__':
    app.run(debug=True, port=5000)

