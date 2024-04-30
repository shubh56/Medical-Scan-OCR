from llama_index import SimpleDirectoryReader,GPTListIndex,GPTVectorStoreIndex,LLMPredictor,PromptHelper,ServiceContext,StorageContext,load_index_from_storage
from langchain import OpenAI
import sys
import os

os.environ["OPENAI_API_KEY"] = "APIKEY"



def createVectorIndex(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 0.2
    
    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
    
    #define LLM
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
    
    #load data
    docs = SimpleDirectoryReader(path).load_data()
    
    #create vector index
    vectorIndex = GPTVectorStoreIndex.from_documents(docs, llm_predictor=llmPredictor, prompt_helper=prompt_helper)
    vectorIndex.storage_context.persist(persist_dir='vectorIndex.json')
    return vectorIndex


vectorIndex = createVectorIndex("Knowledge")

from llama_index import load_index_from_storage, StorageContext

def answerMe(vectorIndex):
    storage_context = StorageContext.from_defaults(persist_dir=vectorIndex)
    vIndex = load_index_from_storage(storage_context)
    query_engine = vIndex.as_query_engine(similarity_top_k=2)
    while True:
        prompt = input('Please ask: ')
        response = query_engine.query(prompt)
        print(f"Response: {response} \n")
        

answerMe('vectorIndex.json')