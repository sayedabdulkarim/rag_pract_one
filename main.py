from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
################# Document Loading ##################
data=None
local_path = "scammer-agent.pdf"

print('start ======')
#
# # Local PDF file uploads
if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    print("Upload a PDF file")

print(data, 'data ======')


################## Chunking ##################
# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)

# Split the documents
if data is None:
    print("No data loaded from PDF")
    exit(1)
else:
    chunks = text_splitter.split_documents(data)

# Print the chunks

print(chunks, ' chunks ======')

############ Embeddings and Vector Database ############

# """
# Embedding - for text to number conversion, makes text searchable
# Vector Database - for storing embeddings and searching embeddings , enables similarity search
# Embeddings and Vector Database
# """

# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag"
)

print(vector_db, 'vector_db ======')


########### Setup Language local_model ############
local_model = "llama3.2:1b"  # Replace with your model name
llm = ChatOllama(model=local_model)

########### Create Query Prompt ############
#   Generated multi Variations of the user question to retrieve relevant documents from a vector database
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
)