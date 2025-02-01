from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
