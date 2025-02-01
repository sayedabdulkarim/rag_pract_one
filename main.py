from langchain_community.document_loaders import UnstructuredPDFLoader

## Document Loading
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

print(data, 'end ======')