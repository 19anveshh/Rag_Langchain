from langchain_core.documents import Document
from typing import List
from pypdf import PdfReader
from pathlib import Path 



def read_pdf_text(pdf_path:str | Path) ->str:
    """ Read and extract the content form pdf"""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError('FILE NOT FOUND')
    reader = PdfReader(str(path))
    text_parts= [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)

def read_multiple_pdfs(pdf_path:list[str | Path])-> list[Document]:
    """read the text from the multiple pdf and make a langchaoin document """
    docs : list[Document]=[]
    for singlepath in pdf_path:
        path_obj = Path(singlepath)
        text =read_pdf_text(path_obj)
        docs.append(Document(page_content=text, metadata={"source": str(path_obj)}))
    return docs

