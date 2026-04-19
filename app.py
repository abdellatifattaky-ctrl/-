import streamlit as st
import os
from PyPDF2 import PdfReader

# استدعاءات حديثة ومباشرة (تجنب المسارات القديمة المسببة للأخطاء)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# --- إعداد المفتاح ---
api_key = st.secrets.get("GOOGLE_API_KEY")

st.set_page_config(page_title="المستشار القانوني", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين الجماعية")

# دالة معالجة النصوص
def get_pdf_text(pdf_docs):
    temp_docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                temp_docs.append(Document(page_content=text, metadata={"source": pdf.name, "page": i+1}))
    return temp_docs

# الواجهة الجانبية
with st.sidebar:
    st.header("إدارة البيانات 📂")
    uploaded_files = st.file_uploader("ارفع ملفات القانون (PDF)", accept_multiple_files=True)
    
    if st.button("بدء الفهرسة الذكية ✨"):
        if uploaded_files and api_key:
            with st.spinner("جاري معالجة الملفات..."):
                raw_docs = get_pdf_text(uploaded_files)
                # استخدام المكتبة الجديدة المستقلة للتقسيم
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                final_chunks = text_splitter.split_documents(raw_docs)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                vector_db = FAISS.from_documents(final_chunks, embeddings)
                vector_db.save_local("faiss_index_bin")
                st.sidebar.success("✅ تم بنجاح!")
        else:
            st.sidebar.error("تأكد من الملفات والمفتاح")

# البحث
query = st.text_input("اسأل سؤالك القانوني:")
if query and api_key and os.path.exists("faiss_index_bin"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = FAISS.load_local("faiss_index_bin", embeddings, allow_dangerous_deserialization=True)
    
    docs = db.similarity_search(query, k=4)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    response = chain.run(input_documents=docs, question=query)
    st.markdown("### 📝 الإجابة:")
    st.info(response)
