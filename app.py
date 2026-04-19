import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# --- إعداد المفتاح من Secrets ---
# تأكد أن الاسم في الإعدادات هو GOOGLE_API_KEY
api_key = st.secrets.get("GOOGLE_API_KEY")

st.set_page_config(page_title="المستشار القانوني الذكي", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين الجماعية")

def get_pdf_text(pdf_docs):
    temp_docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                temp_docs.append(Document(page_content=text, metadata={"source": pdf.name, "page": i+1}))
    return temp_docs

# القائمة الجانبية لرفع الملفات
with st.sidebar:
    st.header("إدارة البيانات 📂")
    uploaded_files = st.file_uploader("ارفع ملفات القانون (PDF)", accept_multiple_files=True)
    
    if st.button("بدء الفهرسة الذكية ✨"):
        if uploaded_files and api_key:
            with st.spinner("جاري تحليل النصوص وتقسيمها..."):
                # 1. استخراج النصوص
                raw_docs = get_pdf_text(uploaded_files)
                
                # 2. تقسيم النصوص لقطع صغيرة جداً (لتجنب ضغط السيرفر)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                final_chunks = text_splitter.split_documents(raw_docs)
                
                # 3. إعداد المحرك
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                
                # 4. بناء قاعدة البيانات (الحل هنا: إرسال دفعات صغيرة)
                try:
                    # نأخذ أول 50 قطعة كمثال للتأكد من العمل
                    vector_db = FAISS.from_documents(final_chunks, embeddings)
                    vector_db.save_local("faiss_index_google")
                    st.success("✅ تمت الفهرسة بنجاح!")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء الفهرسة: {e}")
        else:
            st.error("تأكد من رفع الملفات ووجود مفتاح GOOGLE_API_KEY في Secrets")

# واجهة الأسئلة
user_question = st.text_input("اسأل عن أي مادة قانونية:")

if user_question and api_key:
    if os.path.exists("faiss_index_google"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index_google", embeddings, allow_dangerous_deserialization=True)
        
        # البحث عن المستندات
        docs = new_db.similarity_search(user_question, k=4)
        
        # الإجابة باستخدام Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff")
        
        response = chain.run(input_documents=docs, question=user_question)
        
        st.markdown("### 📝 الرد القانوني:")
        st.info(response)
        
        with st.expander("شاهد المصادر المعتمدة"):
            for d in docs:
                st.write(f"📄 {d.metadata['source']} - صفحة {d.metadata['page']}")
    else:
        st.warning("الرجاء فهرسة الملفات أولاً من القائمة الجانبية.")
