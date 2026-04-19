import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# --- إعدادات الأمان واستدعاء المفتاح ---
# ملاحظة: عند رفع الكود لـ Streamlit Cloud، ستقوم بوضع المفتاح في Settings > Secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except:
    openai_api_key = None
    st.warning("⚠️ لم يتم العثور على مفتاح OpenAI. يرجى إضافته في الإعدادات.")

# --- وظائف المعالجة البرمجية ---
def process_laws(folder_path):
    all_docs = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for filename in files:
        path = os.path.join(folder_path, filename)
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                # تخزين النص مع المصدر ورقم الصفحة
                metadata = {"source": filename, "page": i + 1}
                all_docs.append(Document(page_content=content, metadata=metadata))
    return all_docs

# --- واجهة التطبيق ---
st.set_page_config(page_title="مستشار القوانين الجماعية", layout="wide")
st.title("⚖️ منصة القوانين التنظيمية للجماعات والوظيفة العمومية")

# التأكد من وجود المجلد
lib_path = "laws_library"

# زر بناء قاعدة البيانات (يضغط عليه المستخدم مرة واحدة عند إضافة قوانين جديدة)
if st.sidebar.button("📦 تحديث وفهرسة القوانين"):
    with st.spinner("جاري معالجة الملفات..."):
        documents = process_laws(lib_path)
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(documents)
            
            # تحويل النصوص إلى قاعدة بيانات ذكية (فهرسة)
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)
            vector_db.save_local("unified_legal_db")
            st.sidebar.success("تم التحديث بنجاح!")
        else:
            st.sidebar.error("يرجى وضع ملفات PDF داخل مجلد laws_library")

# --- محرك البحث والإجابة ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني:")

if query and openai_api_key:
    if os.path.exists("unified_legal_db"):
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("unified_legal_db", embeddings, allow_dangerous_deserialization=True)
        
        # استرجاع النصوص المتعلقة بالسؤال
        docs = db.similarity_search(query, k=3)
        
        # صياغة الإجابة باستخدام الذكاء الاصطناعي
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
        from langchain.chains.question_answering import load_qa_chain
        chain = load_qa_chain(llm, chain_type="stuff")
        
        response = chain.run(input_documents=docs, question=query)
        
        st.subheader("📝 الإجابة القانونية:")
        st.success(response)
        
        # عرض المصادر للشفافية
        st.markdown("---")
        st.subheader("📚 المصادر المرجعية:")
        for doc in docs:
            st.info(f"الملف: {doc.metadata['source']} | الصفحة: {doc.metadata['page']}")
    else:
        st.warning("يرجى تحديث المكتبة من القائمة الجانبية أولاً.")
