import streamlit as st
import os
from PyPDF2 import PdfReader

# استدعاءات حديثة (Version 2025 Compatible)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# --- إعدادات الأمان ---
# سيقوم البرنامج بسحب المفتاح من Secrets في Streamlit Cloud
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_api_key = None

# --- وظيفة معالجة الملفات ---
def load_legal_docs(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return documents
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for filename in files:
        path = os.path.join(folder_path, filename)
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # إضافة المصدر ورقم الصفحة لكل قطعة نص
                    meta = {"source": filename, "page": i + 1}
                    documents.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            st.error(f"خطأ في قراءة الملف {filename}: {e}")
    return documents

# --- واجهة المستخدم ---
st.set_page_config(page_title="المستشار القانوني للجماعات", layout="wide")
st.title("⚖️ منصة الذكاء الاصطناعي للقوانين التنظيمية")

# التأكد من وجود مجلد القوانين
lib_path = "laws_library"

# لوحة التحكم الجانبية
with st.sidebar:
    st.header("⚙️ الإعدادات")
    if st.button("🚀 تحديث وفهرسة القوانين"):
        with st.spinner("جاري قراءة القوانين وبناء قاعدة البيانات..."):
            raw_docs = load_legal_docs(lib_path)
            if raw_docs:
                # تقسيم النصوص لقطع صغيرة
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(raw_docs)
                
                # إنشاء قاعدة البيانات المتجهة
                if openai_api_key:
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    vector_db = FAISS.from_documents(chunks, embeddings)
                    vector_db.save_local("legal_vector_db")
                    st.success("تم تحديث المكتبة بنجاح!")
                else:
                    st.error("خطأ: مفتاح OpenAI غير مضبوط في Secrets.")
            else:
                st.error("مجلد laws_library فارغ! أضف ملفات PDF أولاً.")

# --- محرك البحث والإجابة ---
query = st.text_input("🔍 اسأل عن أي مقتضى قانوني (مثلاً: تعويضات الرئيس، رخص التعمير، سندات الطلب):")

if query:
    if os.path.exists("legal_vector_db"):
        if openai_api_key:
            # تحميل قاعدة البيانات والبحث
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            db = FAISS.load_local("legal_vector_db", embeddings, allow_dangerous_deserialization=True)
            
            # البحث عن القطع الأكثر صلة
            related_docs = db.similarity_search(query, k=3)
            
            # توليد الإجابة
            llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=related_docs, question=query)
            
            # عرض النتيجة
            st.markdown("### 📝 الإجابة القانونية:")
            st.info(answer)
            
            # عرض المصادر
            st.markdown("---")
            st.subheader("📚 المراجع المعتمدة:")
            for doc in related_docs:
                st.write(f"🔹 **الملف:** {doc.metadata['source']} | **الصفحة:** {doc.metadata['page']}")
        else:
            st.error("يرجى إعداد مفتاح API Key أولاً.")
    else:
        st.warning("الرجاء الضغط على زر 'تحديث وفهرسة القوانين' من القائمة الجانبية لبدء العمل.")
