import streamlit as st
import openai
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
import base64
import io


# OpenAI API Key (Set securely in environment variables or Streamlit secrets)
api_key_v = "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"
client = openai.OpenAI(api_key=api_key_v)  # Create a client instance

# Function to encode an image to base64 format
def encode_image(uploaded_file):
    # Open the image with PIL and convert to a supported format
    image = Image.open(uploaded_file).convert("RGB")  
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # Convert to JPEG
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
# Function to get response from GPT-4o
def ask_gpt(uploaded_file):
    # Medical document analysis system prompt in Arabic
    base64_image = encode_image(uploaded_file)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical expert. Summarize and explain medical reports in simple terms in arabic."},
            {"role": "user", "content": [
                {"type": "text", "text": "Here is a medical report. Please summarize it and explain it in simple terms for a patient in arabic language."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        max_tokens=500  # Limit response length
    )
    
    return response.choices[0].message.content.strip()


# Streamlit UI Setup
st.set_page_config(page_title="المساعد الطبي الذكي", layout="wide")
st.title("🩺 المساعد الطبي الذكي")
st.write("مرحبًا بك! يمكنني مساعدتك في الإجابة على الأسئلة الطبية أو تحليل الوصفات الطبية والأشعة.")
st.markdown("""
    <style>
    body, .stApp {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    .stChatMessage .css-16idsys {
        justify-content: flex-end !important;
    }
    </style>
    """, unsafe_allow_html=True)
# Sidebar navigation
option = st.sidebar.selectbox("اختر الخدمة", ["الأسئلة الشائعة", "تحميل صورة للتحليل"])

faqs = {
    "كيف أقدر أحجز موعد؟": "تقدر تحجز موعد عن طريق الموقع، التطبيق، أو تتواصل مع خدمة العملاء مباشرة.",
    "أقدر أغير أو ألغي موعدي؟": "أكيد! تواصل معنا قبل موعدك بوقت كاف، ورح نساعدك بتغيير أو إلغاء الموعد.",
    "لازم يكون عندي تحويل طبي للحجز؟": "حسب نوع الفحص. بعض الفحوصات تحتاج تحويل من الطبيب، لكن تقدر تتواصل معنا للتأكد.",
    "ما هي الخدمات اللي تقدمونها؟": "حياك الله، نوفر خدمات الأشعة المتقدمة، التحاليل الطبية، وتقارير الأشعة في نفس اليوم إذا كانت الفحوصات قبل الساعة 5 مساءً.",
    "هل عندكم فحوصات للأورام؟": "نعم، عندنا أجهزة حديثة لفحوصات الأورام وتشخيصها بدقة، بالإضافة لتقارير تفصيلية تساعد الطبيب المعالج.",
    "ما هي الأجهزة الطبية اللي عندكم؟": "المركز مجهز بأحدث أجهزة التصوير الطبي، مثل التصوير بالرنين المغناطيسي (MRI) والتصوير المقطعي (CT) وأجهزة الماموجرام لفحص الثدي.",
    "متى أقدر أستلم نتائج الفحوصات؟": "إذا كان الفحص قبل 5 مساءً، تقدر تستلم التقرير بنفس اليوم. غير كذا، يكون جاهز في يوم العمل التالي.",
    "كيف أستلم النتائج؟": "تقدر تستلمها من المركز أو عن طريق الموقع الإلكتروني إذا كنت مسجّل في البوابة الإلكترونية.",
    "الطبيب يحتاج يشوف نتائجي، كيف أوصلها له؟": "الطبيب يقدر يشوف النتائج مباشرة من خلال نظامنا الإلكتروني إذا كان مخوّل للدخول.",
    "متى أوقات دوامكم؟": "المركز يفتح من الساعة 8 صباحًا إلى 10 مساءً، من السبت للخميس.",
    "عندكم دوام في الإجازات؟": "نعم، عندنا دوام في الإجازات الرسمية للحالات الضرورية.",
    "كيف أتواصل معكم؟": "تقدر تتواصل معنا عن طريق الرقم الموحد أو عبر واتساب لخدمات أسرع.",
    "وين موقعكم بالضبط؟": "حياك الله، موقعنا في مدينة الرياض، ونغطي جميع الأحياء بالخدمات. تقدر تحصل الموقع الدقيق في قسم 'اتصل بنا'.",
    "عندكم فروع ثانية؟": "حاليًا عندنا مركز واحد بالرياض، لكن خططنا تشمل التوسع مستقبلاً.",
    "وش طرق الدفع اللي عندكم؟": "نوفر الدفع النقدي، البطاقات البنكية، والتحويلات الإلكترونية.",
    "تقبلون التأمين الطبي؟": "نعم، نقبل تأمين عدد من الشركات المعتمدة. تقدر تتواصل معنا لمعرفة التفاصيل.",
    "أقدر أدفع عن طريق الموقع؟": "حاليًا الدفع يتم في المركز، لكن نعمل على إضافة الدفع الإلكتروني قريبًا.",
    "عندكم تطبيق للهاتف؟": "تقدر تحمل التطبيق من متجر Google Play أو App Store لحجز الموعد والاطلاع على النتائج.",
    "وش أقدر أسوي في التطبيق؟": "داخل التطبيق تقدر تحجز موعد، تستعرض نتائج الفحوصات، وتتواصل مع الفريق الطبي بسهولة."
}
def respond_to_faq(query):
    # Convert FAQ dictionary to a formatted string
    faq_text = "\n".join([f"- {q}: {a}" for q, a in faqs.items()])

    FAQ_SYSTEM_PROMPT = f"""
    أنت مساعد ذكي مسؤول عن الإجابة على الأسئلة بناءً على قائمة الأسئلة الشائعة (FAQ) أدناه. 
    إذا طُرح عليك سؤال موجود في القائمة او سؤال له نفس المعني او مشابه له ف المعني ، استخدم الجواب المحدد له تمامًا. 
    أما إذا كان السؤال غير موجود، فقل فقط: "عذرًا، لا أملك إجابة لهذا السؤال."
    ### قائمة الأسئلة الشائعة:
    {faq_text}
    """
   
    try:
        messages = [
            {"role": "system", "content": FAQ_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,  # Lower temperature for more deterministic responses
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"عذرًا، حدث خطأ أثناء معالجة استعلامك: {str(e)}"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"]):
        st.markdown(f"{icon} **{message['content']}**")
if option == "الأسئلة الشائعة":
    #st.subheader("📌 الدردشة الذكية - الأسئلة الشائعة")
    # User Input
    
    user_question = st.chat_input("اسألني عن الأسئلة الشائعة...")

    if user_question is not None  and user_question != "":
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        # Get response
        with st.spinner("جارٍ الرد..."):
            response = respond_to_faq(user_question)

        # Display assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Image Upload and Analysis
elif option == "تحميل صورة للتحليل":
    st.subheader("📤 تحميل صورة طبية أو وصفة طبية")
    uploaded_file = st.file_uploader("ارفع صورة الأشعة أو الوصفة الطبية", type=["png", "jpg", "jpeg","webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 الصورة التي تم تحميلها", width=100)
        
        if st.button("🔍 تحليل الصورة"):
            with st.chat_message("user"):
                st.markdown("🖼️ صورة تم رفعها للتحليل")
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": "🖼️ صورة تم رفعها للتحليل"})

            with st.spinner("جارٍ التحليل..."):
                
                summary = ask_gpt(uploaded_file)
            with st.chat_message("assistant"):
                st.markdown(summary)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": summary})
           

 
