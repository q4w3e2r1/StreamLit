import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
import time

class TextClassifier:
    def __init__(self, model_path):
        HF_MODEL_NAME = 'SeKooooo/project'
        self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        self.labels = ['Бывший СССР','Интернет и СМИ','Культура','Мир',
                      'Наука и техника','Россия', 'Спорт', 'Экономика']

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = torch.argmax(probs).item()
        
        return {
            "label": self.labels[predicted_idx],
            "confidence": round(probs[0][predicted_idx].item(), 4)
        }

# Инициализация классификатора с кэшированием
@st.cache_resource
def load_classifier():
    return TextClassifier('model')

classifier = load_classifier()

# Интерфейс Streamlit
st.title("Классификатор текста")
st.write("Определение категории текста (Спорт, Экономика, Наука и др.)")

text = st.text_area("Введите текст для классификации:", height=200)

if st.button("Классифицировать") and text:
    with st.spinner('Анализируем текст...'):
        start_time = time.time()
        result = classifier.predict(text)
        elapsed = time.time() - start_time
        
        st.success(f"Результат: {result['label']}")
        st.info(f"Уверенность: {result['confidence']*100:.2f}%")
        st.caption(f"Время обработки: {elapsed:.2f} сек")
        
        # Визуализация
        st.bar_chart({label: result['confidence'] if label == result['label'] else 0.01 
                     for label in classifier.labels})
else:
    st.warning("Пожалуйста, введите текст")
