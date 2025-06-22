
import json
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from google.colab import userdata
import cv2
import numpy as np
import requests
import base64
import uuid
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

st.set_page_config(
    page_title="의약품 분리배출 AI",
    page_icon="🏥",
    layout="wide"
)

disposal_map = {
    "알약_블리스터팩": "겉 포장지(종이 또는 플라스틱 필름)를 제거하고, 플라스틱 블리스터팩은 그대로 의약품 수거함에 버리세요.",
    "알약_블리스터팩_박스": "종이 박스는 분리하여 종이 재활용함에 버리고, 겉 포장지(종이 또는 플라스틱 필름)를 제거한 후 플라스틱 블리스터팩은 의약품 수거함에 버리세요.",
    "유리병": "약은 폐의약품 수거함에 버리고, 유리병은 내용물을 완전히 비운 후 깨끗이 헹궈 병류로 분리배출하세요.",
    "안약": "용기 그대로 의약품 수거함에 버리세요. 남은 약액은 그대로 두셔도 됩니다.",
    "연고": "튜브를 완전히 비우고 의약품 수거함에 버리세요."
}

text_descriptions = [
    "pill blister pack with plastic packaging",
    "pill blister pack in cardboard box",
    "glass bottle for medicine",
    "small eye drop bottle",
    "ointment tube for skin"
]

labels = ["알약_블리스터팩", "알약_블리스터팩_박스", "유리병", "안약", "연고"]

folder_to_label = {
    "알약_블리스터팩": 0,
    "알약_블리스터팩_박스": 1,
    "유리병": 2,
    "안약": 3,
    "연고": 4
}

medication_keywords = {
    "pill_related": [
        "캡슐", "mg", "정", "캡", "알약", "환", "tablet", "capsule", "cap", "pill",
        "Tablet", "Capsule", "Cap", "Pill", "TABLET", "CAPSULE", "CAP", "PILL",
        "MG", "mg", "Mg", "mcg", "µg", "IU", "unit", "Units", "UNITS"
    ],
    "ointment_related": [
        "연고", "크림", "젤", "로션", "밤", "연고제", "%",
        "ointment", "cream", "gel", "lotion", "balm",
        "Ointment", "Cream", "Gel", "Lotion", "Balm",
        "OINTMENT", "CREAM", "GEL", "LOTION", "BALM"
    ],
    "eye_drop_related": [
        "안약", "점안액", "점안제", "eye", "drop", "drops",
        "Eye", "Drop", "Drops", "EYE", "DROP", "DROPS",
        "ophthalmic", "Ophthalmic", "OPHTHALMIC"
    ]
}

def analyze_medication_keywords(ocr_text):
    if not ocr_text:
        return {"pill": 0, "ointment": 0, "eye_drop": 0}, {}

    scores = {
        "pill": 0,
        "ointment": 0,
        "eye_drop": 0
    }

    found_keywords = {
        "pill": [],
        "ointment": [],
        "eye_drop": []
    }

    for keyword in medication_keywords["pill_related"]:
        if keyword in ocr_text:
            scores["pill"] += 1
            found_keywords["pill"].append(keyword)

    for keyword in medication_keywords["ointment_related"]:
        if keyword in ocr_text:
            scores["ointment"] += 1
            found_keywords["ointment"].append(keyword)

    for keyword in medication_keywords["eye_drop_related"]:
        if keyword in ocr_text:
            scores["eye_drop"] += 1
            found_keywords["eye_drop"].append(keyword)
    return scores, found_keywords

def adjust_clip_predictions(logits, keyword_scores, boost_factor=0.15):

    adjusted_logits = logits.clone()

    label_keyword_mapping = {
        0: "pill",      # 알약_블리스터팩
        1: "pill",      # 알약_블리스터팩_박스
        2: "pill",      # 유리병
        3: "eye_drop",  # 안약
        4: "ointment"   # 연고
    }

    adjustments = {}
    for label_idx, keyword_type in label_keyword_mapping.items():
        if keyword_scores[keyword_type] > 0:
            boost = boost_factor * keyword_scores[keyword_type]
            adjusted_logits[0][label_idx] += boost
            adjustments[labels[label_idx]] = boost
            print(f"라벨 {labels[label_idx]} 점수 증가: +{boost:.3f} (키워드: {keyword_type}, 개수: {keyword_scores[keyword_type]})")

    return adjusted_logits, adjustments

def preprocess_image_for_ocr(image):
    try:
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        binary_inv = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)

        result_image = Image.fromarray(processed)

        return result_image

    except Exception as e:
        print(f"이미지 전처리 오류: {e}")
        return image

def extract_text_from_image_with_clova(image_path: str):
    secret_key = userdata.get("OCR_CLOVA")
    api_url = "https://s79a9gvq9a.apigw.ntruss.com/custom/v1/42492/9274a7c96357207aa7e436c070ceba3fecf0b7f91679a2a8f72283f165493853/general"

    file_size = os.path.getsize(image_path)
    if file_size > 5 * 1024 * 1024:  # 5MB
        print("⚠️ 파일 크기가 5MB를 초과합니다.")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    headers = {
        "X-OCR-SECRET": secret_key,
        "Content-Type": "application/json"
    }

    payload = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [{
            "name": "temp",
            "format": "jpg",
            "data": image_data
        }]
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        result = response.json()

        first_image = result["images"][0]
        fields = first_image["fields"]
        texts = []

        for i, field in enumerate(fields):
            if "inferText" in field and field["inferText"]:
                text = field["inferText"]
                texts.append(text)
                confidence = field.get("inferConfidence", "N/A")
                print(f"텍스트 {i+1}: '{text}' (신뢰도: {confidence})")

        if not texts:
            return "", 0.0

        extracted_text = " ".join(texts)
        return extracted_text, confidence

    except Exception as e:
        print(f"❌ OCR API 호출 오류: {e}")
        return "", 0.0

def classify_and_dispose(image_path: str, model_clip, processor_clip):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")

    processed_image = preprocess_image_for_ocr(image)
    temp_path = "temp_processed.png"
    processed_image.save(temp_path)

    ocr_text, ocr_confidence = extract_text_from_image_with_clova(temp_path)
    if ocr_text is None:
        ocr_text = ""

    keyword_scores, found_keywords = analyze_medication_keywords(ocr_text)

    inputs = processor_clip(text=text_descriptions, images=image, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_clip(**inputs)

    original_logits = outputs.logits_per_image
    adjusted_logits, logit_adjustments = adjust_clip_predictions(original_logits, keyword_scores)

    original_probs = original_logits.softmax(dim=1)
    adjusted_probs = adjusted_logits.softmax(dim=1)

    original_predicted_idx = torch.argmax(original_probs, dim=1).item()
    original_predicted_label = labels[original_predicted_idx]
    original_confidence = torch.max(original_probs).item()

    predicted_idx = torch.argmax(adjusted_probs, dim=1).item()
    predicted_label = labels[predicted_idx]
    confidence = torch.max(adjusted_probs).item()

    final_label = predicted_label
    ocr_correction_applied = False

    if keyword_scores["ointment"] >= 1 and predicted_label != "연고":
        if confidence < 0.7:
            final_label = "연고"
            confidence = min(confidence + 0.15, 0.95)
            ocr_correction_applied = True

    if keyword_scores["eye_drop"] >= 1 and predicted_label != "안약":
        if confidence < 0.7:
            final_label = "안약"
            confidence = min(confidence + 0.15, 0.95)
            ocr_correction_applied = True

    context = disposal_map.get(final_label, "종류에 따라 적절한 방법으로 의약품 수거함에 버리세요.")

    return {
        "final_label": final_label,
        "disposal_context": context,
        "final_confidence": confidence,
        "ocr_text": ocr_text,
        "ocr_confidence" : ocr_confidence,
        "original_clip_label": original_predicted_label,
        "original_clip_confidence": original_confidence,
        "adjusted_clip_label": predicted_label,
        "adjusted_clip_confidence": torch.max(adjusted_probs).item(),
        "keyword_scores": keyword_scores,
        "found_keywords": found_keywords,
        "logit_adjustments": logit_adjustments,
        "ocr_correction_applied": ocr_correction_applied
    }

@st.cache_resource
def load_model():
    try:
        model_path = "/content/drive/MyDrive/trash_ai/clip_medication_test/best"
        if os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CLIPModel.from_pretrained(model_path).to(device)
            processor = CLIPProcessor.from_pretrained(model_path)
            return model, processor
        else:
            st.error(f"❌ 모델 경로를 찾을 수 없습니다: {model_path}")
            return None, None
    except Exception as e:
        st.error(f"❌ 모델 로딩 오류: {str(e)}")
        return None, None

def main():
    st.title("🏥 의약품 분리배출 AI")
    st.markdown("---")
    st.write("의약품 이미지를 업로드하면 종류를 분류하고 올바른 분리배출 방법을 알려드립니다.")

    with st.sidebar:
        st.header("📋 분류 가능한 의약품")
        st.write("• 💊 알약 블리스터팩")
        st.write("• 📦 알약 블리스터팩 (박스 포함)")
        st.write("• 🍼 유리병")
        st.write("• 👁️ 안약")
        st.write("• 🧴 연고")

        st.header("📖 사용 방법")
        st.write("1. 의약품 사진 업로드")
        st.write("2. AI 자동 분석 대기")
        st.write("3. 분리배출 방법 확인")
        st.warning("⚠️ 텍스트가 선명하게 보이도록 촬영하면 더 정확한 분류가 가능합니다.")

        st.header("🔧 시스템 정보")
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.write(f"디바이스: {device}")

    with st.spinner("모델 로딩 중... 잠시만 기다려주세요."):
        model_clip, processor_clip = load_model()

    if model_clip is None or processor_clip is None:
        st.error("🚫 모델을 로드할 수 없습니다. 모델 경로와 파일을 확인해주세요.")
        st.info("💡 모델 학습이 완료되었는지 확인하세요.")
        return

    st.success("✅ 모델 로딩 완료!")

    uploaded_file = st.file_uploader(
        "📸 의약품 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png"],
        help="JPG, JPEG, PNG 형식의 이미지를 업로드할 수 있습니다. 텍스트가 선명하게 보이는 이미지일수록 정확도가 높아집니다."
    )

    if uploaded_file is not None:
        try:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📷 업로드된 이미지")
                st.image(uploaded_file, caption="분석할 이미지", use_container_width=True)

            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("🔍 분석 시작", type="primary"):
                with col2:
                    st.subheader("📊 분석 결과")

                    # 프로그레스 바
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("이미지 전처리 중...")
                    progress_bar.progress(20)

                    status_text.text("OCR 텍스트 추출 중...")
                    progress_bar.progress(40)

                    status_text.text("키워드 분석 중...")
                    progress_bar.progress(50)

                    status_text.text("AI 모델 실행 중...")
                    progress_bar.progress(70)

                    result = classify_and_dispose(temp_path, model_clip, processor_clip)

                    progress_bar.progress(90)
                    status_text.text("결과 생성 중...")

                    progress_bar.progress(100)
                    status_text.text("분석 완료!")

                    st.success("✅ 분석 완료!")

                    st.markdown(f"""
                    <div style="padding: 1rem; border-radius: 0.5rem; background-color: #e8f4fd; margin: 1rem 0; border-left: 4px solid #1f77b4;">
                        <h3>🔍 최종 분류 결과: {result['final_label']}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    if result['final_confidence'] > 0.8:
                        st.success(f"📊 신뢰도: {result['final_confidence']:.1%} (높음)")
                    elif result['final_confidence'] > 0.6:
                        st.warning(f"📊 신뢰도: {result['final_confidence']:.1%} (보통)")
                    else:
                        st.error(f"📊 신뢰도: {result['final_confidence']:.1%} (낮음)")
                        st.warning("신뢰도가 낮습니다. 더 선명한 이미지를 사용해보세요.")

                    st.info(f"""♻️ **분리배출 방법**: {result['disposal_context']}""")

                    with st.expander("🔍 분석 과정 상세 정보", expanded=True):
                        st.markdown("### 1️⃣ CLIP 모델 원본 예측")
                        st.write(f"- 예측: **{result['original_clip_label']}**")
                        st.write(f"- 신뢰도: {result['original_clip_confidence']:.1%}")

                        st.markdown("### 2️⃣ OCR 키워드 분석")
                        if result['ocr_text'] and result['ocr_confidence']:
                            st.write("**추출된 텍스트:**")
                            st.code(result['ocr_text'])
                            st.write("**OCR 결과 신뢰도:**")
                            st.code(result['ocr_confidence'])

                            col_k1, col_k2 = st.columns(2)
                            with col_k1:
                                st.write("**키워드 점수:**")
                                for category, score in result['keyword_scores'].items():
                                    emoji = {"pill": "💊", "liquid": "🧴", "ointment": "🧴", "eye_drop": "👁️"}
                                    st.write(f"{emoji.get(category, '•')} {category}: {score}개")

                            with col_k2:
                                st.write("**발견된 키워드:**")
                                for category, keywords in result['found_keywords'].items():
                                    if keywords:
                                        st.write(f"**{category}:** {', '.join(keywords)}")
                        else:
                            st.warning("OCR로 텍스트를 추출하지 못했습니다.")

                        st.markdown("### 3️⃣ 키워드 기반 점수 조정")
                        if result['logit_adjustments']:
                            st.write("**점수 조정 내역:**")
                            for label, boost in result['logit_adjustments'].items():
                                st.write(f"- {label}: +{boost:.3f}")
                            st.write(f"- 조정 후 예측: **{result['adjusted_clip_label']}**")
                            st.write(f"- 조정 후 신뢰도: {result['adjusted_clip_confidence']:.1%}")
                        else:
                            st.write("키워드 기반 점수 조정이 적용되지 않았습니다.")

                        st.markdown("### 4️⃣ OCR 기반 최종 보정")
                        if result['ocr_correction_applied']:
                            st.success(f"✅ OCR 키워드 기반으로 '{result['final_label']}'로 최종 보정되었습니다.")
                        else:
                            st.info("OCR 기반 추가 보정이 적용되지 않았습니다.")

                        st.markdown("### 📋 변경 사항 요약")
                        changes = []
                        if result['original_clip_label'] != result['adjusted_clip_label']:
                            changes.append(f"CLIP 키워드 조정: {result['original_clip_label']} → {result['adjusted_clip_label']}")
                        if result['ocr_correction_applied']:
                            changes.append(f"OCR 기반 보정: {result['adjusted_clip_label']} → {result['final_label']}")

                        if changes:
                            for change in changes:
                                st.write(f"• {change}")
                        else:
                            st.write("• 원본 CLIP 예측이 그대로 유지되었습니다.")

            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
            st.write("오류 상세 정보:")
            st.code(str(e))

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "<p>🌱 환경을 생각하는 AI 기반 의약품 분리배출 도우미 (키워드 분석 시각화 버전)</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
