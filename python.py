# python.py

import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
from google import genai
from google.genai.errors import APIError
import json

# --- Cấu hình trang ---
st.set_page_config(page_title="App Đánh Giá Phương Án Đầu Tư", layout="wide")
st.title("📊 Ứng dụng Đánh Giá Phương Án Đầu Tư Tự Động")

# --- Hàm đọc file Word ---
def read_word_file(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

# --- Hàm gọi AI trích lọc dữ liệu dự án ---
def extract_project_info(text, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.0-flash"

        prompt = f"""
        Hãy đọc kỹ nội dung sau và trích lọc ra các thông tin của dự án đầu tư (chỉ trả về JSON, không giải thích thêm):
        - von_dau_tu (tỷ đồng)
        - dong_doi_du_an (năm)
        - doanh_thu_hang_nam (tỷ đồng)
        - chi_phi_hang_nam (tỷ đồng)
        - wacc (%)
        - thue_suat (%)
        Nội dung:
        {text}
        """

        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text

    except APIError as e:
        return f"Lỗi Gemini API: {e}"
    except Exception as e:
        return f"Lỗi khác: {e}"

# --- Hàm tính dòng tiền và chỉ tiêu hiệu quả ---
def calculate_metrics(von_dau_tu, doanh_thu, chi_phi, thue, wacc, dong_doi):
    dong_tien = []
    for i in range(1, dong_doi + 1):
        loi_nhuan = (doanh_thu - chi_phi) * (1 - thue / 100)
        dong_tien.append(loi_nhuan)

    # NPV
    npv = -von_dau_tu + sum(cf / ((1 + wacc / 100) ** (i + 1)) for i, cf in enumerate(dong_tien))

    # IRR
    cash_flows = [-von_dau_tu] + dong_tien
    irr = np.irr(cash_flows)

    # PP
    cum_cf = np.cumsum(cash_flows)
    pp = np.argmax(cum_cf > 0) if np.any(cum_cf > 0) else None

    # DPP
    discounted_cf = [cf / ((1 + wacc / 100) ** (i + 1)) for i, cf in enumerate(dong_tien)]
    cum_discounted_cf = np.cumsum([-von_dau_tu] + discounted_cf)
    dpp = np.argmax(cum_discounted_cf > 0) if np.any(cum_discounted_cf > 0) else None

    df = pd.DataFrame({
        "Năm": range(1, dong_doi + 1),
        "Dòng tiền ròng (tỷ đồng)": dong_tien
    })
    return df, npv, irr, pp, dpp

# --- Hàm phân tích AI các chỉ tiêu ---
def ai_analysis(npv, irr, pp, dpp, api_key):
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.0-flash"

    prompt = f"""
    Dự án có các chỉ số hiệu quả như sau:
    - NPV: {npv:.2f}
    - IRR: {irr*100:.2f}%
    - PP: {pp} năm
    - DPP: {dpp} năm

    Hãy phân tích hiệu quả đầu tư, độ hấp dẫn, rủi ro và đưa ra khuyến nghị đầu tư ngắn gọn (3–4 đoạn).
    """

    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text

# --- Giao diện chính ---
uploaded_file = st.file_uploader("📁 1. Tải file Word phương án đầu tư", type=["docx"])

if uploaded_file:
    text = read_word_file(uploaded_file)
    st.success("✅ Đã đọc nội dung file thành công.")

    if st.button("🧠 2. Trích lọc dữ liệu bằng AI"):
        api_key = st.secrets.get("GEMINI_API_KEY")

        if api_key:
            with st.spinner("Đang phân tích nội dung bằng Gemini..."):
                extracted = extract_project_info(text, api_key)
                st.subheader("📄 Kết quả trích lọc thông tin dự án (AI)")
                st.code(extracted, language="json")

                try:
                    info = json.loads(extracted)
                    von = float(info.get("von_dau_tu", 0))
                    doi = int(info.get("dong_doi_du_an", 10))
                    dt = float(info.get("doanh_thu_hang_nam", 0))
                    cp = float(info.get("chi_phi_hang_nam", 0))
                    wacc = float(info.get("wacc", 10))
                    thue = float(info.get("thue_suat", 20))

                    df, npv, irr, pp, dpp = calculate_metrics(von, dt, cp, thue, wacc, doi)

                    st.subheader("📈 3. Bảng dòng tiền dự án")
                    st.dataframe(df, use_container_width=True)

                    st.subheader("📊 4. Các chỉ tiêu hiệu quả tài chính")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("NPV (tỷ đồng)", f"{npv:,.2f}")
                    col2.metric("IRR (%)", f"{irr*100:.2f}")
                    col3.metric("PP (năm)", pp if pp is not None else "Không hoàn vốn")
                    col4.metric("DPP (năm)", dpp if dpp is not None else "Không hoàn vốn")

                    if st.button("🤖 5. Phân tích hiệu quả dự án bằng AI"):
                        with st.spinner("Gemini đang phân tích hiệu quả dự án..."):
                            analysis = ai_analysis(npv, irr, pp, dpp, api_key)
                            st.markdown("### 🧩 Kết quả phân tích của AI:")
                            st.info(analysis)
                except Exception as e:
                    st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        else:
            st.error("Không tìm thấy API Key. Hãy cấu hình 'GEMINI_API_KEY' trong Streamlit Secrets.")

else:
    st.info("Vui lòng tải lên file Word để bắt đầu đánh giá.")
