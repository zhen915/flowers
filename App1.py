import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('best.pt')

st.title("植物辨識應用")
st.write("使用相機或上傳圖片來偵測並辨識身邊的花草，學習更多植物知識。")

option = st.radio("選擇來源", ("拍照", "上傳圖片"))
if option == "拍照":
    st.subheader("相機預覽")
    cap = cv2.VideoCapture(0)
    try:
        if st.button("拍照"):
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('photo.jpg', frame)
                st.image(frame, channels="BGR", caption="拍攝的影像")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        # 釋放相機資源
        cap.release()
        
elif option == "上傳圖片":
    uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上傳的圖片", use_column_width=True)

st.subheader("辨識結果")
if 'image' in locals():
    image_np = np.array(image)
    
    results = model(image_np)
    predictions = results[0].boxes.data.cpu().numpy()

    if len(predictions) > 0:
        for row in predictions:
            plant_name = model.names[int(row[5])]  # 假設類別標籤在第六欄
            st.write(f"植物名稱：{plant_name}")

            # 根據植物名稱顯示更多資訊（需自行填入資料）
            if plant_name == "特定植物":
                st.write("季節：春天")
                st.write("藥用價值")
                st.write("分佈")
                st.write("毒性")
                st.write("是否可食用")
    else:
        st.write("未能辨識出植物")
        
st.subheader("環保資訊")
st.write("""
透過認識植物，我們可以更了解植物在減緩全球暖化中的角色。
植物不僅能提供氧氣，還能吸收二氧化碳，對維持生態平衡有很大幫助。
""")

