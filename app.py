# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict, predict_demo

from rembg import remove
from io import BytesIO
import base64


def remove_background(img):
    with BytesIO() as f:
        img.save(f, format="PNG")
        f.seek(0)
        img_data = f.read()
    result = remove(img_data)
    result_img = Image.open(BytesIO(result)).convert("RGBA")
    return result_img


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.write("オリジナルの画像認識モデルを使って何の画像かを判定します。")
st.sidebar.header("「飛行機, 自動車, 鳥, 猫, 鹿, 犬, カエル, 馬, 船, トラック」の中から分類します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")


        # 予測
        results_demo = predict_demo(img)

        # 結果の表示
        st.subheader("通常版AIによる判定結果")
        n_top = 3  # 確率が高い順に3位まで返す
        for result_demo in results_demo[:n_top]:
            st.write(str(round(result_demo[2]*100, 2)) + "%の確率で" + result_demo[0] + "です。")

        # 円グラフの表示
        pie_labels_demo = [result_demo[1] for result_demo in results_demo[:n_top]]
        pie_labels_demo.append("others")
        pie_probs_demo = [result_demo[2] for result_demo in results_demo[:n_top]]
        pie_probs_demo.append(sum([result_demo[2] for result_demo in results_demo[n_top:]]))
        fig_demo, ax_demo = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax_demo.pie(pie_probs_demo, labels=pie_labels_demo, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig_demo)


        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        results = predict(img)

        # 結果の表示
        st.subheader("改良版AIによる判定結果")
        n_top = 3  # 確率が高い順に3位まで返す
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")

        # 円グラフの表示
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)


        st.subheader("↓「背景を除去した画像の予測結果を以下に表示します」↓")

        img_no_bg = remove_background(img)

        # 予測
        results = predict(img_no_bg)

        # 背景除去後の画像の表示
        st.subheader("背景を除去した画像")
        st.image(img_no_bg, caption="背景を除去した画像", width=480)

        # 結果の表示
        st.subheader("改良版AIによる背景を除去した画像の判定結果")
        n_top = 3
        for result in results[:n_top]:
            st.write(str(round(result[2] * 100, 2)) + "%の確率で" + result[0] + "です。")

        # 円グラフの表示
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)
