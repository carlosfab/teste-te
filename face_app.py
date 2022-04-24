# PROJETO FEITO EXCLUSIVAMENTE PARA AULA DA SEMANA DSNP
# NÃO SE TRATA DE UM PROJETO FUNCIONAL, APENAS CONCEITUAL

# importar pacotes necessários
import streamlit as st
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

OUTPUT_WIDTH = 500


def main():
    """Projeto NFT"""

    st.title("NFT Generator")
    st.text("powered by Python")

    opcoes_menu = ["Transformações", "Sobre"]
    choice = st.sidebar.selectbox("Escolha uma Opção", opcoes_menu)

    our_image = Image.open("empty.jpg")

    if choice == 'Transformações':

        image_file = st.file_uploader("Carregue uma foto sua e escolha uma transformação no menu lateral", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.sidebar.image(our_image, width=150)

        filtros = st.sidebar.radio("Filtros", ["Original", "Desenho", "Canny", "Pixel Art"])

        if filtros == 'Desenho':
            converted_image = np.array(our_image.convert('RGB'))
            gray_image = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
            inv_gray_image = 255 - gray_image
            blurred_image = cv2.GaussianBlur(inv_gray_image, (21, 21), 0, 0)
            sketch_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)
            st.image(sketch_image, width=OUTPUT_WIDTH)

        elif filtros == 'Canny':
            converted_image = np.array(our_image.convert('RGB'))
            converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(converted_image, (11, 11), 0)
            canny = cv2.Canny(blur_image, 100, 150)
            st.image(canny, width=OUTPUT_WIDTH)

        elif filtros == 'Pixel Art':
            # código para pixelar baseado no código do cientista da computação
            # Jeffery Russell (jrtechs)

            converted_image = np.array(our_image.convert('RGB'))

            def pixelate(img, w, h):
                height, width = img.shape[:2]
                resized_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                return cv2.resize(resized_img, (width, height), interpolation=cv2.INTER_NEAREST)

            def color_clustering(idx, img, k):
                cluster_values = []
                for _ in range(0, k):
                    cluster_values.append([])
                for r in range(0, idx.shape[0]):
                    for c in range(0, idx.shape[1]):
                        cluster_values[idx[r][c]].append(img[r][c])
                img_cluster = np.copy(img)

                cluster_averages = []

                for i in range(0, k):
                    cluster_averages.append(np.average(cluster_values[i], axis=0))

                for r in range(0, idx.shape[0]):
                    for c in range(0, idx.shape[1]):
                        img_cluster[r][c] = cluster_averages[idx[r][c]]

                return img_cluster

            def segment_img_clr_rgb(img, k):
                img_c = np.copy(img)

                h = img.shape[0]
                w = img.shape[1]

                img_c.shape = (img.shape[0] * img.shape[1], 3)
                kmeans = KMeans(n_clusters=k, random_state=0).fit(img_c).labels_
                kmeans.shape = (h, w)

                return kmeans

            def k_means_image(image, k):
                idx = segment_img_clr_rgb(image, k)
                return color_clustering(idx, image, k)

            img32 = pixelate(converted_image, 32, 32)
            pixel_image = k_means_image(img32, 5)
            st.image(pixel_image, width=OUTPUT_WIDTH)
            # st.image(img32, width=OUTPUT_WIDTH)

        elif filtros == 'Original':
            st.image(our_image, width=OUTPUT_WIDTH)
        else:
            st.image(our_image, width=OUTPUT_WIDTH)

    elif choice == 'Sobre':
        st.subheader("Sobre a Masterclass de Visão Computacional")
        st.markdown("Masterclass disponível em [Sigmoidal](https://sigmoidal.ai)")
        st.text("Carlos Melo")
        st.success("Instagram @carlos_melo.py")


if __name__ == '__main__':
    main()
