import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
from PIL import Image, ImageOps

# ---------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Miniprojeto - MNIST - Classificação de Dígitos",
    page_icon="🔢",
    layout="wide"
)

# ---------------------------------------------------------
# FUNÇÃO NECESSÁRIA PARA CARREGAR O MODELO
# ---------------------------------------------------------
def normalizar_pixels(X):
    return X / 255.0


# ---------------------------------------------------------
# CARREGAMENTO DO MODELO
# ---------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    return joblib.load("models/mlp_final.joblib")


modelo = carregar_modelo()


# ---------------------------------------------------------
# TÍTULO
# ---------------------------------------------------------
st.title("🔢 Classificador de Dígitos MNIST")
st.markdown(
    """
    Aplicação desenvolvida para reconhecer dígitos manuscritos de **0 a 9**
    utilizando um modelo de **Machine Learning baseado em MLPClassifier**.
    """
)

st.divider()


# ---------------------------------------------------------
# MENU LATERAL
# ---------------------------------------------------------
pagina = st.sidebar.radio(
    "Navegação",
    [
        "Predição",
        "Informações do Modelo",
        "Como funciona",
        "Sobre o projeto"
    ]
)


# =========================================================
# PÁGINA 1 - PREDIÇÃO
# =========================================================
if pagina == "Predição":

    st.header("Predição de dígitos")

    st.write(
        "Envie uma imagem contendo um único número manuscrito."
    )

    arquivo = st.file_uploader(
        "Selecione uma imagem",
        type=["png", "jpg", "jpeg"]
    )

    if arquivo is not None:

        imagem_original = Image.open(arquivo).convert("L")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagem original")
            st.image(
                imagem_original,
                width=300
            )

        # -------------------------------------------------
        # PRÉ-PROCESSAMENTO
        # -------------------------------------------------

        imagem = ImageOps.invert(imagem_original)

        imagem = imagem.resize((28, 28))

        imagem_array = np.array(imagem)

        # vetor 784 pixels
        entrada = imagem_array.reshape(1, -1)

        with col2:

            st.subheader("Imagem processada")

            st.image(
                imagem,
                width=300
            )

            st.caption(
                "Imagem convertida para escala de cinza e redimensionada para 28 × 28 pixels."
            )


        st.divider()

        # -------------------------------------------------
        # BOTÃO DE PREDIÇÃO
        # -------------------------------------------------

        if st.button(
            "🔎 Realizar predição",
            use_container_width=True
        ):

            inicio = time.perf_counter()

            predicao = modelo.predict(entrada)[0]

            fim = time.perf_counter()

            tempo_predicao = fim - inicio


            # -------------------------------------------------
            # PROBABILIDADES
            # -------------------------------------------------

            probabilidades = None

            if hasattr(modelo, "predict_proba"):

                probabilidades = modelo.predict_proba(entrada)[0]

                confianca = probabilidades[int(predicao)] * 100

            else:

                confianca = None


            # -------------------------------------------------
            # RESULTADOS
            # -------------------------------------------------

            st.success(
                f"### Dígito previsto: {predicao}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predição",
                    int(predicao)
                )

            with col2:

                if confianca is not None:

                    st.metric(
                        "Confiança",
                        f"{confianca:.2f}%"
                    )

                else:

                    st.metric(
                        "Confiança",
                        "Não disponível"
                    )

            with col3:

                st.metric(
                    "Tempo de predição",
                    f"{tempo_predicao * 1000:.3f} ms"
                )


            # -------------------------------------------------
            # GRÁFICO DE PROBABILIDADES
            # -------------------------------------------------

            if probabilidades is not None:

                st.subheader(
                    "Probabilidade para cada classe"
                )

                df_probabilidades = pd.DataFrame(
                    {
                        "Dígito": list(range(10)),
                        "Probabilidade (%)":
                            probabilidades * 100
                    }
                )

                df_probabilidades = (
                    df_probabilidades
                    .set_index("Dígito")
                )

                st.bar_chart(
                    df_probabilidades
                )


                # -------------------------------------------------
                # TABELA
                # -------------------------------------------------

                with st.expander(
                    "Ver probabilidades detalhadas"
                ):

                    st.dataframe(
                        df_probabilidades.style.format(
                            "{:.2f}%"
                        ),
                        use_container_width=True
                    )


            # -------------------------------------------------
            # INTERPRETAÇÃO
            # -------------------------------------------------

            st.subheader("Interpretação")

            if confianca is not None:

                if confianca >= 90:

                    st.success(
                        "O modelo apresenta alta confiança nesta predição."
                    )

                elif confianca >= 70:

                    st.warning(
                        "O modelo apresenta confiança moderada nesta predição."
                    )

                else:

                    st.error(
                        "O modelo apresenta baixa confiança. "
                        "A imagem pode apresentar características diferentes "
                        "das imagens utilizadas durante o treinamento."
                    )


# =========================================================
# PÁGINA 2 - MODELO
# =========================================================

elif pagina == "Informações do Modelo":

    st.header("Informações do modelo")

    st.write(
        """
        O classificador utilizado nesta aplicação foi treinado
        utilizando o conjunto de dados **MNIST**.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Tipo do modelo",
        "MLP"
    )

    col2.metric(
        "Classes",
        "10"
    )

    col3.metric(
        "Entrada",
        "784"
    )

    col4.metric(
        "Imagem",
        "28 × 28"
    )


    st.subheader("Estrutura da entrada")

    st.write(
        """
        Cada imagem possui resolução de **28 × 28 pixels**.

        Portanto:

        **28 × 28 = 784 características**

        Cada pixel da imagem passa a representar uma variável
        utilizada pelo modelo para realizar a classificação.
        """
    )


    st.subheader("Algoritmo")

    st.write(
        """
        O modelo utilizado é uma **Rede Neural Artificial do tipo
        Multi-Layer Perceptron (MLP)**.

        Durante o treinamento, a rede aprende padrões presentes
        nos pixels das imagens e utiliza esses padrões para
        identificar qual dígito está representado.
        """
    )


    with st.expander(
        "Mostrar parâmetros do modelo"
    ):

        try:

            st.write(
                modelo.get_params()
            )

        except Exception:

            st.info(
                "Os parâmetros internos não estão disponíveis "
                "para este objeto."
            )


# =========================================================
# PÁGINA 3 - COMO FUNCIONA
# =========================================================

elif pagina == "Como funciona":

    st.header("Como funciona a classificação")

    st.markdown(
        """
        ### 1. Entrada da imagem

        O usuário envia uma imagem contendo um dígito.

        ### 2. Conversão

        A imagem é convertida para escala de cinza.

        ### 3. Redimensionamento

        A imagem é transformada em uma matriz de:

        **28 × 28 pixels**

        ### 4. Vetorização

        A matriz bidimensional é transformada em um vetor:

        **28 × 28 = 784 características**

        ### 5. Normalização

        Os pixels originalmente podem assumir valores entre:

        **0 e 255**

        Durante o pré-processamento, esses valores são convertidos
        para uma escala entre:

        **0 e 1**

        ### 6. Predição

        O vetor é enviado ao modelo MLP.

        O modelo calcula qual das classes de **0 a 9** apresenta
        maior probabilidade.

        ### 7. Resultado

        A aplicação exibe:

        - dígito previsto;
        - confiança da predição;
        - probabilidades das classes;
        - tempo de predição.
        """
    )


# =========================================================
# PÁGINA 4 - SOBRE
# =========================================================

elif pagina == "Sobre o projeto":

    st.header("Sobre o projeto")

    st.markdown(
        """
        Este projeto foi desenvolvido como parte de um estudo de
        **Machine Learning aplicado à classificação de imagens**.

        O objetivo é construir um sistema capaz de reconhecer
        automaticamente dígitos manuscritos.

        O projeto envolve diversas etapas do ciclo de Machine Learning:

        **Análise Exploratória dos Dados**

        ↓

        **Pré-processamento**

        ↓

        **Normalização**

        ↓

        **Modelagem**

        ↓

        **Treinamento**

        ↓

        **Avaliação**

        ↓

        **Seleção do modelo final**

        ↓

        **Deploy com Streamlit**

        O modelo final é armazenado utilizando **Joblib** e carregado
        pela aplicação para realizar previsões em novas imagens.
        """
    )


# ---------------------------------------------------------
# RODAPÉ
# ---------------------------------------------------------

st.divider()

st.caption(
    "Projeto MNIST • Machine Learning • Streamlit"
)