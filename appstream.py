# ============================================================
# IMPORTAÇÕES
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas


# ============================================================
# FUNÇÃO USADA NO PIPELINE SALVO
# ============================================================

def normalizar_pixels(X):
    """
    Função utilizada no Pipeline salvo com joblib.

    IMPORTANTE:
    Ela precisa existir antes do joblib.load(),
    porque o modelo salvo contém um FunctionTransformer
    que referencia esta função.
    """

    return (
        X.astype(np.float32)
        / 255.0
    )


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Classificador MNIST",
    page_icon="✍️",
    layout="wide"
)


st.title("✍️ Classificador de Dígitos Manuscritos")

st.write(
    """
    Aplicação desenvolvida com **Streamlit** para classificar
    dígitos manuscritos de **0 a 9** utilizando o modelo
    **Multilayer Perceptron (MLP)** treinado no conjunto MNIST.

    Você pode enviar uma imagem ou desenhar diretamente na tela.
    """
)


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CAMINHO_MODELO = (
    BASE_DIR
    / "models"
    / "mlp_final.joblib"
)


# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================

@st.cache_resource
def carregar_modelo():

    if not CAMINHO_MODELO.exists():

        st.error(
            f"Modelo não encontrado em:\n{CAMINHO_MODELO}"
        )

        st.stop()

    return joblib.load(
        CAMINHO_MODELO
    )


modelo = carregar_modelo()


# ============================================================
# DETECÇÃO AUTOMÁTICA DO FUNDO
# ============================================================

def detectar_fundo(
    img_array
):

    """
    Analisa pixels das bordas da imagem para estimar
    se o fundo é claro ou escuro.
    """

    altura, largura = img_array.shape

    margem_y = max(
        1,
        int(altura * 0.08)
    )

    margem_x = max(
        1,
        int(largura * 0.08)
    )

    bordas = np.concatenate([
        img_array[
            :margem_y,
            :
        ].ravel(),

        img_array[
            -margem_y:,
            :
        ].ravel(),

        img_array[
            :,
            :margem_x
        ].ravel(),

        img_array[
            :,
            -margem_x:
        ].ravel()
    ])

    intensidade_fundo = float(
        np.median(bordas)
    )

    if intensidade_fundo > 127:

        fundo = "Claro"

    else:

        fundo = "Escuro"

    return (
        fundo,
        intensidade_fundo
    )


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def preprocessar_imagem(
    imagem,
    modo_fundo="Automático",
    limiar=60
):

    """
    Pré-processa uma imagem para o padrão MNIST.

    Etapas:
    - converte para escala de cinza;
    - detecta automaticamente o fundo;
    - inverte apenas quando necessário;
    - remove ruído;
    - localiza o dígito;
    - recorta;
    - redimensiona;
    - centraliza em canvas 28x28;
    - mantém pixels entre 0 e 255.

    O Pipeline salvo já realiza a divisão por 255.
    """

    # --------------------------------------------------------
    # Converter para PIL
    # --------------------------------------------------------

    if not isinstance(
        imagem,
        Image.Image
    ):

        imagem = Image.open(
            imagem
        )

    # --------------------------------------------------------
    # Escala de cinza
    # --------------------------------------------------------

    imagem = imagem.convert(
        "L"
    )

    img = np.array(
        imagem,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Detectar fundo
    # --------------------------------------------------------

    (
        fundo_detectado,
        intensidade_fundo
    ) = detectar_fundo(
        img
    )

    # --------------------------------------------------------
    # Definir se deve inverter
    # --------------------------------------------------------

    if modo_fundo == "Automático":

        inverter = (
            fundo_detectado
            == "Claro"
        )

    elif modo_fundo == "Fundo claro":

        inverter = True

    elif modo_fundo == "Fundo escuro":

        inverter = False

    else:

        inverter = False

    # --------------------------------------------------------
    # Padronizar:
    #
    # fundo preto
    # dígito branco/claro
    # --------------------------------------------------------

    if inverter:

        img = (
            255 - img
        )

    # --------------------------------------------------------
    # Remover ruído fraco
    # --------------------------------------------------------

    img = np.where(
        img >= limiar,
        img,
        0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Localizar região do dígito
    # --------------------------------------------------------

    mascara = (
        img > 0
    )

    coordenadas = np.argwhere(
        mascara
    )

    if coordenadas.size == 0:

        raise ValueError(
            "Nenhum dígito foi detectado na imagem."
        )

    y_min, x_min = coordenadas.min(
        axis=0
    )

    y_max, x_max = coordenadas.max(
        axis=0
    )

    # --------------------------------------------------------
    # Margem
    # --------------------------------------------------------

    margem = 2

    y_min = max(
        0,
        y_min - margem
    )

    x_min = max(
        0,
        x_min - margem
    )

    y_max = min(
        img.shape[0] - 1,
        y_max + margem
    )

    x_max = min(
        img.shape[1] - 1,
        x_max + margem
    )

    # --------------------------------------------------------
    # Recortar dígito
    # --------------------------------------------------------

    img_recortada = img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    imagem_recortada = Image.fromarray(
        img_recortada
    )

    # --------------------------------------------------------
    # Redimensionar mantendo proporção
    #
    # MNIST normalmente deixa o dígito ocupando cerca
    # de 20 pixels dentro da imagem 28x28.
    # --------------------------------------------------------

    imagem_recortada.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Canvas MNIST
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    x = (
        28
        - imagem_recortada.width
    ) // 2

    y = (
        28
        - imagem_recortada.height
    ) // 2

    canvas.paste(
        imagem_recortada,
        (x, y)
    )

    # --------------------------------------------------------
    # Converter para NumPy
    #
    # NÃO normalizamos para 0-1 aqui.
    # O pipeline salvo já faz isso.
    # --------------------------------------------------------

    imagem_final = np.array(
        canvas,
        dtype=np.float32
    )

    return (
        imagem_final,
        fundo_detectado,
        intensidade_fundo,
        inverter
    )


# ============================================================
# PREDIÇÃO
# ============================================================

def realizar_predicao(
    imagem_processada
):

    entrada = imagem_processada.reshape(
        1,
        -1
    )

    predicao = int(
        modelo.predict(
            entrada
        )[0]
    )

    probabilidades = modelo.predict_proba(
        entrada
    )[0]

    confianca = float(
        probabilidades.max()
    )

    return (
        predicao,
        probabilidades,
        confianca
    )


# ============================================================
# GRÁFICO
# ============================================================

def criar_grafico_probabilidades(
    probabilidades
):

    classes = np.arange(
        10
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        classes,
        probabilidades
    )

    ax.set_xlabel(
        "Dígito"
    )

    ax.set_ylabel(
        "Probabilidade"
    )

    ax.set_title(
        "Probabilidade por classe"
    )

    ax.set_xticks(
        classes
    )

    ax.set_ylim(
        0,
        1
    )

    plt.tight_layout()

    return fig


# ============================================================
# MÉTODO DE ENTRADA
# ============================================================

st.divider()

st.subheader(
    "Entrada da imagem"
)

modo_entrada = st.radio(
    "Escolha como deseja fornecer o dígito:",
    [
        "Upload de imagem",
        "Desenhar na tela"
    ],
    horizontal=True
)


imagem_entrada = None
origem_canvas = False


# ============================================================
# OPÇÃO 1 - UPLOAD
# ============================================================

if modo_entrada == "Upload de imagem":

    arquivo = st.file_uploader(
        "Envie uma imagem contendo um dígito manuscrito",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp"
        ]
    )

    if arquivo is not None:

        imagem_entrada = Image.open(
            arquivo
        )

        st.subheader(
            "Imagem enviada"
        )

        st.image(
            imagem_entrada,
            width=350
        )


# ============================================================
# OPÇÃO 2 - CANVAS
# ============================================================

else:

    origem_canvas = True

    st.subheader(
        "Desenhe um dígito"
    )

    st.write(
        """
        Use o mouse para desenhar um número de **0 a 9**.

        O desenho utiliza **fundo preto e traço branco**,
        semelhante ao padrão do MNIST.
        """
    )

    espessura = st.slider(
        "Espessura do traço",
        min_value=5,
        max_value=35,
        value=18,
        step=1
    )

    canvas_result = st_canvas(

        fill_color="rgba(255, 255, 255, 1)",

        stroke_width=espessura,

        stroke_color="#FFFFFF",

        background_color="#000000",

        width=280,

        height=280,

        drawing_mode="freedraw",

        update_streamlit=True,

        key="canvas_mnist"
    )

    # --------------------------------------------------------
    # Converter canvas em PIL
    # --------------------------------------------------------

    if canvas_result.image_data is not None:

        img_canvas = (
            canvas_result
            .image_data[
                :, :, :3
            ]
            .astype(
                np.uint8
            )
        )

        imagem_pil = Image.fromarray(
            img_canvas
        )

        imagem_cinza = np.array(
            imagem_pil.convert("L")
        )

        # ----------------------------------------------------
        # Verificar se usuário realmente desenhou
        # ----------------------------------------------------

        quantidade_pixels = np.sum(
            imagem_cinza > 20
        )

        if quantidade_pixels > 20:

            imagem_entrada = imagem_pil

        else:

            imagem_entrada = None

            st.info(
                "Desenhe um dígito no quadro para realizar a classificação."
            )


# ============================================================
# PROCESSAMENTO
# ============================================================

if imagem_entrada is not None:

    st.divider()

    st.subheader(
        "Pré-processamento"
    )

    # ========================================================
    # CONFIGURAÇÕES DIFERENTES PARA CANVAS E UPLOAD
    # ========================================================

    if origem_canvas:

        # Canvas já possui fundo preto e desenho branco.

        modo_fundo = "Fundo escuro"

        limiar = 10

        st.info(
            """
            Desenho realizado diretamente no canvas.
            O fundo já está no padrão do MNIST,
            portanto não é necessária inversão.
            """
        )

    else:

        col1, col2 = st.columns(
            2
        )

        with col1:

            modo_fundo = st.selectbox(
                "Tratamento do fundo",
                [
                    "Automático",
                    "Fundo claro",
                    "Fundo escuro"
                ]
            )

        with col2:

            limiar = st.slider(
                "Limiar de remoção de ruído",
                min_value=0,
                max_value=150,
                value=60,
                step=5
            )

        st.caption(
            """
            No modo automático, o aplicativo analisa
            as bordas da imagem para detectar se
            o fundo é claro ou escuro.
            """
        )

    # ========================================================
    # EXECUTAR PRÉ-PROCESSAMENTO
    # ========================================================

    try:

        (
            imagem_processada,
            fundo_detectado,
            intensidade_fundo,
            invertida
        ) = preprocessar_imagem(
            imagem_entrada,
            modo_fundo=modo_fundo,
            limiar=limiar
        )

        # ====================================================
        # INFORMAÇÕES
        # ====================================================

        st.subheader(
            "Análise da imagem"
        )

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "Fundo detectado",
                fundo_detectado
            )

        with col2:

            st.metric(
                "Intensidade do fundo",
                f"{intensidade_fundo:.0f}"
            )

        with col3:

            st.metric(
                "Imagem invertida",
                "Sim" if invertida else "Não"
            )

        # ====================================================
        # ORIGINAL + PROCESSADA
        # ====================================================

        st.subheader(
            "Comparação"
        )

        col_original, col_processada = st.columns(
            2
        )

        with col_original:

            st.write(
                "**Imagem original**"
            )

            st.image(
                imagem_entrada,
                width=300
            )

        with col_processada:

            st.write(
                "**Imagem processada**"
            )

            st.image(
                imagem_processada,
                width=300,
                clamp=True
            )

            st.caption(
                "Imagem final 28 × 28 pixels"
            )

        # ====================================================
        # PREDIÇÃO
        # ====================================================

        (
            predicao,
            probabilidades,
            confianca
        ) = realizar_predicao(
            imagem_processada
        )

        st.divider()

        st.subheader(
            "Resultado"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            st.metric(
                "Dígito previsto",
                predicao
            )

        with col2:

            st.metric(
                "Confiança",
                f"{confianca * 100:.2f}%"
            )

        # ====================================================
        # INTERPRETAÇÃO DA CONFIANÇA
        # ====================================================

        if confianca >= 0.80:

            st.success(
                f"Predição: {predicao} — "
                f"confiança de {confianca * 100:.2f}%."
            )

        elif confianca >= 0.50:

            st.warning(
                f"Predição: {predicao} — "
                f"confiança moderada de "
                f"{confianca * 100:.2f}%."
            )

        else:

            st.warning(
                f"Predição: {predicao} — "
                f"baixa confiança de "
                f"{confianca * 100:.2f}%."
            )

        # ====================================================
        # GRÁFICO
        # ====================================================

        st.subheader(
            "Probabilidades por classe"
        )

        fig = criar_grafico_probabilidades(
            probabilidades
        )

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )

        # ====================================================
        # TABELA
        # ====================================================

        st.subheader(
            "Probabilidade de cada dígito"
        )

        df_probabilidades = pd.DataFrame({
            "Dígito":
                np.arange(10),

            "Probabilidade (%)":
                probabilidades * 100
        })

        df_probabilidades[
            "Probabilidade (%)"
        ] = (
            df_probabilidades[
                "Probabilidade (%)"
            ]
            .round(2)
        )

        df_probabilidades = (
            df_probabilidades
            .sort_values(
                by="Probabilidade (%)",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        st.dataframe(
            df_probabilidades,
            use_container_width=True,
            hide_index=True
        )

    except Exception as erro:

        st.error(
            f"Erro durante o processamento: {erro}"
        )


# ============================================================
# INFORMAÇÕES SOBRE O PROJETO
# ============================================================

st.divider()

with st.expander(
    "Sobre o modelo e o projeto"
):

    st.markdown(
        """
        ### Modelo utilizado

        O aplicativo utiliza uma **Multilayer Perceptron (MLP)**
        treinada com o conjunto de dados MNIST.

        Durante o projeto foram comparados:

        - K-Nearest Neighbors (KNN);
        - Random Forest;
        - Multilayer Perceptron (MLP).

        A MLP apresentou o melhor desempenho geral,
        atingindo aproximadamente **98,01% de acurácia**
        no conjunto de teste MNIST.

        ### Pré-processamento

        As imagens externas são convertidas para o padrão:

        - escala de cinza;
        - fundo preto;
        - dígito claro;
        - dimensão 28 × 28 pixels.

        O próprio pipeline do modelo realiza a normalização
        dos pixels de 0–255 para 0–1.

        ### Importante

        A confiança apresentada pelo classificador não representa
        garantia de acerto.

        Durante os experimentos do projeto foram observados casos
        de elevada confiança em classificações incorretas,
        demonstrando o fenômeno de **overconfidence**.
        """
    )