# predict.py
import os
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model
from meme_detector import ocr_extract_with_confidences, preprocess_image_for_visual # Importa as funções

# Configurações do Tesseract e Ambiente
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'

# Caminho do modelo (deve corresponder ao nome salvo)
FUSION_MODEL_PATH = "model_fusion.h5" 
IMAGE_PATH = "/Users/samirandrade/Documents/dev/back/meme-detector/data/test/image.png"

# Carrega o modelo treinado
try:
    fusion_model = load_model(FUSION_MODEL_PATH)
    print(f"✅ Modelo carregado com sucesso de {FUSION_MODEL_PATH}")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    print("Certifique-se de que o arquivo model_fusion.h5 foi criado executando 'python train.py' primeiro.")
    exit()

# --- Funções de Inferência Simples ---
def classify_image_simple(path):
    img_cv = cv2.imread(path)

    # 1. OCR (Extrair texto e stats)
    ocr_result = ocr_extract_with_confidences(img_cv)
    text_data = ocr_result['text']
    ocr_stats = np.array([[
        ocr_result['stats']['mean_conf'], 
        ocr_result['stats']['std_conf'], 
        ocr_result['stats']['min_conf']
    ]])

    # **ATENÇÃO:** Para rodar a inferência, você PRECISA do TextEncoder treinado (TF-IDF). 
    # Como o TextEncoder (TF-IDF) é complexo de serializar, 
    # este exemplo simples **NÃO VAI FUNCIONAR** sem carregar o TextEncoder (pickle).
    # A inferência completa DEVE usar predict_with_explainability do meme_detector.

    return "Para a classificação real, use predict_with_explainability do meme_detector.py!"

if __name__ == "__main__":
    print("\n--- INFERÊNCIA SIMPLES ---")
    # Para rodar a inferência de teste, você precisa executar o predict_with_explainability
    # que utiliza o TextEncoder treinado, que não é trivial carregar aqui.
    print("O fluxo de inferência precisa do TextEncoder treinado (TF-IDF) para rodar.")
    print("Você deve usar a função predict_with_explainability do meme_detector.py para o teste real.")