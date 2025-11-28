# train.py
import pickle # Adicionar este import
import os
import pytesseract
from meme_detector import train_demo

# Configurações do Tesseract e Ambiente
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/' # MANTENHA ESSA LINHA PARA CORRIGIR ERRO DE IDIOMA

if __name__ == "__main__":
    print("Iniciando Treinamento...")
    # Use o alias 'fusion_model' que você corrigiu no meme_detector.py
    results = train_demo('data/clean', 'data/manipulated', epochs=8) 

    # Salva o modelo treinado (fusion_model)
    fusion_model = results["fusion_model"]
    fusion_model.save("model_fusion.h5")

    # Salva o encoder visual para a interpretabilidade Grad-CAM
    visual_encoder = results["visual_encoder"]
    visual_encoder.save("visual_encoder.h5") 

# ... (código anterior)
    with open('text_encoder.pkl', 'wb') as f:
        pickle.dump(results["text_encoder"], f)
            
    # Estes prints devem ficar alinhados à esquerda (fora do with open)
    print("Modelo salvo como model_fusion.h5")
    print("Encoder visual salvo como visual_encoder.h5")
    print("TextEncoder (TF-IDF) salvo como text_encoder.pkl")
    pickle.dump(results["text_encoder"], f)
            
    print("Modelo salvo como model_fusion.h5")
    print("Encoder visual salvo como visual_encoder.h5")
    print("TextEncoder (TF-IDF) salvo como text_encoder.pkl")