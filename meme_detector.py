"""
meme_detector.py

Pipeline para detecção de memes / screenshots adulteradas integrando:
- pré-processamento de imagem
- OCR (pytesseract) com extração de confidências
- extração de features visuais (MobileNetV2)
- processamento de texto (TF-IDF ou embedding + LSTM opcional)
- fusão e classificador denso
- interpretabilidade: Grad-CAM + listagem de palavras com baixa confiança

Salvar como: meme_detector.py
"""

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


import os
import random
import json
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageOps
import cv2

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Scikit-learn for TF-IDF + metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIG
# -------------------------
# Ajuste o caminho do executável do Tesseract se necessário:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # windows

IMG_SIZE = (224, 224)  # para MobileNetV2
BATCH_SIZE = 16
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -------------------------
# UTILITÁRIOS DE PRÉ-PROCESSAMENTO
# -------------------------
def preprocess_image_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Pré-processamento voltado para OCR: grayscale, denoise, adaptive threshold.
    Input: imagem OpenCV BGR ou grayscale.
    Returns: imagem grayscale final (uint8).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Equalização adaptativa para realçar contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(denoised)
    # Binarização adaptativa
    thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 9)
    return thresh


def preprocess_image_for_visual(img: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    """
    Pré-processamento para rede visual (MobileNetV2): resize + preprocess_input
    Input: imagem OpenCV BGR
    Returns: tensor pronto (target_size + channels), float32
    """
    # convert BGR -> RGB
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(img_rgb)
    pil = pil.resize(target_size)
    arr = np.asarray(pil).astype(np.float32)
    arr = mobilenet_preprocess(arr)  # aplica normalização esperada
    return arr


# -------------------------
# OCR: extração de texto + confidências
# -------------------------
def ocr_extract_with_confidences(img: np.ndarray, psm: int = 6, lang: str = 'por+eng') -> Dict[str, Any]:
    """
    Executa pytesseract e retorna:
    - text: texto completo
    - words: lista de dicts {text, conf, bbox}
    - stats: média e desvio das confidências
    """
    # Pré-processa para OCR
    proc = preprocess_image_for_ocr(img)
    # Config para Tesseract
    config = f'--psm {psm} -l {lang} --oem 3'
    # usa output as data frame style
    data = pytesseract.image_to_data(proc, config=config, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['text'])
    words = []
    confidences = []
    for i in range(n_boxes):
        txt = data['text'][i].strip()
        if txt == "":
            continue
        try:
            conf = float(data['conf'][i])
        except:
            conf = -1.0
        bbox = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        words.append({'text': txt, 'conf': conf, 'bbox': bbox})
        confidences.append(conf)
    full_text = " ".join([w['text'] for w in words])
    if len(confidences) > 0:
        conf_arr = np.array(confidences)
        stats = {'mean_conf': float(np.mean(conf_arr)), 'std_conf': float(np.std(conf_arr)), 'min_conf': float(np.min(conf_arr))}
    else:
        stats = {'mean_conf': 0.0, 'std_conf': 0.0, 'min_conf': 0.0}
    return {'text': full_text, 'words': words, 'stats': stats, 'raw_data': data}


# -------------------------
# FEATURES VISUAIS (MobileNetV2)
# -------------------------
def build_visual_encoder(input_shape=(224,224,3), pooling='avg'):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape, pooling=pooling)
    base.trainable = False  # congelar por padrão; depois pode afinar
    input_layer = layers.Input(shape=input_shape)
    x = base(input_layer)
    # saída é vetor
    model = models.Model(inputs=input_layer, outputs=x, name='visual_encoder')
    return model


# -------------------------
# ENCODER DE TEXTO (TF-IDF ou EMBEDDING + LSTM)
# -------------------------
@dataclass
class TextEncoder:
    use_tfidf: bool = True
    tfidf_vectorizer: TfidfVectorizer = None
    max_tokens: int = 5000

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_tokens, ngram_range=(1,2))
            X = self.tfidf_vectorizer.fit_transform(texts)
            return X.toarray()
        else:
            raise NotImplementedError("Embedding-based encoder not implemented in this script. Use TF-IDF or extend here.")

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.use_tfidf:
            X = self.tfidf_vectorizer.transform(texts)
            return X.toarray()
        else:
            raise NotImplementedError


# -------------------------
# CONSTRUIR MODELO DE FUSÃO
# -------------------------
def build_fusion_classifier(visual_vector_dim: int, text_vector_dim: int, ocr_stats_dim: int = 3, hidden_units=256, dropout=0.3):
    """
    visual_vector_dim: dimensão do output do visual encoder (por pooling)
    text_vector_dim: dimensão do vetor TF-IDF
    ocr_stats_dim: 3 (mean, std, min) ou 0
    """
    v_in = layers.Input(shape=(visual_vector_dim,), name='visual_input')
    t_in = layers.Input(shape=(text_vector_dim,), name='text_input')
    o_in = layers.Input(shape=(ocr_stats_dim,), name='ocr_stats_input')

    x = layers.Concatenate()([v_in, t_in, o_in])
    x = layers.Dense(hidden_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden_units//2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid', name='out')(x)

    model = models.Model(inputs=[v_in, t_in, o_in], outputs=out, name='fusion_classifier')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -------------------------
# FUNÇÕES DE TREINO / AVALIAÇÃO
# -------------------------
def evaluate_model(model, Xv, Xt, Xo, y_true):
    y_prob = model.predict([Xv, Xt, Xo], batch_size=BATCH_SIZE).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }
    return metrics, y_prob, y_pred


# -------------------------
# INTERPRETABILIDADE: Grad-CAM para MobileNetV2
# -------------------------
# Em meme_detector.py

# Em meme_detector.py

def grad_cam_visual_heatmap(visual_model, preprocessed_image: np.ndarray, last_conv_layer_name=None):
    """
    Gera heatmap Grad-CAM.
    Correção: Detecta se a MobileNetV2 está aninhada dentro de outra camada e corrige o erro de sintaxe 'except'.
    """
    # 1. Identificar o modelo interno real (Desembrulhar se necessário)
    inner_model = visual_model
    
    # Verifica se o visual_model tem camadas "escondidas" (Nested Model)
    for layer in visual_model.layers:
        if isinstance(layer, models.Model) or 'mobilenet' in layer.name.lower():
            inner_model = layer
            break

    # 2. Encontrar a última camada de convolução (4D) dentro do modelo correto
    if last_conv_layer_name is None:
        # Tenta pegar 'out_relu' (padrão MobileNetV2)
        try:
            inner_model.get_layer('out_relu')
            last_conv_layer_name = 'out_relu'
        except:
            # Se falhar, varre de trás para frente procurando output 4D
            for layer in reversed(inner_model.layers):
                try:
                    # Verifica se output_shape existe e tem 4 dimensões
                    if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                        last_conv_layer_name = layer.name
                        break
                except AttributeError:  # <--- AQUI ESTAVA O ERRO (Faltava o espaço)
                    continue

    if last_conv_layer_name is None:
        print("⚠️ Grad-CAM: Não foi possível encontrar camada convolucional (4D). Heatmap vazio.")
        return np.zeros((224, 224))

    # 3. Construir modelo de gradiente usando o inner_model
    try:
        last_conv_layer = inner_model.get_layer(last_conv_layer_name)
        
        # Cria um modelo novo que vai da entrada do inner_model até a conv e a saída
        grad_model = models.Model(
            inputs=inner_model.inputs,
            outputs=[last_conv_layer.output, inner_model.output]
        )
    except Exception as e:
        print(f"⚠️ Erro ao montar grafo Grad-CAM: {e}")
        return np.zeros((224, 224))

    # 4. Calcular Gradientes
    img_tensor = np.expand_dims(preprocessed_image, axis=0)
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        score = tf.reduce_mean(predictions)

    grads = tape.gradient(score, conv_outputs)
    
    if grads is None:
        return np.zeros((224, 224))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
        
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
        
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    return heatmap


# -------------------------
# UTIL: criar manipulações sintéticas (para dataset)
# -------------------------
def simulate_text_edit(img_pil: Image.Image, replace_with: str = "FAKE", position: Tuple[int,int]=None, font_path=None):
    """
    Substitui/regenera texto em uma imagem PIL: exemplo simples que desenha texto preto com fundo branco para simular edição.
    Deve ser usado com cuidado; é apenas uma heurística para gerar exemplos manipulados.
    """
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if position is None:
        position = (int(w*0.05), int(h*0.05))
    # choose font
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", size=int(h*0.05))
    except Exception:
        font = ImageFont.load_default()
    # draw a filled rectangle under text to obscure original region
    bbox = draw.textbbox(position, replace_with, font=font)
    rect = [position[0]-5, position[1]-5, bbox[2]+5, bbox[3]+5]
    draw.rectangle(rect, fill=(255,255,255))
    draw.text(position, replace_with, fill=(0,0,0), font=font)
    return img


def simulate_compression_and_noise(img_pil: Image.Image, quality: int=30, gaussian_sigma=1.5):
    """
    Simula compressão JPEG + ruído para adulterar a imagem.
    """
    import io
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    noisy = Image.open(buf).convert('RGB')
    arr = np.array(noisy)
    arr = cv2.GaussianBlur(arr, (0,0), gaussian_sigma)
    return Image.fromarray(arr)


# -------------------------
# EXEMPLO DE FLUXO: preparar dataset e treinar
# -------------------------
def build_dataset_from_folders(clean_dir: str, manipulated_dir: str, sample_limit=None) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Lê imagens de duas pastas com labels:
    - clean_dir -> label 0
    - manipulated_dir -> label 1

    Retorna: imagens (BGR), texts (ocr text), labels
    """
    images = []
    texts = []
    labels = []
    def read_folder(folder, label):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if sample_limit:
            files = files[:sample_limit]
        for f in files:
            img_cv = cv2.imread(f)
            if img_cv is None:
                continue
            ocr = ocr_extract_with_confidences(img_cv)
            images.append(img_cv)
            texts.append(ocr['text'])
            labels.append(label)
    read_folder(clean_dir, 0)
    read_folder(manipulated_dir, 1)
    return images, texts, labels


def prepare_features(images: List[np.ndarray], texts: List[str], text_encoder: TextEncoder, visual_encoder_model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrai: Xv (visual vectors), Xt (text vectors), Xo (ocr stats)
    """
    # Visual
    Xv = []
    Xo = []
    for img in images:
        v = preprocess_image_for_visual(img)
        feat = visual_encoder_model.predict(np.expand_dims(v, axis=0))[0]
        Xv.append(feat)
        ocr = ocr_extract_with_confidences(img)
        stats = ocr['stats']
        Xo.append([stats['mean_conf'], stats['std_conf'], stats['min_conf']])
    Xv = np.array(Xv)
    Xo = np.array(Xo)

    # Texto (usar TF-IDF)
    Xt = text_encoder.transform(texts)
    return Xv, Xt, Xo


def train_demo(clean_dir: str, manipulated_dir: str, epochs=10):
    """
    Exemplo de uso end-to-end (pequeno) para demonstrar.
    """
    # 1) Carregar dataset
    images, texts, labels = build_dataset_from_folders(clean_dir, manipulated_dir)
    y = np.array(labels)

    # 2) Split
    imgs_train, imgs_test, texts_train, texts_test, y_train, y_test = train_test_split(images, texts, y, test_size=0.2, random_state=SEED, stratify=y)

    # 3) Text encoder fit
    te = TextEncoder(use_tfidf=True, max_tokens=2000)
    Xtext_train = te.fit_transform(texts_train)
    Xtext_test = te.transform(texts_test)

    # 4) Visual encoder
    vis_enc = build_visual_encoder()

    def extract_visual_list(imgs, vis_model):
        feats = []
        stats = []
        for img in imgs:
            v = preprocess_image_for_visual(img)
            feat = vis_model.predict(np.expand_dims(v, axis=0))[0]
            feats.append(feat)
            ocr = ocr_extract_with_confidences(img)
            stats.append([
                ocr['stats']['mean_conf'],
                ocr['stats']['std_conf'],
                ocr['stats']['min_conf']
            ])
        return np.array(feats), np.array(stats)

    Xv_train, Xo_train = extract_visual_list(imgs_train, vis_enc)
    Xv_test, Xo_test = extract_visual_list(imgs_test, vis_enc)

    # 5) Fusão
    fusion = build_fusion_classifier(
        visual_vector_dim=Xv_train.shape[1],
        text_vector_dim=Xtext_train.shape[1],
        ocr_stats_dim=Xo_train.shape[1]
    )

    es = callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    fusion.fit(
        [Xv_train, Xtext_train, Xo_train],
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[es]
    )

    # 6) Avaliação
    metrics, probs, preds = evaluate_model(
        fusion, Xv_test, Xtext_test, Xo_test, y_test
    )

    print("Métricas de teste:", metrics)
    print(classification_report(y_test, preds, digits=4))

    # --- 👇 AQUI está o ajuste importante ---
    return {
        'model': fusion,                # alias para compatibilidade
        'fusion_model': fusion,
        'visual_encoder': vis_enc,
        'text_encoder': te,
        'metrics': metrics
    }


# -------------------------
# FUNÇÃO DE PREDIÇÃO/INFERÊNCIA COM SAÍDAS EXPLICÁVEIS
# -------------------------
def predict_with_explainability(image_path: str, visual_encoder, text_encoder: TextEncoder, fusion_model) -> Dict[str, Any]:
    img_cv = cv2.imread(image_path)
    # OCR
    ocr = ocr_extract_with_confidences(img_cv)
    text = ocr['text']
    ocr_stats = np.array([[ocr['stats']['mean_conf'], ocr['stats']['std_conf'], ocr['stats']['min_conf']]])
    # visual feat
    v = preprocess_image_for_visual(img_cv)
    vfeat = visual_encoder.predict(np.expand_dims(v, axis=0))
    # text vect
    tvec = text_encoder.transform([text])
    # pred
    prob = fusion_model.predict([vfeat, tvec, ocr_stats])[0,0]
    label = int(prob >= 0.5)
    # heatmap
    heatmap = grad_cam_visual_heatmap(visual_encoder, v)
    # palavras com baixa confiança
    low_conf_words = [w for w in ocr['words'] if w['conf'] < 50]  # limiar ajustável
    return {
        'probability_manipulated': float(prob),
        'predicted_label': label,
        'ocr_text': text,
        'ocr_stats': ocr['stats'],
        'low_confidence_words': low_conf_words,
        'gradcam_heatmap': heatmap  # array 2D entre 0 e 1
    }


# -------------------------
# EXEMPLO DE USO (INSTRUÇÕES)
# -------------------------
if __name__ == "__main__":
    print("Este módulo contém funções para treinar/inferir um detector de screenshots adulteradas.")
    print("Modo de uso (exemplo):")
    print("1) Prepare duas pastas: data/clean/ e data/manipulated/ com imagens .jpg/.png")
    print("2) Ajuste o caminho do tesseract se necessário (pytesseract.pytesseract.tesseract_cmd = '...')")
    print("3) Execute: from meme_detector import train_demo; train_demo('data/clean','data/manipulated', epochs=8)")
    print("4) Após treinar, chame predict_with_explainability para analisar imagens individuais.")
