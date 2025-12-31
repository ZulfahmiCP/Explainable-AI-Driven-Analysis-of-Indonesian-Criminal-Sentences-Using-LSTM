import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import altair as alt
import traceback
import nltk
import re
import pdfplumber  # Pastikan sudah pip install pdfplumber
from keras.src.utils.sequence_utils import pad_sequences
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords as nltk_stopwords

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA WARNA
# ==========================================
st.set_page_config(
    page_title="LegalLens AI - Verdict Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palet Warna (Navy & Maroon) - Professional Legal Look
COLOR_NAVY = "#2C3E50"
COLOR_MAROON = "#C0392B"
COLOR_GREEN = "#27AE60"
COLOR_BG = "#0e1117"

# Custom CSS untuk UI/UX
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {COLOR_BG};
    }}
    /* Header Styling */
    h1, h2, h3 {{
        color: {COLOR_NAVY};
        font-family: 'Helvetica', sans-serif;
    }}
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_NAVY};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 2rem;
    }}
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: white !important;
    }}
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {{
        color: {COLOR_MAROON};
        font-size: 3rem !important;
        font-weight: 800;
    }}
    /* Custom Button */
    .stButton>button {{
        background-color: {COLOR_MAROON};
        color: white;
        border-radius: 8px;
        height: 50px;
        width: 100%;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #A93226;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    /* Highlight Box for Text */
    .highlight-box {{
        padding: 20px;
        border-radius: 8px;
        background-color: white;
        border-left: 5px solid {COLOR_NAVY};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-family: 'Georgia', serif;
        line-height: 1.6;
        max-height: 500px;
        overflow-y: auto;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI PREPROCESSING (SINKRON DENGAN NOTEBOOK)
# ==========================================
def cleaning_text(text):
    if pd.isna(text): return ""
    text = str(text)

    # --- CLEANING PDF ARTIFACTS (TAMBAHAN) ---
    # Hapus kata "maa" yang berulang (akibat watermark Mahkamah Agung)
    text = re.sub(r'\b(maa\s*)+\b', '', text, flags=re.IGNORECASE)
    # Hapus karakter aneh sisa konversi
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Hapus Disclaimer Mahkamah Agung (muncul di setiap halaman)
    # Contoh: "Disclaimer... fungsi peradilan."
    text = re.sub(r'Disclaimer.*?fungsi peradilan\.', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Hapus Info Kontak (muncul di footer)
    # Contoh: "Dalam hal Anda menemukan... (ext.318)"
    text = re.sub(r'Dalam hal Anda menemukan.*?\(ext\.\d+\)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Hapus Header Direktori Putusan
    # Contoh: "Direktori Putusan Mahkamah Agung Republik Indonesia"
    text = re.sub(r'Direktori Putusan Mahkamah Agung Republik Indonesia', '', text, flags=re.IGNORECASE)
    text = re.sub(r'hkama.*?Indonesi', '', text, flags=re.DOTALL | re.IGNORECASE) # Pola artifak OCR

    # Hapus Penanda Halaman
    # Contoh: "Halaman 1 dari 21 Putusan Nomor..."
    text = re.sub(r'Halaman \d+ dari \d+.*?(Putusan Nomor|halaman)', '', text, flags=re.IGNORECASE)

    # Lowercase
    text = text.lower()
    
    # Hapus karakter spesial/sisa OCR 
    text = re.sub(r'[-=]{3,}', ' ', text) 
    
    # Hapus spasi berlebih dan newline 
    text = re.sub(r'\s+', ' ', text).strip()

    # Hapus nominal rupiah
    # Contoh: "Rp231.076.000,00" atau "Rp. 5.000,-" atau "Rp 1.000.000"
    text = re.sub(r'rp[\s\.]*[\d\.,]+(?:,-)?', '', text)
    
    return text

legal_entities_count = {
    'pasal': 0,
    'meringankan': 0,
    'memberatkan': 0
}

def extract_legal_entities(text):
    tokens = []

    # EKSTRAKSI PASAL 
    # Pola Pasal X Ayat Y
    pasal_matches = re.findall(r'pasal\s+(\d+)(?:\s+ayat\s*\(?(\d+)\)?)?', text)
    for p in pasal_matches:
        if p[1]: # Ada ayat
            tokens.append(f"fitur_pasal{p[0]}ayat{p[1]}")
        else:
            tokens.append(f"fitur_pasal{p[0]}")
        legal_entities_count['pasal'] += 1
            
    # EKSTRAKSI KEADAAN MERINGANKAN
    # Faktor-faktor yang mengundang simpati hakim atau menunjukkan itikad baik terdakwa untuk memperbaiki diri
    block = re.search(r'(keadaan|hal-hal)\s+yang\s+meringankan.*?(menimbang|mengingat|memperhatikan)', text, flags=re.DOTALL)
    if block:
        content = block.group(0)
        keywords = {
            'sopan': 'fitur_sikap_sopan',
            'menyesal': 'fitur_menyesal',
            'tulang punggung': 'fitur_tulang_punggung',
            'tanggungan': 'fitur_tulang_punggung', 
            'belum pernah': 'fitur_belum_dihukum',
            'mengakui': 'fitur_mengakui_perbuatan',
            'terus terang': 'fitur_mengakui_perbuatan', 
            'muda': 'fitur_usia_muda',
            'masih sekolah': 'fitur_usia_muda',
            'maaf': 'fitur_sudah_dimaafkan', 
            'damai': 'fitur_sudah_dimaafkan'
        }
        
        for key, token in keywords.items():
            if key in content:
                tokens.append(token)
                legal_entities_count['meringankan'] += 1

    # EKSTRAKSI KEADAAN MEMBERATKAN
    # Faktor-faktor yang dianggap memperparah tindak pidana atau menunjukkan itikad buruk terdakwa
    block = re.search(r'(keadaan|hal-hal)\s+yang\s+memberatkan.*?(keadaan|hal-hal)\s+yang\s+meringankan', text, flags=re.DOTALL)
    if block:
        content = block.group(0)
        
        keywords = {
            'meresahkan': 'fitur_meresahkan_masyarakat',
            'berbelit': 'fitur_berbelit_belit',
            'tidak mengakui': 'fitur_tidak_mengakui',
            'luka berat': 'fitur_luka_berat',
            'cacat': 'fitur_luka_berat',
            'meninggal': 'fitur_korban_mati',
            'nyawa': 'fitur_korban_mati',
            'hilang': 'fitur_korban_mati', # Menghilangkan nyawa
            'pernah dihukum': 'fitur_residivis',
            'residivis': 'fitur_residivis',
            'mengulangi': 'fitur_residivis',
            'sadis': 'fitur_perbuatan_keji',
            'kejam': 'fitur_perbuatan_keji',
            'tidak manusiawi': 'fitur_perbuatan_keji',
            'generasi muda': 'fitur_merusak_generasi', # Spesifik Narkoba
            'program pemerintah': 'fitur_anti_program_pemerintah', # Spesifik Narkoba
            'aparat': 'fitur_aparat_penegak_hukum',
            'polisi': 'fitur_aparat_penegak_hukum',
            'kerugian': 'fitur_kerugian_besar' # Jika kata kerugian muncul di memberatkan
        }
        
        for key, token in keywords.items():
            if key in content:
                tokens.append(token)
                legal_entities_count['memberatkan'] += 1

    return ' '.join(set(tokens))

def extract_case_narrative(text):
    # Berhenti membaca sebelum hakim menjatuhkan vonis untuk mencegah leakage

    stop_markers = ["mengadili:", "mengadili", "memutuskan:", "memutuskan", "menetapkan:", "menetapkan"]
    
    stop_index = len(text)
    for word in stop_markers:
        idx = text.find(word)
        if idx != -1 and idx < stop_index:
            stop_index = idx
        
    text = text[:stop_index]
    
    # Bagian paling penting untuk prediksi biasanya ada di "Duduk Perkara" atau "Menimbang".
    # Bagian awal (Identitas) kurang berguna.
    
    start_markers = [
        "dakwaan", "menimbang", "duduk perkara", "tentang hukumnya",     
        "menuntut", "tuntutan", "didakwa", "mendengar keterangan", 
        "memperhatikan", "fakta-fakta hukum", "fakta fakta hukum",    
    ]
    
    start_index = 0
    found_indices = []
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            found_indices.append(idx)
            
    if found_indices:
        first_marker_idx = min(found_indices)
        
        pre_text = text[:first_marker_idx]
        pre_words = pre_text.split()
        if len(pre_words) > 10:
            # Ambil 10 kata terakhir lalu gabung kembali 
            buffer_text = " ".join(pre_words[-10:]) + " " 
        else:
            buffer_text = pre_text 
            
        text = buffer_text + text[first_marker_idx:]
    
    text = text[start_index:]

    return text

nltk.download('stopwords')
list_stopwords = set(nltk_stopwords.words('indonesian'))

# Domain Specific Stopwords
legal_stopwords = {
    'terdakwa', 'saksi', 'hakim', 'penuntut', 'pengadilan', 'negeri', 'persidangan', 
    'diduga', 'memiliki', 'menguasai', 'barang', 'bukti', 'dirampas', 'dimusnahkan', 
    'dikembalikan', 'biaya', 'perkara', 'terlampir', 'berkas', 'keterangan', 'direktori',
    'pendapat', 'kepaniteraan', 'mahkamah', 'agung', 'republik', 'indonesia', 'saksi',
}

final_stopwords = set(list_stopwords) | set(legal_stopwords)

def normalize_text(text):
    if pd.isna(text) or text == "":
        return ""
    
    # PUNCTUATION & NUMBER HANDLING -> Menghapus apapun yang bukan huruf, angka, spasi
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Hapus angka yang berdiri sendiri, biasanya itu tanggal, nomor halaman, atau jumlah uang 
    text = re.sub(r'\b\d+\b', '', text)
    
    # TOKENIZATION 
    tokens = text.split()
    tokens = [token for token in tokens if (token not in final_stopwords and len(token) > 2)]
    
    text = ' '.join(tokens)
    
    return text

def process_full_pipeline(raw_text):
    clean_full = cleaning_text(raw_text)
    features = extract_legal_entities(clean_full)
    body = extract_case_narrative(clean_full)
    
    # Injection Strategy (Fitur diulang 2x untuk boosting)
    final_text = f"{features} {features} {body}"
    final_text = normalize_text(final_text)
    return final_text

# ==========================================
# 3. LOAD ASSETS (Model Standar Keras)
# ==========================================
@st.cache_resource
def load_assets():
    try:
        # Load Model Biasa (.h5) tanpa custom object
        model = tf.keras.models.load_model('model_sentence_prediction_lstm.h5', compile=False)
        
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model/tokenizer. Pastikan file ada di folder yang sama.\nError: {e}")
        return None, None

model, tokenizer = load_assets()
MAX_LEN = 300 

# ==========================================
# 4. PREDIKSI & WRAPPER LIME (DIPERBAIKI)
# ==========================================

# def predict_fn_lime(texts):
#     """
#     Wrapper fungsi prediksi yang mengikuti logika notebook:
#     Tokenize -> Pad -> Predict -> Inverse Log (Expm1)
#     """
#     # 1. Pastikan input berupa List (Sesuai kebutuhan tokenizer)
#     if isinstance(texts, str):
#         texts = [texts]
        
#     # 2. Tokenizing & Padding (Sesuai Notebook)
#     seqs = tokenizer.texts_to_sequences(texts)
#     padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    
#     # 3. Predict & Inverse Log (Sesuai Notebook)
#     preds = model.predict(padded)
#     preds = np.expm1(preds) 
    
#     # 4. FIX UTAMA: Flattening untuk LIME
#     # Notebook return: [[12.5]] (2D)
#     # LIME Streamlit butuh: [12.5] (1D)
#     # Jika tidak di-flatten, LIME akan error saat mencoba slicing index tuple
#     return preds.flatten()

def predict_fn_lime(texts):
    # Wrapper fungsi prediksi agar kompatibel dengan LIME
    # 1. Pastikan input list
    if isinstance(texts, str):
        texts = [texts]
        
    # 2. Tokenize & Pad
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict
    preds = model.predict(padded)
    
    # Handling jika output model berupa list (misal ada attention)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
        
    # 4. FIX: JANGAN FLATTEN.
    # LIME butuh bentuk (N, 1) agar bisa membaca shape-nya sebagai "1 kelas output".
    # Kita hanya lakukan inverse log.
    return np.expm1(preds)

def generate_highlighted_text(text, exp_list):
    """Membuat HTML teks dengan highlight warna berdasarkan bobot LIME"""
    word_colors = {}
    
    # Normalisasi bobot untuk opacity warna
    max_weight = max([abs(x[1]) for x in exp_list]) if exp_list else 1.0
    
    for word, weight in exp_list:
        # Normalisasi opacity (0.1 - 0.6)
        opacity = (abs(weight) / max_weight) * 0.5 + 0.1
        
        if weight > 0: # Memberatkan (Merah)
            word_colors[word] = f"rgba(192, 57, 43, {opacity})" 
        else: # Meringankan (Hijau)
            word_colors[word] = f"rgba(39, 174, 96, {opacity})"

    html_parts = []
    # Batasi tampilan teks agar tidak terlalu panjang di UI
    display_words = text.split()[:300] 
    
    for word in display_words:
        clean_w = re.sub(r'[^\w]', '', word).lower()
        if clean_w in word_colors:
            color = word_colors[clean_w]
            # Tambahkan tooltip bobot
            html_parts.append(f'<span style="background-color: {color}; padding: 0 4px; border-radius: 3px; font-weight: 500;" title="Bobot: {weight:.2f}">{word}</span>')
        else:
            html_parts.append(word)
    
    return " ".join(html_parts) + " ... (teks dipotong untuk tampilan)"

# ==========================================
# 5. USER INTERFACE (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title("🏛️ LegalLens AI")
    st.markdown("### *AI-Powered Criminal Verdict Predictor*")
    st.markdown("---")
    
    input_mode = st.radio("Pilih Metode Input:", 
                          ["Upload File PDF", "Input Teks Manual", "Demo (Contoh Kasus)"])
    
    st.info("""
    **Tentang Model:**
    - **Engine:** Bi-Directional LSTM
    - **Dataset:** Putusan Pidana Indonesia
    - **Fitur:** Auto-Extraction (Pasal, Tuntutan, dll)
    """)
    
    st.markdown("---")
    st.caption("Dibuat untuk Portofolio Data Science")

# ==========================================
# 6. USER INTERFACE (MAIN)
# ==========================================
st.title("Analisis Prediksi Vonis Pidana")
st.markdown("""
Sistem ini menggunakan **Deep Learning (LSTM)** untuk membaca dokumen putusan, mengekstrak fakta hukum, 
dan memprediksi lama hukuman penjara. Dilengkapi dengan **Explainable AI** untuk transparansi.
""")

raw_text = ""

# --- LOGIKA INPUT DATA ---

# --- LOGIKA INPUT DATA (DIPERBAIKI) ---
if input_mode == "Upload File PDF":
    uploaded_file = st.file_uploader("Upload Dokumen Putusan (.pdf)", type="pdf")
    if uploaded_file is not None:
        try:
            # Menggunakan pdfplumber untuk hasil lebih rapi
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = []
                for page in pdf.pages:
                    # Extract text dengan layout preservation
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    
                    # --- CLEANING SPESIFIK UNTUK PDF KOTOR ---
                    lines = page_text.split('\n')
                    clean_lines = []
                    for line in lines:
                        # 1. Hapus baris yang isinya cuma "maa maa" atau noise pendek
                        if len(line.strip()) < 4 or "maa maa" in line.lower():
                            continue
                        
                        # 2. Hapus Header Direktori Putusan yang sering muncul
                        if "direktori putusan" in line.lower() or "mahkamah agung" in line.lower():
                            continue
                            
                        clean_lines.append(line)
                    
                    full_text.append("\n".join(clean_lines))
                
                raw_text = "\n".join(full_text)
                
            st.success(f"✅ Dokumen berhasil dibaca ({len(raw_text)} karakter). Siap dianalisis.")
            
            # Tampilkan sedikit preview agar user yakin teksnya benar
            with st.expander("Cek Kualitas Teks (Preview Awal)"):
                st.text(raw_text[:500] + "...")
                
        except Exception as e:
            st.error(f"Gagal membaca PDF: {e}")

elif input_mode == "Input Teks Manual":
    st.info("Silakan copy-paste teks putusan (Dakwaan, Menimbang, Mengadili) ke kolom di bawah.")
    raw_text = st.text_area("Tempel Teks Putusan:", height=300, placeholder="Contoh: DAKWAAN: Bahwa terdakwa... MENIMBANG: ...")

elif input_mode == "Demo (Contoh Kasus)":
    case_option = st.selectbox("Pilih Skenario Demo:", [
        "Kasus 1: Pencurian (Pasal 362) - Ringan",
        "Kasus 2: Narkotika (Pasal 112) - Sedang",
        "Kasus 3: Pembunuhan (Pasal 338) - Berat"
    ])
    
    if "Kasus 1" in case_option:
        raw_text = """
        DAKWAAN: Bahwa terdakwa Budi pada hari Senin mengambil dompet korban di pasar. 
        Tuntutan Jaksa Penuntut Umum: Menuntut pidana penjara selama 1 tahun.
        MENIMBANG: Terdakwa bersikap sopan, mengakui perbuatannya, dan belum pernah dihukum. 
        Terdakwa adalah tulang punggung keluarga. Kerugian korban Rp 500.000.
        Memperhatikan Pasal 362 KUHP tentang Pencurian.
        """
    elif "Kasus 2" in case_option:
        raw_text = """
        DAKWAAN: Terdakwa ditangkap membawa 1 paket sabu seberat 0.5 gram. 
        Tuntutan: Menuntut pidana penjara selama 5 tahun.
        MENIMBANG: Perbuatan terdakwa meresahkan masyarakat dan tidak mendukung program pemerintah.
        Terdakwa berbelit-belit di persidangan. Melanggar Pasal 112 UU Narkotika.
        """
    else:
        raw_text = """
        DAKWAAN: Terdakwa dengan sengaja menghilangkan nyawa orang lain dengan menusuk korban menggunakan pisau.
        Tuntutan: Menuntut pidana penjara selama 15 tahun.
        MENIMBANG: Perbuatan terdakwa tergolong sadis dan kejam. Korban meninggal dunia. 
        Keluarga korban tidak memaafkan. Terdakwa merupakan residivis kasus serupa.
        Memperhatikan Pasal 338 KUHP.
        """
    
    st.text_area("Preview Teks Putusan:", value=raw_text, height=150, disabled=True)

# --- TOMBOL EKSEKUSI ---
if st.button("Analisis Dokumen", type="primary"):
    if not raw_text:
        st.warning("⚠️ Silakan masukkan teks atau upload PDF terlebih dahulu.")
    elif model is None:
        st.error("⚠️ Model belum dimuat. Cek file .h5 dan .pickle.")
    else:
        # PROGRESS BAR
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Cleaning
            status_text.text("1/3 Membersihkan dokumen & ekstraksi fitur hukum...")
            clean_text = process_full_pipeline(raw_text)
            progress_bar.progress(40)
            
            # 2. Extract Features for Display (Manual Check)
            raw_clean = cleaning_text(raw_text)
            extracted_features = extract_legal_entities(raw_clean).split()
            
            # 3. Prediction
            status_text.text("2/3 Menjalankan model LSTM...")
            
            # Panggil fungsi prediksi
            # Karena outputnya sekarang (N, 1), kita ambil [0][0] untuk nilai skalar
            prediction_array = predict_fn_lime([clean_text])
            
            if len(prediction_array) > 0:
                pred_raw = prediction_array[0] # Ini masih float (misal 2.5 bln)
                
                # Handling jika model memprediksi negatif atau nol
                if pred_raw < 0.1: 
                    pred_raw = 0.1 # Minimalisir agar tidak 0 bulat
                
                pred_bulan = pred_raw
                pred_tahun = pred_bulan / 12
                
                # DEBUG (Hapus nanti): Lihat nilai asli prediksi sebelum formatting
                # st.write(f"Raw Prediction: {pred_raw}") 
            else:
                st.error("Model tidak menghasilkan output.")
                st.stop()
                
            progress_bar.progress(80)
            
            # 4. Explainability (LIME)
            status_text.text("3/3 Menghasilkan penjelasan (Explainable AI)...")
            
            # Inisialisasi LIME dengan class_names agar tidak bingung
            # Kita sebut saja outputnya "Vonis"
            explainer = LimeTextExplainer(class_names=['Vonis']) 
            
            # Generate Explanation
            # labels=(0,) artinya kita mau menjelaskan kolom index ke-0 (output prediksi)
            exp = explainer.explain_instance(
                clean_text, 
                predict_fn_lime, 
                num_features=10,
                labels=(0,)
            )
            
            # Ambil list bobot untuk label 0
            exp_list = exp.as_list(label=0)
            
            progress_bar.progress(100)
            status_text.empty() 
            
            # --- TAMPILAN HASIL (DASHBOARD) ---
            # (Kode tampilan ke bawah SAMA seperti sebelumnya)
            st.markdown("---")
            
            # A. KARTU RINGKASAN
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 📜 Pasal & Tuntutan")
                pasal_list = [f.replace('fitur_', '').upper() for f in extracted_features if 'pasal' in f or 'tuntutan' in f]
                if pasal_list:
                    for p in pasal_list: st.caption(f"• {p}")
                else: st.caption("- Tidak terdeteksi spesifik -")
                
            with col2:
                st.markdown("##### ⚖️ Faktor Kunci")
                faktor_list = [f.replace('fitur_', '').replace('_', ' ').title() for f in extracted_features if 'pasal' not in f and 'tuntutan' not in f]
                if faktor_list:
                    for f in faktor_list: 
                        color = "red" if "Meresahkan" in f or "Residivis" in f else "green"
                        st.markdown(f"<span style='color:{color};'>• {f}</span>", unsafe_allow_html=True)
                else: st.caption("- Netral -")
            
            with col3:
                st.markdown("##### ⏱️ Prediksi Vonis")
                st.metric(label="", value=f"{pred_bulan:.1f} Bulan", delta=f"± {pred_tahun:.1f} Tahun")

            # B. EXPLAINABILITY TABS
            st.subheader("🔍 Mengapa AI Memprediksi Ini?")
            tab1, tab2 = st.tabs(["📊 Grafik Pengaruh (LIME)", "📝 Highlight Teks"])
            
            with tab1:
                if exp_list:
                    df_exp = pd.DataFrame(exp_list, columns=['Kata Kunci', 'Bobot'])
                    df_exp['Efek'] = df_exp['Bobot'].apply(lambda x: 'Menambah Hukuman (+)' if x > 0 else 'Mengurangi Hukuman (-)')
                    df_exp = df_exp.sort_values(by='Bobot', ascending=False)
                    
                    chart = alt.Chart(df_exp).mark_bar().encode(
                        x=alt.X('Bobot', title='Kekuatan Pengaruh (Bulan)'),
                        y=alt.Y('Kata Kunci', sort=None), 
                        color=alt.Color('Efek', scale=alt.Scale(domain=['Menambah Hukuman (+)', 'Mengurangi Hukuman (-)'], range=[COLOR_MAROON, COLOR_GREEN])),
                        tooltip=['Kata Kunci', 'Bobot', 'Efek']
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Tidak ada fitur yang cukup signifikan untuk ditampilkan.")
                
            with tab2:
                st.info("Warna **MERAH** = Kata yang memberatkan. Warna **HIJAU** = Kata yang meringankan.")
                html_view = generate_highlighted_text(clean_text, exp_list)
                st.markdown(f'<div class="highlight-box">{html_view}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat proses: {e}")
            with st.expander("Detail Error"):
                st.write(e)
                st.code(traceback.format_exc())