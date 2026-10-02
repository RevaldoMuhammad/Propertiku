import streamlit as st
import os
import cloudinary
import cloudinary.uploader
from supabase import create_client, Client
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

# 1. Mengambil kunci akses dari file .env
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

# 2. Menghubungkan ke Supabase dan Cloudinary
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# 3. Menyiapkan Model AI (CLIP) untuk membaca gambar
# St.cache_resource digunakan agar AI tidak dimuat ulang setiap kali tombol ditekan
@st.cache_resource
def load_ai_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_ai_model()

def get_image_vector(image):
    """Fungsi untuk mengubah gambar menjadi deretan angka (vektor)"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Mengubah format angka agar bisa disimpan di Supabase
    vector = image_features[0].numpy().tolist()
    return vector

# --- TAMPILAN APLIKASI (STREAMLIT) ---
st.set_page_config(page_title="Sistem Pencarian Listing Properti", page_icon="🏠")
st.title("🏠 Manajemen Listing Properti AI")

# Membuat menu dengan tab
tab1, tab2, tab3 = st.tabs(["Tambah Listing Baru", "Cari dengan Foto (AI)", "Cari dengan Teks"])

# --- TAB 1: TAMBAH LISTING ---
with tab1:
    st.header("Masukkan Properti Baru")
    with st.form("tambah_properti", clear_on_submit=True):
        owner_name = st.text_input("Nama Pemilik (Contoh: Bapak Budi)")
        owner_phone = st.text_input("Nomor HP Pemilik")
        description = st.text_area("Deskripsi Singkat (Alamat, Harga, Fasilitas)")
        uploaded_images = st.file_uploader("Pilih Foto Rumah (Bisa lebih dari satu)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        submit_button = st.form_submit_button("Simpan Properti")

        if submit_button:
            if not owner_name or not uploaded_images:
                st.error("Nama pemilik dan minimal 1 foto wajib diisi!")
            else:
                with st.spinner('Sedang mengunggah dan memproses AI...'):
                    try:
                        # 1. Simpan data teks ke Supabase terlebih dahulu
                        prop_data = {
                            "owner_name": owner_name,
                            "owner_phone": owner_phone,
                            "description": description
                        }
                        prop_res = supabase.table("properties").insert(prop_data).execute()
                        property_id = prop_res.data[0]['id']

                        # 2. Proses dan unggah setiap foto
                        for img_file in uploaded_images:
                            img = Image.open(img_file).convert('RGB')

                            img_file.seek(0) # <--- TAMBAHKAN BARIS INI

                            # A. Unggah foto asli ke Cloudinary untuk mendapatkan link
                            upload_result = cloudinary.uploader.upload(img_file)
                            image_url = upload_result['secure_url']

                            # B. Minta AI mengubah foto menjadi vektor
                            vektor_mentah = get_image_vector(img)
                            
                            # --- BARIS PERBAIKAN ---
                            # Kita bongkar lapisan kurung sikunya agar jadi 1 dimensi (1D)
                            if isinstance(vektor_mentah[0], list):
                                if isinstance(vektor_mentah[0][0], list):
                                    img_vector = vektor_mentah[0][0] # Jika berlapis 3 (3D)
                                else:
                                    img_vector = vektor_mentah[0]    # Jika berlapis 2 (2D)
                            else:
                                img_vector = vektor_mentah           # Jika sudah benar 1D
                            # -----------------------

                            # C. Simpan link foto dan vektor ke Supabase
                            img_data = {
                                "property_id": property_id,
                                "image_url": image_url,
                                "image_vector": img_vector
                            }
                            supabase.table("property_images").insert(img_data).execute()

                        st.success("Listing berhasil ditambahkan!")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {e}")

# --- TAB 2: PENCARIAN DENGAN FOTO (REVERSE IMAGE SEARCH) ---
with tab2:
    st.header("Cari Listing Menggunakan Foto")
    search_image = st.file_uploader("Unggah foto rumah dari pembeli", type=['jpg', 'jpeg', 'png'])

    if search_image:
        img_to_search = Image.open(search_image).convert('RGB')
        st.image(img_to_search, caption="Foto yang dicari", width=250)

        if st.button("Mulai Pencarian AI"):
            with st.spinner('AI sedang mencocokkan kemiripan...'):
                # 1. Ubah foto pencarian menjadi vektor
                vektor_mentah = get_image_vector(img_to_search)

                # Ratakan kurung siku menjadi 1 dimensi
                if isinstance(vektor_mentah[0], list):
                    if isinstance(vektor_mentah[0][0], list):
                        query_vector = vektor_mentah[0][0] 
                    else:
                        query_vector = vektor_mentah[0]    
                else:
                    query_vector = vektor_mentah

                # 2. Cari di database (menggunakan fitur bawaan pgvector: <-> untuk kemiripan kosinus)
                # Kita urutkan dari yang jarak vektornya paling kecil (paling mirip)
                try:
                    # Ini menggunakan perintah RPC (Remote Procedure Call). 
                    # Kita akan membuat fungsinya sebentar lagi di Supabase.
                    response = supabase.rpc(
                        'match_images', 
                        {'query_embedding': query_vector, 'match_threshold': 0.7, 'match_count': 3}
                    ).execute()

                    if response.data:
                        st.success("Ditemukan properti yang mirip!")
                        for match in response.data:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(match['image_url'], width=150)
                            with col2:
                                st.write(f"**Pemilik:** {match['owner_name']}")
                                st.write(f"**Kontak:** {match['owner_phone']}")
                                st.write(f"**Deskripsi:** {match['description']}")
                                st.write(f"*Tingkat Kecocokan: {int(match['similarity'] * 100)}%*")
                            st.divider()
                    else:
                        st.warning("Tidak ada rumah yang mirip di database.")
                except Exception as e:
                     st.error(f"Anda perlu menjalankan kode SQL Fungsi Pencarian (RPC) di Supabase. Lihat panduan selanjutnya.")


# --- TAB 3: PENCARIAN TEKS BIASA ---
with tab3:
    st.header("Cari Berdasarkan Teks")
    search_query = st.text_input("Ketik nama pemilik, no HP, atau deskripsi (Cth: Budi, 0812, Jakarta)")

    if st.button("Cari Listing"):
        if search_query:
            # Menggunakan fungsi 'ilike' (case-insensitive) untuk mencari di 3 kolom sekaligus
            res = supabase.table("properties").select("*").or_(
                f"owner_name.ilike.%{search_query}%,owner_phone.ilike.%{search_query}%,description.ilike.%{search_query}%"
            ).execute()

            if res.data:
                for prop in res.data:
                    st.subheader(f"🏠 {prop['owner_name']}")
                    st.write(f"📞 {prop['owner_phone']}")
                    st.write(f"📝 {prop['description']}")
                    st.divider()
            else:
                st.info("Listing tidak ditemukan.")