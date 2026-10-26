import streamlit as st
import os
import cloudinary
import cloudinary.uploader
from supabase import create_client, Client
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Pencarian Listing Properti", page_icon="🏠")

# --- KODE CSS KUSTOM UNTUK MEMPERCANTIK TAMPILAN PROPERTIKU ---
st.markdown("""
<style>
    /* 1. Desain Tombol Utama (Warna Khas: Oranye Senja) */
    .stButton>button {
        background-color: #E67E22; /* Oranye lembut */
        color: white;
        border-radius: 8px; /* Sudut melengkung yang modern */
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease; /* Animasi halus saat disentuh */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Efek saat tombol disentuh/di-hover */
    .stButton>button:hover {
        background-color: #D35400; /* Oranye sedikit lebih gelap */
        transform: translateY(-2px); /* Tombol seolah terangkat */
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
    }
    
    /* 2. Desain Kotak Input (Formulir yang lebih ramah mata) */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #BDC3C7;
        padding: 10px;
        transition: all 0.3s;
    }
    /* Efek menyala oranye saat sedang mengetik */
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #E67E22;
        box-shadow: 0 0 5px rgba(230, 126, 34, 0.4);
    }

    /* 3. Desain Kartu Katalog (Expander) agar rapi */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(149, 165, 166, 0.3);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        overflow: hidden;
        background-color: rgba(255, 255, 255, 0.02); /* Transparan agar cocok di mode gelap/terang */
    }

    /* 4. Tampilan Tab yang lebih elegan */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

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

# 3. Menyiapkan Model AI (CLIP)
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
    vector = image_features[0].numpy().tolist()
    return vector

# ==========================================
# 4. SISTEM LOGIN & OTENTIKASI
# ==========================================
if 'user' not in st.session_state:
    st.title("🔐 Gerbang Katalog Properti")
    
    opsi = st.radio("Pilih akses:", ["Login", "Daftar Agen Baru"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit_login = st.button("Eksekusi")
    
    if submit_login:
        if opsi == "Daftar Agen Baru":
            try:
                res = supabase.auth.sign_up({"email": email, "password": password})
                st.success("Akun berhasil dibuat! Silakan pilih opsi 'Login' untuk masuk.")
            except Exception as e:
                st.error(f"Gagal mendaftar: {e}")
        else:
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state['user'] = res.user
                st.rerun()
            except Exception as e:
                st.error("Login gagal. Periksa kembali email dan password Anda.")
    
    st.stop() # Hentikan kode di bawah ini jika belum login

# ==========================================
# 5. TAMPILAN UTAMA APLIKASI
# ==========================================
# Sidebar untuk info agen
st.sidebar.write(f"Masuk sebagai: **{st.session_state['user'].email}**")
if st.sidebar.button("Logout"):
    supabase.auth.sign_out()
    st.session_state.pop('user', None)
    st.rerun()

st.title("🏠 Manajemen Listing Properti AI")

# Membuat menu dengan 4 tab (Tambahan Katalog)
tab1, tab2, tab3, tab4 = st.tabs(["➕ Tambah Listing", "🔍 Cari Foto (AI)", "🔍 Cari Teks", "📋 Katalog Saya"])

# --- TAB 1: TAMBAH LISTING ---
with tab1:
    st.header("Masukkan Properti Baru")
    with st.form("tambah_properti", clear_on_submit=True):
        owner_name = st.text_input("Nama Pemilik (Contoh: Bapak Budi)")
        owner_phone = st.text_input("Nomor HP Pemilik")
        description = st.text_area("Deskripsi Singkat (Alamat, Harga, Fasilitas)")
        # Info kapasitas maksimal diubah menjadi 30
        uploaded_images = st.file_uploader("Pilih Foto Rumah (Maksimal 30 foto)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        submit_button = st.form_submit_button("Simpan Properti")

        if submit_button:
            if not owner_name or not uploaded_images:
                st.error("Nama pemilik dan minimal 1 foto wajib diisi!")
            elif len(uploaded_images) > 30:
                st.error("Maksimal unggah adalah 30 foto sekaligus. Silakan kurangi foto Anda.")
            else:
                with st.spinner('Sedang mengunggah dan memproses AI...'):
                    try:
                        # 1. Simpan data teks + ID Agen (user_id) ke Supabase
                        prop_data = {
                            "owner_name": owner_name,
                            "owner_phone": owner_phone,
                            "description": description,
                            "user_id": st.session_state['user'].id # LABEL KEPEMILIKAN
                        }
                        prop_res = supabase.table("properties").insert(prop_data).execute()
                        property_id = prop_res.data[0]['id']

                        # 2. Proses dan unggah setiap foto
                        for img_file in uploaded_images:
                            img = Image.open(img_file).convert('RGB')
                            img_file.seek(0) 

                            # A. Unggah ke Cloudinary
                            upload_result = cloudinary.uploader.upload(img_file)
                            image_url = upload_result['secure_url']

                            # B. Vektor AI
                            vektor_mentah = get_image_vector(img)
                            if isinstance(vektor_mentah[0], list):
                                if isinstance(vektor_mentah[0][0], list):
                                    img_vector = vektor_mentah[0][0]
                                else:
                                    img_vector = vektor_mentah[0]
                            else:
                                img_vector = vektor_mentah

                            # C. Simpan ke Supabase
                            img_data = {
                                "property_id": property_id,
                                "image_url": image_url,
                                "image_vector": img_vector
                            }
                            supabase.table("property_images").insert(img_data).execute()

                        st.success("Listing berhasil ditambahkan!")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {e}")

# --- TAB 2: PENCARIAN FOTO ---
with tab2:
    st.header("Cari Listing Menggunakan Foto")
    search_image = st.file_uploader("Unggah foto rumah dari pembeli", type=['jpg', 'jpeg', 'png'])

    if search_image:
        img_to_search = Image.open(search_image).convert('RGB')
        st.image(img_to_search, caption="Foto yang dicari", width=250)

        if st.button("Mulai Pencarian AI"):
            with st.spinner('AI sedang mencocokkan kemiripan...'):
                vektor_mentah = get_image_vector(img_to_search)
                if isinstance(vektor_mentah[0], list):
                    if isinstance(vektor_mentah[0][0], list):
                        query_vector = vektor_mentah[0][0] 
                    else:
                        query_vector = vektor_mentah[0]    
                else:
                    query_vector = vektor_mentah

                try:
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
                     st.error(f"Error pencarian: {e}")

# --- TAB 3: PENCARIAN TEKS ---
with tab3:
    st.header("Cari Berdasarkan Teks")
    search_query = st.text_input("Ketik nama pemilik, no HP, atau deskripsi (Cth: Budi, 0812, Jakarta)")

    if st.button("Cari Listing"):
        if search_query:
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

# --- TAB 4: KATALOG & HAPUS LISTING ---
with tab4:
    st.header("Katalog Properti Anda")
    # Hanya menarik data properti milik agen yang sedang login
    res_katalog = supabase.table("properties").select("*").eq("user_id", st.session_state['user'].id).execute()
    
    if res_katalog.data:
        for prop in res_katalog.data:
            with st.expander(f"🏠 {prop['owner_name']} - {prop['owner_phone']}"):
                st.write(f"**Deskripsi:** {prop['description']}")
                
                # Tombol hapus dengan Key unik berdasarkan ID properti
                if st.button("❌ Hapus Listing Ini", key=f"del_{prop['id']}"):
                    try:
                        # Hapus foto-fotonya dulu, baru hapus propertinya
                        supabase.table("property_images").delete().eq("property_id", prop['id']).execute()
                        supabase.table("properties").delete().eq("id", prop['id']).execute()
                        st.success("Listing berhasil dihapus!")
                        st.rerun() # Refresh halaman agar data yang dihapus hilang dari layar
                    except Exception as e:
                        st.error(f"Gagal menghapus: {e}")
    else:
        st.info("Anda belum memiliki listing properti.")