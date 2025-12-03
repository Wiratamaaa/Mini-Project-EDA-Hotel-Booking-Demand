import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Dashboard Hotel Booking Demand",
    layout="wide"
)

# Cleaning Data
@st.cache_data
def load_raw_data():
    df = pd.read_csv("data_hotel_booking_demand.csv")
    return df

# Cleaning Data
@st.cache_data
def clean_data(df_input):
    df_clean = df_input.copy()
    
    if 'country' in df_clean.columns:
        df_clean['country'].fillna('Unknown', inplace=True)
    return df_clean

try:
    df_raw = load_raw_data()       
    df_eda = clean_data(df_raw)    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

with st.sidebar:
    st.title("Navigation") 

    selected = option_menu(
        menu_title=None,  
        options=["Business Understanding", "Data Overview", "Exploratory Data Analysis"], 
        default_index=0,  
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#F5EFE6",  
                "font-family": "sans-serif",
                "border-radius": "10px"
            },
            "icon": {"display": "none"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin":"0px",
                "color": "#624235",              
                "background-color": "#E8DCC8",    
                "--hover-color": "#D6B594",       
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#A67B5B",   
                "color": "white",
                "font-weight": "normal",
                "border-radius": "8px"
            },
        }
    )

if selected == "Business Understanding":
    st.title("Hotel Booking Demand Prediction")
    st.subheader("Business Problem")
    st.write(
        """
        Sebuah hotel yang berlokasi di Portugal menghadapi tingginya jumlah 
        pembatalan pemesanan kamar. Banyak pelanggan yang melakukan booking namun 
        membatalkannya sebelum hari kedatangan. Kondisi ini menyebabkan hotel 
        kehilangan potensi pendapatan, kesulitan dalam memaksimalkan tingkat hunian 
        kamar, serta mengganggu perencanaan operasional seperti alokasi kamar dan 
        penjadwalan staf. 
        
        Sehingga untuk mengatasi masalah tersebut, manajemen hotel ingin 
        mengetahui dan memprediksi lebih awal apakah sebuah pelanggan yang telah 
        melakukan booking berpotensi dibatalkan atau tidak. Informasi terkait negara asal, 
        tipe deposit, tipe kamar, kebutuhan tempat parkir mobil dan permintaan spesial ada di 
        tangan dari pelanggan hotel.
        """
    )

    st.subheader("Problem Statement")
    st.write(""" 
        Tingkat pembatalan pemesanan kamar hotel dapat menimbulkan kerugian bagi 
        pihak hotel karena kamar yang dibatalkan sering kali tidak dapat dijual kembali dalam 
        waktu terbatas. Proses pengelolaan reservasi menjadi tidak efisien apabila hotel 
        memperlakukan semua pemesanan dengan tingkat prioritas yang sama tanpa 
        mengetahui mana yang memiliki risiko tinggi untuk dibatalkan. Ketidakpastian ini 
        membuat hotel kesulitan dalam merencanakan okupansi dan menetapkan strategi 
        overbooking yang tepat. 

        Ketika hotel mengalokasikan tenaga kerja, persiapan kamar, serta perencanaan 
        kapasitas untuk seluruh reservasi tanpa mempertimbangkan risiko pembatalan, 
        sebagian besar usaha berpotensi terbuang sia-sia apabila tamu tidak datang, sehingga 
        efisiensi operasional menurun dan biaya tambahan tidak terhindarkan.
    """
    )

    st.subheader("Goals")
    st.write(""" 
        Tujuan utama dari proyek ini adalah membuat model yang dapat memprediksi 
        dan mengidentifikasi kemungkinan pembatalan reservasi oleh pelanggan, sehingga 
        pihak manajemen hotel dapat meminimalkan kerugian, meningkatkan akurasi 
        perencanaan okupansi dan mendukung keputusan bisnis strategi penjualan. 
    """
    )

elif selected == "Data Overview":
    st.header("Data Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Data Asli", df_raw.shape[0])
    col2.metric("Total Fitur", df_eda.shape[1])

    st.subheader("Cuplikan Dataset")
    st.dataframe(df_eda.head())

    st.subheader("Statistik Deskriptif")
    st.write(df_eda.describe())

    st.subheader("Cek Tipe Data")
    buffer = pd.DataFrame(df_eda.dtypes, columns=['Data Type']).astype(str)
    st.table(buffer)

elif selected == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Analisis Target", 
        "Fitur Numerik", 
        "Fitur Kategorikal", 
        "Analisis Negara", 
        "Korelasi"
    ])

    # TAB 1
    with tab1:
        st.subheader("Distribusi Kelas Target (is_canceled)")
    
        cancel_rate = df_eda['is_canceled'].mean() * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Booking", f"{len(df_eda):,}")
        with col2:
            st.metric("Persentase Cancel", f"{cancel_rate:.2f}%")
        with col3:
            st.metric("Persentase Check-In", f"{100 - cancel_rate:.2f}%")

        counts_df = df_eda['is_canceled'].value_counts().reset_index()
        counts_df.columns = ['Status', 'Jumlah']
        counts_df['Status'] = counts_df['Status'].map({0: 'Not Cancel (0)', 1: 'Cancel (1)'})
            
        fig_target = px.bar(
            counts_df,
            x='Status', y='Jumlah',
            color='Status',
            text_auto=True,
            title="Perbandingan Jumlah Booking",
            color_discrete_sequence=['#8B6F47', '#C19A6B']   
        )
        st.plotly_chart(fig_target, use_container_width=True)

    # TAB 2 NUMERIK
    with tab2:
        st.subheader("Analisis Variabel Numerik terhadap Pembatalan")

        fitur_numerik = st.selectbox(
            "Pilih Fitur Numerik:",
            ["days_in_waiting_list", "total_of_special_requests", "booking_changes", "previous_cancellations"]
        )

        col_hist, col_box = st.columns(2)

        df_vis = df_eda.copy()
        df_vis['is_canceled'] = df_vis['is_canceled'].map({0:'Not Cancel', 1:'Cancel'})

        with col_hist:
            fig_hist = px.histogram(
                df_vis, x=fitur_numerik, color='is_canceled',
                barmode='group',
                title=f"Distribusi {fitur_numerik}",
                color_discrete_sequence=['#8B6F47', '#C19A6B']
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_box:
            fig_box = px.box(
                df_vis, x='is_canceled', y=fitur_numerik, color='is_canceled',
                title=f"Boxplot {fitur_numerik}",
                color_discrete_sequence=['#8B6F47', '#C19A6B']
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # TAB 3 KATEGORIKAL
    with tab3:
        st.subheader("Rasio Pembatalan per Kategori")

        fitur_kat = st.selectbox(
            "Pilih Fitur Kategorikal:",
            ["market_segment", "deposit_type", "customer_type", "required_car_parking_spaces"]
        )

        fig_cat = px.histogram(
            df_vis, x=fitur_kat, color="is_canceled",
            barnorm='percent',
            text_auto='.1f',
            title=f"Rasio Cancel pada {fitur_kat}",
            color_discrete_map={'Not Cancel': '#C19A6B', 'Cancel': '#8B6F47'}
        )
        
        fig_cat.add_hline(y=50, line_dash="dash", line_color="black")
        fig_cat.update_layout(yaxis_title="Persentase (%)")

        st.plotly_chart(fig_cat, use_container_width=True)

    # TAB 4 NEGARA
    with tab4:
        st.subheader("Top 10 Negara dengan Booking Terbanyak")

        top_countries = df_eda['country'].value_counts().head(10).index
        df_top_country = df_eda[df_eda['country'].isin(top_countries)]

        country_cancel = df_top_country.groupby('country')['is_canceled'].mean().reset_index()
        country_cancel['Percent_Cancel'] = country_cancel['is_canceled'] * 100
        country_cancel = country_cancel.sort_values('Percent_Cancel', ascending=False)

        fig_country = px.bar(
            country_cancel,
            x='country',
            y='Percent_Cancel',
            title="Persentase Cancel pada Top 10 Negara",
            text_auto='.1f',
            color='Percent_Cancel',
            color_continuous_scale=['#F5E9DA', '#C19A6B', '#8B6F47']
        )

        st.plotly_chart(fig_country, use_container_width=True)

        st.write("Negara *PRT (Portugal)* memiliki tingkat pembatalan tertinggi di antara top 10 negara asal tamu.")

    # TAB 5 KORELASI
    with tab5:
        st.subheader("Correlation Heatmap")

        numeric_df = df_eda.select_dtypes(include=['number'])
        corr = numeric_df.corr()

        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap='YlOrBr',
            linewidths=0.5,
            ax=ax
        )
        st.pyplot(fig_corr)