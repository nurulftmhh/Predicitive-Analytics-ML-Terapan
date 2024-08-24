# Laporan Proyek Machine Learning Terapan (Predictive Analytics) - Nurul Fatimah

## Domain Proyek
Perusahaan e-commerce di New York yang bergerak di bidang penjualan pakaian secara online menghadapi tantangan dalam meningkatkan pengeluaran tahunan pengguna (Yearly Amount Spent). Meskipun memiliki platform digital yang cukup kuat, mereka juga menawarkan sesi pertemuan offline di toko fisik, di mana pelanggan dapat berkonsultasi dengan personal stylist untuk mendapatkan saran gaya. Setelah sesi konsultasi, pelanggan memiliki opsi untuk memesan pakaian melalui aplikasi seluler atau situs web perusahaan. Namun, perusahaan melihat bahwa walaupun fasilitas ini sudah tersedia, pengeluaran tahunan pengguna tidak meningkat secara signifikan. Oleh karena itu, perusahaan perlu menganalisis faktor-faktor yang dapat mempengaruhi pengeluaran tahunan pengguna, terutama dengan mempertimbangkan durasi keanggotaan pelanggan (Length of Membership) dan preferensi pelanggan dalam menggunakan aplikasi seluler atau situs web untuk melakukan pemesanan.

Masalah utama yang dihadapi perusahaan adalah bagaimana meningkatkan pengeluaran tahunan pengguna. Jika masalah ini tidak diselesaikan, perusahaan mungkin kehilangan peluang untuk meningkatkan pendapatan dari pelanggan yang sudah ada. Oleh karena itu, penting untuk memahami faktor-faktor yang memengaruhi pengeluaran tahunan pengguna. 

Dari permasalahan tersebut, maka penulis membuat sebuah penelitian yang dapat mengembalikan seberapa banyak uang yang akan pelanggan habiskan dengan cara memasukkan beberapa fitur dari datasetnya serta membantu pengambilan keputusan terkait upaya perusahaan untuk memfokuskan penjualan mereka pada pengalaman aplikasi eluler atau situs web mereka. 

## Business Understanding
### Problem Statements:

1. Bagaimana cara meningkatkan pengeluaran tahunan pengguna dengan memahami pengaruh durasi keanggotaan dan preferensi penggunaan aplikasi seluler atau situs web?
2. Bagaimana cara memprediksi berapa banyak uang yang akan dihabiskan pelanggan berdasarkan faktor-faktor tertentu seperti durasi keanggotaan dan platform yang digunakan?

### Goals:
1. Mengidentifikasi fitur utama yang mempengaruhi pengeluaran tahunan pengguna
2. Membuat model prediksi pengeluaran tahunan berdasarkan data pelanggan
3. Memberikan rekomendasi strategi untuk meningkatkan penjualan

### Solution Statement
1. Solusi pertama dengan menggunakan metode Linear Regression, metode ini digunakan untuk memodelkan hubungan antara variable dependen dengan satu atau lebih variabel independent.
2. Solusi kedua dengan metode Random Forest, merupakan metode ensemble dalam machine learning yang menggabungkan beberapa pohon keputusan (decision tree) untuk meningkatkan kinerja prediktif 
3. Metrik yang digunakan adalah:
   - Mean Absolute Error (MAE): rata-rata dari selisih absolut antara nilai yang diprediksi dan nilai actual.
   - Mean Squared Error (MSE): rata-rata dari kuadrat selisih antara nilai yang diprediksi dan nilai aktual.
   - R-Squared (R²): metrik yang menunjukkan seberapa baik model regresi menjelaskan variasi data.   

## Data Understanding
Data yang digunakan berasal dari Kaggle dengan nama New York Ecommerce Customers atau dapat didownload [disini](https://www.kaggle.com/datasets/vibgyor24/new-york-ecommerce-customers/data).
- Overview Dataset (Pendekatan Statistik):
  - Dataset berdimensi 500 rows x 8 columns
   
    ![image](https://github.com/user-attachments/assets/78894ee4-3b27-4f20-8236-4eedcb148701)

  - Terdapat 3 fitur ber tipe object, dan 5 fitur bertipe float64

    ![image](https://github.com/user-attachments/assets/b8e2cde6-1b1c-4d41-bd32-dd6d04aded04)
 

  - Tidak ditemukan data kosong dan tidak ada data duplikat
   
    ![image](https://github.com/user-attachments/assets/25d42225-e568-4ffe-839f-9071fb0b8947)
 
- Fitur Dataset
Dataset (Ecommerce_Customers.csv) berisi variabel-variabel berikut:
  -	Email : alamat email pelanggan.
  -	Address : alamat rumah pelanggan.
  -	Avatar : avatar yang dipilih oleh pelanggan di situs online toko.
  -	Avg. Session Length : rata-rata waktu yang dihabiskan ketika mengikuti sesi saran gaya di dalam toko.
  -	Time on App : rata-rata waktu yang dihabiskan di Aplikasi dalam hitungan menit.
  -	Time on Website : rata-rata waktu yang dihabiskan di Situs Web dalam hitungan menit.
  -	Length of Membership : berapa tahun pelanggan telah menjadi anggota.
  -	Yearly Amount Spent : uang yang digunakan oleh pelanggan untuk membeli produk per tahunnya. Ini merupakan fitur target.

## Data Preparation
-	Menentukan kolom yang akan digunakan untuk analisis/prediksi yang disimpan dalam variabel selected_X. Hal tersebut dikarenakan terdapat 2 fitur kategorikal, yang mana fitur tersebut kurang relevan atau baik untuk prediksi.
-	Melihat ada tidaknya outlier yang divisualisasikan menggunakan boxplot. Outlier adalah data yang memiliki nilai yang sangat berbeda atau menyimpang jauh dari sebagian besar data lainnya dalam dataset, sehingga perlu dilakukan penghapusan jika terdapat outlier agar tidak mengurangi kinerja model/kesalahan prediksi.
-	Standarisasi menggunakan StandardScaller. Proses standarisasi membantu dalam membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. StandardScaller melakukan proses standarisasi dengan mengurangkan mean atau nilai rata rata kemudian membaginya dengan standar deviasi untuk menggeser nilai distribusi.Menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Scalling dengan Standard Scaler dari sklearn,dengan tujuan membuat numerik dari data agar dapat diproses atau dibaca oleh mesin.
-	Dari hasil pengecekkan outlier, perlu dilakukan penghapusan outlier. Dalam hal ini menggunakan metode IQR (Interquartile Range). IQR adalah jarak antara kuartil ketiga dan kuartil pertama. IQR digunakan untuk mengukur rentang tengah dari data dan sering digunakan untuk mendeteksi outlier.
-	Train/test split atau train_test_split merupakan salah satu metode yang digunakan untuk mengevaluasi performa model machine learning. Metode ini membagi dataset menjadi 2 bagian yaitu dataset untuk training dan dataset untuk testing dan sudah ditentukan proporsi pembagiannya, umumnya data train 80% dan data test 20%, atau data train 75% dan data test 25%, tergantung kebutuhan. Penggunaan train_test_split ini juga sangat dianjurkan untuk dataset yang berukuran besar atau bisa juga untuk multiclass. 

## Modelling
Dalam hal ini, menggunakan 2 metode atau algoritma, yaitu:
- Linear Regression
 
  Linear regression merupakan salah satu algoritma supervised learning, yang mana regresi membuat model prediksi untuk target variabel berdasarkan dari bebasnya.
  
  ![image](https://github.com/user-attachments/assets/43675bf8-370b-42cf-b2cb-380ad4134d11)

  Rumus secara umumnya yaitu:
  
  ![image](https://github.com/user-attachments/assets/be0b4194-38ae-4c35-b1c2-2cb7654ef544)

   Dalam model ini, kita menggunakan data pelatihan (training dataset) untuk memprediksi hubungan antara variabel independen (X) dan variabel dependen (Y). Tujuannya adalah menemukan garis linear terbaik yang dapat menjelaskan hubungan tersebut.

- Random Forest

  Random forest merupakan kombinasi dari masing – masing tree yang baik kemudian dikombinasikan ke dalam satu model agar membentuk sebuah prediksi.
   
  ![image](https://github.com/user-attachments/assets/9d257da7-c7ff-4fa5-9fd3-d2a2d5ba8cf2)

  Rumus secara umumnya yaitu:
  
  ![image](https://github.com/user-attachments/assets/4133a51f-ee99-41b9-aa19-8b0faae555bc)

  Setiap tree menciptakan penduga masing masing atau disebut n_estimators(subpohon) dalam mengambil sebuah keputusan yang berbeda. Dari situlah,kemudian akan digabungkan agar membentuk 1 pohon utuh (akhir) agar dapat memberikan hasil klasifikasi dan prediksinya.

Tahapan yang dilakukan:
1. Mendefinisikan metode atau algoritma yang digunakan, dalam hal ini adalah linear regression dan random forest
2. Melakukan train data menggunakan kedua algoritma tersebut dengan dataset yang sama
3. Melakukan testing/predict terhadap model yang sudah dibuat
Tidak ada parameter tambahan yang digunakan dalam kedua model ini, karena default parameters sudah cukup untuk kasus umum.
 
Berdasarkan hasil dari proses modelling, model linear regression memberikan hasil yang lebih baik dibandingkan random forest. Berikut detailnya:
  - Model linear regression
   
    ![image](https://github.com/user-attachments/assets/ad6657a7-6887-4c16-ae51-58221b6398e7)
 
    ![image](https://github.com/user-attachments/assets/a30fcca3-ca17-44dd-9b1b-8f66d26c41ff)

  - Model random forest
   
    ![image](https://github.com/user-attachments/assets/52d8a5e6-bd04-4133-8628-4f4dedb2b64b)
 
    ![image](https://github.com/user-attachments/assets/ff4c1a5d-a21e-4aae-988a-de8126f89f8c)
 
## Evaluation
Metrik evaluasi yang digunakan diantaranya Mean Absoulte Error (MSE), Mean Squared Error (MSE), R-squared.
- Mean Absolute Error (MAE)
  MAE adalah rata-rata dari nilai absolut selisih antara prediksi model dan nilai aktual. MAE mengukur seberapa besar kesalahan rata-rata yang terjadi tanpa memperhitungkan arah (positif atau negatif) dari kesalahan tersebut.

  Formula:
  
  ![image](https://github.com/user-attachments/assets/518a8091-8bdf-4fd2-82d4-a4337db535fe)

  Dimana 𝑦𝑖 adalah nilai aktual, 𝑦^𝑖 adalah nilai prediksi, dan N adalah jumlah data.
  
- Mean Squared Error (MSE)
  MSE adalah rata-rata dari kuadrat selisih antara prediksi model dan nilai aktual. MSE lebih sensitif terhadap outlier dibandingkan MAE karena perbedaan yang lebih besar diberi bobot lebih besar (karena dikuadratkan).

  Formula:
  
  ![image](https://github.com/user-attachments/assets/6ee9c4a7-2e6f-471b-b044-1e162fb159b6)

- R-squared (R²)
  R² adalah ukuran statistik yang menunjukkan seberapa baik model regresi menjelaskan variasi dalam data. Nilai R² berkisar dari 0 hingga 1. R² mengukur proporsi variasi total dari data yang dapat dijelaskan oleh model. Semakin mendekati 1, semakin baik model dalam menjelaskan data.

  Formula:
  
  ![image](https://github.com/user-attachments/assets/2a8f7f1a-fc45-4674-9205-02bec85a3e52)

  Di mana 𝑦ˉ adalah rata-rata dari nilai aktual.
   
Hasil evaluasi model regresi linear yang didapat, yaitu:

- Mean Absolute Error (MAE): 0.1033. Semakin kecil nilai MAE, semakin baik model dalam memprediksi.
- Mean Squared Error (MSE): 0.0174. Semakin kecil nilai MSE, semakin baik performa model.
- R-squared (R²): 0.9798. Nilai R² berkisar dari 0 hingga 1, di mana nilai yang lebih mendekati 1 menunjukkan bahwa model mampu menjelaskan sebagian besar variansi data target, berarti sekitar 97.98% variasi dari target (Yearly Amount Spent) dapat dijelaskan oleh model regresi linier ini.
Secara keseluruhan, model ini memiliki performa yang baik, dengan error yang rendah (MAE dan MSE) dan R² yang tinggi, menunjukkan bahwa model ini cukup akurat dalam memprediksi variabel target.

Hasil evaluasi model random forest yang didapat dari kedua metrik tersebut yaitu:

- Mean Squared Error (MSE): 0.0656. Semakin kecil nilai MSE, semakin baik performa model.
- R-squared (R²): 0.924. Nilai R² berkisar dari 0 hingga 1, di mana nilai yang lebih mendekati 1 menunjukkan bahwa model mampu menjelaskan sebagian besar variansi data target, berarti sekitar 92.4% variasi dari target (Yearly Amount Spent) dapat dijelaskan oleh model random forest ini.
Secara keseluruhan, model Random Forest yang digunakan memiliki performa yang kuat, dengan kesalahan prediksi yang rendah dan kemampuan yang tinggi dalam menjelaskan variasi data.

**Kesimpulan Hasil Evaluasi:**

Dalam kasus ini, model Regresi Linear tampaknya memberikan hasil yang lebih baik dibandingkan Random Forest, baik dari segi kesalahan prediksi (MSE) yang lebih rendah maupun kemampuan menjelaskan variasi dalam data (R²) yang lebih tinggi.
Oleh karena itu, model Regresi Linear lebih cocok digunakan untuk data ini daripada model Random Forest, karena memberikan prediksi yang lebih akurat dan lebih baik dalam menjelaskan hubungan antara variabel prediktor dan target.


  
 

