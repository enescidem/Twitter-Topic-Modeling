# Twitter Topic Modeling

### Medium Links:
English: https://medium.com/@enescidem/twitter-x-topic-modeling-556d5aad5d0a

Türkçe: https://medium.com/@enescidem/twitter-x-konu-modelleme-e713fea6224f

---

## Dünyada En Çok Etkileşime Giren Yerli Twitter Hesaplarının Topic Modeling'i
Topic Modeling, bir metin belgesinde “topics(konular)” adı verilen kelime gruplarını bulmak için kullanılan denetimsiz(unsupervised) bir yaklaşımdır. Twitter(X)'dan alınmış olan farklı hesaplardan (BabalaTv, ProfDemirtas, haluklevent, pusholder, Darkwebhaber, yirmiucderece, solcugazete, OguzhanUgur, vekilince, GalatasaraySK, aykiricomtr ve RTErdogan) çekilen tweetler analiz edilip topiclerinin belirlenmesi.
<br/>
## Kullanılan Kütüphaneler
-Pandas<br/>
-NumPy<br/>
-Ntscrapper<br/>
-Matplotlibt<br/>
-Scikit-learn<br/>

# VERİ SETİ'NİN OLUŞTURULMASI

Dataseti Twitter(X)'dan çekilen ve istenilen hesaplardan atılmış tweetlerden oluşacaktır. Seçilen hesaplar BabalaTv, ProfDemirtas, haluklevent, pusholder, Darkwebhaber, yirmiucderece, solcugazete, OguzhanUgur, vekilince, GalatasaraySK, aykiricomtr ve RTErdogan ' dır. Bu hesaplardaki veriler ile model eğitileceği için bu verilerin düzgün veriler olması gerekir. Bu yüzden verilerin çekileceği twitter hesapları düzgün bir şekilde seçilmiştir.

<br/>

Bu twitter hesaplarının seçilme sebebi dünyada en çok etkileşime giren yerli Twitter(X) hesapları olmasıdır:

![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/bb5cbcbc-70d6-4581-bcbf-9877be9488bc) 
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/7866e486-5890-4a11-8daa-c07ebc7472d6)

[https://dogruveri.com/wp-content/uploads/2023/03/Dunyada-en-cok-etkilesime-giren-yerli-Twitter-hesaplari-.png](https://dogruveri.com/dunyada-en-cok-etkilesime-giren-yerli-twitter-hesaplari/)


<br/>
Pythonda twitter'dan istenilen tweetleri çekebilmek için scraping işlemi yapan ntscraper kütüphanesi kullanılıyor. <br/>
https://github.com/zedeus/nitter?tab=readme-ov-file

<br/>
<br/>

Bu kütüphane verilen kullanıcı adına veya hastage göre ve istenilen tweet sayısına göre istenilen tweetleri döndürüyor.<br/><br/> 

Data seti'ni oluştururken yapılan sorgular 12 adet hesaptan 1000'er adet tweet çekilerek oluşturulmuştur. Her hesaptan 1000'er adet tweet çekemediği için 900, 800, 750, 600 olarak değişkenlik göstermektedir. Toplam 12 hesaptan 8810 adet tweet çekilmiştir.

<br/>
<br/>

12 hesap için 8810 adet verinin çekilmesi işlemi ortalama 35-40 dakika sürüyor.<br/><br/>

<br/>
<br/>

Veri seti'ndeki kolonlarımız aşağıdaki gibidir:<br/>
-Tweet'in linki <br/>
-Tweet'in kendisi.<br/>
-Tweet'in like sayısı.<br/>
-Tweet'in yorum sayısı.<br/>

### Oluşturulmuş Olan Veri seti'nin Görseli:<br/><br/>
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/a4859957-af91-4d98-9c55-fcd7ad4dde03)
<br/><br/>
# Tweetlerin Temizlenmesi ve Lemmatization İşlemi<br/>
Veriseti oluşturulduktan sonra modelin daha iyi çalışması ve başarı oranının daha yüksek olması için tweetlerin temizlenmesi gerekmektedir. Tweetlerin içerisinde emojiler, noktalama işaretleri, stopwordsler, linkler gibi istenmeyen ve modelin başarısını düşürecek veriler tweetlerin içerisinden temizleniyor. 
<br/><br/>
### Tweetlerin Temizlenmesi Sırasında Yapılan İşlemler<br/><br/>
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/ec423e8e-3dca-4f75-8d44-9b18042e34f7)
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/fef03545-0fd3-4c30-91c1-de82ba31a037)

<br/><br/>
### Oluşturulan Temiz Tweet Görseli<br/><br/>
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/5a9c312f-c416-492d-a2d9-8967e6532907)
<br/><br/>

# Modelin Oluşturulması ve Tweetlerin Kategorilendirilmesi<br/>

## Model Tanımı
Modelimiz metin verileri üzerinde iki farklı konu çıkarma yöntemi olan LDA ve NMF'nin çıkarılması ve her iki yöntemin sonuçlarını görselleştirmek için grafikler oluşturulmasından ibaretdir. 

### Latent Dirichlet Allocation (LDA):

LDA, belgelerin bir koleksiyonu içinde gizli olan temel konuları (topic) çıkarmak için kullanılan bir olasılıksal bir modeldir.
Temel bir varsayım, belgelerin bir veya birden fazla konuya ait olabileceği ve her bir konunun bir olasılık dağılımı ile temsil edilebileceğidir.
LDA, belge-konu ve kelime-konu olasılık dağılımlarını keşfetmeye çalışarak belgeleri bu gizli konulara ayırır.
Bu model, genellikle metin belgeleri üzerinde konu çıkarma, belge sınıflandırma ve benzeri görevlerde kullanılır.

### Non-negative Matrix Factorization (NMF):

NMF, bir matrisin çarpanlarını (faktörlerini) bulmaya çalışan bir matris ayrıştırma tekniğidir.
Belirli bir veri matrisini, iki veya daha fazla alt matrise çarpanları çarpanları (faktörleri) olarak ayrıştırmaya odaklanır. Bu faktörler genellikle pozitif değerlere sahiptir.
Metin madenciliği bağlamında, NMF belgeleri ve kelimeleri içeren bir matrisi, temel konuların (topic) lineer kombinasyonları olarak ifade etmeye çalışır.
NMF'nin kullanıldığı yerler arasında konu çıkarma, görüntü işleme ve özellik seçimi gibi alanlar bulunmaktadır.

## Modelin Oluşturulmaya Başlanması

### Veri Setinin Hazırlanması
Clean tweetlerin bulunduğu `clean_tweets.csv` dosyası dataframeye aktarılmıştır. Daha sonra veri seti içinde NaN değeri olan satırlar varsa bu satırlar silinir. NaN değerleri olursa modelimiz hata vermektedir.<br/><br/>

### Metin Verilerini Sayısal Bir Formata Dönüştürme İşlemi
TfidfVectorizer fonksiyonu bir metin madenciliği aracıdır ve metin verilerini TF-IDF (Term Frequency-Inverse Document Frequency) vektörlerine dönüştürmek için kullanılır.
`max_df` ve `min_df` parametreleri, vektörleştirme işlemi sırasında dikkate alınacak terimlerin belirlenmesine yardımcı olan önemli parametrelerdir:
   `max_df`: Belirtilen bir eşik değerinden yüksek olan terimler, belgelerin yüzde kaçında görülüyorsa, dikkate alınmaz.       Bu, genellikle sık kullanılan kelimelerin (stop words) veya çok spesifik kelimelerin filtrelenmesinde kullanılır.

   `min_df`: Belirtilen bir eşik değerinden düşük olan terimler, belgelerin yüzde kaçında görülüyorsa, dikkate alınmaz.   
    Bu, nadir görülen terimleri filtrelemek için kullanılır.
    
`vectorizer.fit_transform(df["clean"])` ile, `clean` adlı sütundaki metin verileri üzerinde TF-IDF vektörleştirmesi yapılır. Bu işlem, her bir belgeyi vektörlerle temsil eden bir matris oluşturur.

![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/f28d94e1-3d82-4dbb-b937-8e87bdfaa346)

### Latent Dirichlet Allocation (LDA) modeli oluşturma
`LatentDirichletAllocation` bir konu modelleme tekniğidir ve belgelerin gizli konularını çıkarmak için kullanılır.

`n_components=8` parametresi, modelin kaç adet konu çıkaracağını belirtir. Bu örnekte, 8 adet konu belirlenmiştir.

`random_state=42` parametresi, modelin tekrarlanabilirliğini sağlamak için kullanılır. Aynı `random_state` değeri kullanıldığında, modelin başlangıç durumu her seferinde aynı olacaktır.

Bu yapılandırma ile `LatentDirichletAllocation` sınıfından bir LDA modeli oluşturulur. Bu model, veri setindeki belgelerin gizli konularını keşfetmeye çalışacaktır.

Daha sonra, bu model `lda.fit(X)` ile eğitilir. `X` önceki aşamada oluşturulan TF-IDF vektör matrisini temsil eder. Model, bu vektör matrisi üzerinde çalışarak belgelerin gizli konularını öğrenir.

![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/246cabaa-cea6-4f15-992b-73a79ebfdd8a)

 <br/><br/>

### Modelleri Görselleştirme
Modeli görselleştirmek için `plot_top_words` fonksiyonunu yazıyoruz.<br/><br/>
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/432eadd3-78e2-489d-92a3-47b865ed82d0)


### LDA Modelinin Görselleştirilmiş Hali
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/01078756-1e79-41c1-916c-58c301484da9)


### NMF Modelinin Görselleştirilmiş Hali
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/cd91089b-ba66-4fe0-96bc-030441e871c3)


# Modellerin Tweetlere Göre Önerdiği Topicler<br/>


## LDA Modelinin Tweetlere Göre Önerdiği Topicler
### Test 1:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/525e0018-6c7a-4045-a7ea-61e0afdc3afa)

### Test 2:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/17a798d0-0d7b-43af-b048-e97974569059)

### Test 3:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/9b993aa8-5be5-4f71-8e90-2fb117ce13ea)

### Test 4:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/ab5051d6-f3a3-4b45-9de1-1a04d21c9fda)

### Test 5:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/255f2c48-0199-4ab7-9e63-f26d3b21cb47)


## NMF Modelinin Tweetlere Göre Önerdiği Topicler
### Test 1:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/43628e55-6a36-490c-970a-04aa24a7fca5)

### Test 2:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/67e9d1a0-1faa-46a7-9725-56725ed5c8e8)

### Test 3:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/34fc4328-0c48-4aa6-9337-928fe2c13c75)

### Test 4:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/df151241-a2ea-45eb-ad54-5fdce8fe1595)

### Test 5:
![image](https://github.com/enescidem/Dogal_Dil_Isleme/assets/92892867/e1d495e3-4bdc-40e2-83d3-1c979a6f032c)





# Sonuç
Sonuca bakıldığında yapılan projede seçilen modelin ne kadar önemli olduğu görülmektedir. Veri setindeki verilerin miktarı bu model için yeterli olmayabilir bu yüzden daha çok veri ile proje daha iyi hale getirilebilir. Model seçiminin önemi kadar verisetinin güzel bir şekilde hazırlanmış olması, verisetindeki noktalama işaretleri, semboller ve modele olumsuz etki edecek stopwordslerden arındırılmış olmasıdır. Bu projede başarı oranı çok yüksek değildir bunun en önemli sebebi verilerin yeterli sayıda olamamsıdır. Seçilmiş olan twitter hesaplarının türkiyede en çok etkileşim alan hesaplardan seçilmiş olması bu model için uygundur. Ancak bütün hesaplar aynı tweetleri paylaşabildiği için bu modelde farklı kategorilerin bulunduğu hesaplar seçilebilir. Örneğin bir spor paylaşımı yapan twitter hesabı, asgari ücret açıklandığında önemli bir haber olduğu varsayılarak aynı tweeti paylaşabilmektedir. Verisetinde 8500 üzerinde tweet olduğu için bu tweetlerin tek tek kontrol edilmesi mümkün değildir.
