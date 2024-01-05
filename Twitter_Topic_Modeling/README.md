# Dünyada En Çok Etkileşime Giren Yerli Twitter Hesaplarının Topic Modeling'i

<br/>

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


Naive Bayes             |  Support Vector Machine
:-------------------------:|:-------------------------:
![nb_basari (2)](https://user-images.githubusercontent.com/77435563/209439633-0d852e76-224c-40b6-8fa3-d5f37f361ae0.jpg)  |  ![svm_basari (2)](https://user-images.githubusercontent.com/77435563/209439642-34c95bb2-7912-49d0-98fb-7ee312272075.jpg) 
<br/>
Sonuçlar incelendiğinde Her iki modelin de en başarılı tahmin yaptığı kategorinin spor kategorisi olduğunu görülmektedir. Ekonomi ve siyaset alanında ise başarının 0.8 oranına kadar düştüğü görülmektedir. Bunun sebebi ekonomi ve siyaset alanında atılan tweetlerin birbiri ile bağlantılı tweetler olmasıdır. Çünkü ülke yönetimi yüzünden ekonominin kötü gidişatı hakkında atılan bir tweet düşünülürse bunun hem ekonomi, hem de siyasi bir tweet olduğunu söyleyebiliriz. Modelimiz tweeti bir kategoriye sokmak için zorladığından hata yapma oranı bu kategorilerde daha yüksektir.

<br/>

### Modellerin Tweetlere Göre Yaptığı Tahminler<br/>
Aşağıdaki görseldeki kolonlarda, tweet, tweetin bulunduğu kategori, Naive Bayes modelinin tweet için tahmini ve Support Vector Machine modelinin tweet için tahmini gösterilmektedir. <br/><br/>
0=spor ,  1=ekonomi ,  2=siyaset ,  3=teknoloji & bilim

![](https://user-images.githubusercontent.com/77435563/209441394-a70d3e80-9285-43fe-9495-59bc022935bb.png)
<br/>

Tweetlerin kategorilerine ve modellerin yaptığı tahminlere bakıldığında modeller oldukça iyi çalışıyor gibi gözüküyor. 4666. tweete bakıldığında naive bayes modelinin bu tweet için yanlış tahmin yaptığını görüyoruz.
<br/>
### Modellerin Hata Oranları
Modellerin hata oranlarını tespit etmek için 2 farklı metrik kullanılmıştır. Bunlar Ortalama Kare Hatası(MSE) ve Ortalama Mutlak Hata(MAE) dir. <br/>

##### Ortalama Kare Hatası(MSE)
Ortalama Kare Hatası tahmin edilen sonuçlarınızın gerçek sayıdan ne kadar farklı olduğuna dair size mutlak bir sayı verir.<br/>

##### Ortalama Mutlak Hata(MAE)
Ortalama mutlak hata, mutlak hata değerinin toplamını alır, hata terimlerinin toplamının daha doğrudan bir temsilidir.<br/>

![image](https://user-images.githubusercontent.com/77435563/209441007-321dd72c-936a-499d-b6ed-98d4fa4e533a.png)<br/><br/>
Sonuçlar incelendiğinde Support Vector Machine modelinin Naive Bayes modeline göre biraz daha fazla hata yaptığını görüyoruz.

<br/><br/>

### Manuel Tweet Testi
Modeli manuel olarak test etmek için elle bazı tweetler girilecek ve modelin bu tweetlerin hangi kategoriye ait olduğunu tahmin etmesi istenecektir. Test sonuçları aşağıda gösterilmiştir.
<br/>
##### Test1:
![test1](https://user-images.githubusercontent.com/77435563/209444285-c282e9bc-783d-46b9-9a78-c9680c74316a.jpg)
<br/>
##### Test2:
![test2](https://user-images.githubusercontent.com/77435563/209444313-207e0ace-2c6a-496d-816f-c42937360ecc.jpg)
<br/>
##### Test3
![test3](https://user-images.githubusercontent.com/77435563/209444333-fd699f6e-24c6-46b6-8d36-fe75841cb7a3.jpg)
<br/>
##### Test4
![test4](https://user-images.githubusercontent.com/77435563/209444345-0478781e-582a-4de5-817d-17e82a2834da.jpg)
<br/>
##### Test5
![test5](https://user-images.githubusercontent.com/77435563/209444369-68534217-4604-4b11-bd1f-22c7ef512f40.jpg)
<br/>
##### Test6
![test6](https://user-images.githubusercontent.com/77435563/209444379-f7e20721-ffac-4919-b7b4-ce5f19d3c624.jpg)
<br/>
##### Test7
![test7](https://user-images.githubusercontent.com/77435563/209444389-8d4d6302-1816-4df8-80d8-6fbc83653bed.jpg)
<br/>
##### Test8
![test8](https://user-images.githubusercontent.com/77435563/209444402-6eab3c37-0a33-44f2-accb-4dff92a3a4fe.jpg)
<br/>
##### Test9
![test9](https://user-images.githubusercontent.com/77435563/209444412-0cd7d777-a978-4355-a8ea-65774555555e.jpg)
<br/>
##### Test10
![test10](https://user-images.githubusercontent.com/77435563/209444419-4a078e5c-f34c-4cd1-a116-9221cd52c875.jpg)
<br/>
##### Test11
![test11](https://user-images.githubusercontent.com/77435563/209444428-702adbc1-a23f-4d5c-8368-03b34f58807e.jpg)
<br/>
##### Test12
![test12](https://user-images.githubusercontent.com/77435563/209444433-0c887130-7d0d-4a31-adbf-f4c270922631.jpg)
<br/>

# Sonuç
Sonuca bakıldığında yapılan projede seçilen modelin ne kadar önemli olduğu görülmektedir. Başlangıçta test edilmiş olan DecitionTree ve K-Nearest Neighbor modelinin bu veriseti için uygun modeller olmadığı görülmüştür. Yapılan farklı projelerde farklı modeller daha iyi sonuç verebilmektedir. Model seçiminin önemi kadar verisetinin güzel bir şekilde hazırlanmış olması, verisetindeki noktalama işaretleri ve modele olumsuz etki edecek stopwordslerden arındırılmış olması, verisetindeki verilerin miktari, eğitim-test verilerinin parçalanma oranı vs. modelin başarısına etki eden çok önemli faktörlerdir. Bu projede ulaşılan başarı oranı(%86) yüksek bir başarı oranına sahip değildir bunun en temel sebebi verisetini oluştururken farklı kategorilerden alınmış tweetlerin seçilmiş olan hesaplardan alınmasıdır. Seçilmiş hesaplar ne kadar özenle seçilmiş olsa da bazı tweetlerinde paylaşım yaptığı kategorinin dışına çıktığı görülmüştür. Örneğin spor paylaşımı yapan twitter hesabı, asgari ücret açıklandığında önemli bir gelişme olduğu için bunu da paylaşım yapabilmektedir. Verisetinde 23.000 üzerinde tweet bulunduğundan dolayı bu tweetlerin tek tek kontrol edilme imkanı yoktur. Veriseti oluşturulurken seçilen paylaşımların ve hesapların başarıda önemli bir rolü vardır.
