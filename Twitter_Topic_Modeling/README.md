# Twitterda Atılmış Tweetlerin Kategori Tahmini

<br/>

Bir sınıflandırma algoritmasıdır. Twitterdan alınmış olan farklı kategorilerdeki (Siyaset, Ekonomi, Spor, Teknoloji ve Bilim) tweetleri ile model eğitiliyor ve modele bir tweet gönderildiğinde bunun hangi kategoride bir tweet olduğunu tespit ediyor.
<br/>
## Kullanılan Kütüphaneler
-Pandas<br/>
-NumPy<br/>
-snscrape<br/>
-Scikit-learn<br/>
-unicodedata <br/>
-Nltk<br/>
-Zeyrek<br/>
-Pyplot<br/>
-Tqdm<br/><br/>


# VERİ SETİ'NİN OLUŞTURULMASI

Dataseti twitter'dan çekilen ve istenilen kategorilerde atılmış tweetlerden oluşacaktır. Seçilen kategoriler Spor, Ekonomi, Siyaset, Teknoloji & Bilim ' dir. Bu kategorilerdeki veriler ile model eğitileceği için bu verilerin düzgün veriler olması gerekir. Bu yüzden verilerin çekileceği twitter hesapları düzgün bir şekilde seçilmiştir. Seçilen hesaplarda dikkat edilen özellikler şöyledir:

<br/>
-Hesap aktif olarak paylaşım yapmalı.<br/>
-Yaptığı paylaşımlar sadece o alan ile ilgili olmalı.<br/>
-Günde ortalama 5-10 arası tweet atmalı.<br/>
-Atmış olduğu tweetler metin ağırlıklı olmalı.<br/>


<br/>

Bu kriterleri sağlayan her bir alan için ortalama 8 tane twitter hesabı seçilmiştir. Birden fazla hesap seçilmesinin amacı daha fazla veriye ulaşmaktır. 


<br/>
Pythonda twitter'dan istenilen tweetleri çekebilmek için scraping işlemi yapan snscrape pyton kütüphanesi kullanılıyor. <br/>
https://github.com/JustAnotherArchivist/snscrape

<br/>
<br/>

Bu kütüphane verilen sorgu cümlesine göre ve istenilen tweet sayısına göre istenilen tweetleri döndürüyor. Sorgu cümlesinde hesap adı ve istenilen tarih aralığı seçilebilir.<br/><br/> 
### Örnek bir sorgu:<br/><br/>

![ornek_sorgu](https://user-images.githubusercontent.com/77435563/208323507-bd8fee5f-4572-40fe-bb05-75aee43c919e.jpg)

<br/>
Bu sorguda "elonmusk" isimli kullanıcının 17 aralık 2022 tarihinden önce atmış olduğu tweetler gösteriliyor.

<br/>
<br/>

Data seti'ni oluştururken yapılan sorgular 2022 yılının tamamı için yapılmıştır. Her farklı kategori için 1 günde 1 hesaptan 16 adet tweet çekilmiştir. Hesaplar gün gün değişmektedir ve eğer 8 hesap varsa 8 gün sonunda aynı hesaptan tekrardan 16 adet tweet çekilmektedir. Verisetinin bu şekilde oluşturulmasının amacı aynı gün ve saatte birden fazla hesaptan aynı tweet gelme ihtimalinin engellenmesidir. 

<br/>

Veriseti oluşturmak için twitterdan her gün bir kategori için 16 adet tweet çekiliyor. 365 ayrı gün için tweet çekildiği için her kategori için 5840 adet tweet bulunmakta.
4 kategorimiz olduğu için veri seti'nde toplam 23.360 tweet bulunuyor. Modelin başarısını arttırmak için her kategori için eşit miktarda tweet çekiliyor.<br/><br/>
### Veri seti'nde Bulunan Kategorilerdeki Tweet Sayıları:<br/><br/>

![data_set_veri_sayisi](https://user-images.githubusercontent.com/77435563/208323609-097d1ee4-3259-43ca-83ae-c65f41f1a387.jpg)

<br/>
<br/>

4 kategori için 23.360 verinin çekilmesi işlemi ortalama 11 dakika 14 saniye sürüyor.<br/><br/>
### Verilerin Çekilme Süreleri:<br/><br/>

![Calisma_Suresi](https://user-images.githubusercontent.com/77435563/208323627-4d1c1486-3fd5-4eac-85db-676e4d7ae64f.jpg)

<br/>
<br/>

Veri seti'ndeki kolonlarımız aşağıdaki gibidir:<br/>
-Tweet'in atıldığı tarih.<br/>
-Tweet'i atan kullanıcının hesap adı.<br/>
-Tweet'in kendisi.<br/>
-Tweet'in hangi kategoride atıldığı.<br/>

<br/><br/>
Veri seti'ne veriler eklenirken kategorileri ile birlikte ekleniyorlar. Bunun sebebi oluşturulacak olan modeli eğitmek için hangi kategoriye ait olduğunu bilmemiz gerektiğidir. <br/><br/>

### Oluşturulmuş Olan Veri seti'nin Görseli:<br/><br/>
![data_set](https://user-images.githubusercontent.com/77435563/208323700-beb37519-6467-4230-b2e3-b59df79e17e8.jpg)
<br/><br/>
# Tweetlerin Temizlenmesi ve Lemmatization İşlemi<br/>
Veriseti oluşturulduktan sonra modelin daha iyi çalışması ve başarı oranının daha yüksek olması için tweetlerin temizlenmesi gerekmektedir. Tweetlerin içerisinde emojiler, noktalama işaretleri, stopwordsler, linkler gibi istenmeyen ve modelin başarısını düşürecek veriler tweetlerin içerisinden temizleniyor. Daha sonra lemmatization (kelimelerin köklerinin alınması) işlemi yapılarak temiz ve kelimelerin köklerinden oluşan tweetler elde ediliyor.
<br/><br/>
### Oluşturulan Temiz Tweet Görseli<br/><br/>
![clean](https://user-images.githubusercontent.com/77435563/209435327-a5f9ffd7-bdcd-4f6e-954c-bb912484d395.jpg)
<br/><br/>

# Modelin Oluşturulması ve Tweetlerin Kategorilendirilmesi<br/>

## Model Seçimi
Yapılacak kategorilendirme işleminin hangi modelde daha yüksek başarı oranı vereceğini tespit etmek amacıyla araştırma yapılıp aynı zamanda bazı modeller üzerinde de test edilmiştir. Başlangıç olarak 3 popüler model üzerinde denemeler yapılmıştır. Bu modeller Naive Bayes, DecitionTree ve K-Nearest Neighbor modelidir. Veriseti üzerinde bu modellerin accuracy ve f1 score ları test edilmiştir. Projedeki test veriseti sonuçlarına bakıldığında:<br/>
Naive Bayes Modeli için  accuracy: 0.864 <br/>
DecitionTree Modeli için accuracy: 0.765 <br/>
K-Nearest Neighbor için accuracy: 0.774 <br/>
<br/>
Sonuçlar incelendiğinde DecitionTree ve K-Nearest Neighbor modelinin projede kullanılan verisetine göre yapacağı kategorilendirmenin başarısı yeterli olmamıştır. Bu iki modelin kullanımından vazgeçilmiştir. Alternatif model arayışı için araştırma yapılıp yine sınıflandırma için çok kullanılan model olan Support Vector Machine modeli araştırılıp başarısının ölçülmesi için test edilmiştir. Projedeki test veriseti sonuçlarına bakıldığında:<br/>
Support Vector Machine Modeli için accuracy: 0.868 <br/>
<br/><br/>
Yapılan test işlemleri sonucunda projede Naive Bayes ve Support Vector Machine modeli kullanılmıştır. 

## Modelin Oluşturulmaya Başlanması

### Etiketleme
Clean tweetlerin bulunduğu "clean_all_tweets.csv" dosyası dataframeye aktarılmıştır. Tweetlerin kategorileri kelime şeklinde kayıtlı olduğu için bu kategoriler 0, 1, 2, 3 gibi bilgisayarın anlayabileceği bir formata dönüştürülmelidir. Labels adında yeni bir kolon açılarak kategorisi spor olan tweetler için 0 rakamı, kategorisi ekonomi olan tweetler için 1 rakamı, kategorisi siyaset olan tweetler için 2 rakamı ve kategorisi teknoloji & bilim olan tweetler için 3 rakamı labels olarak eklenmiştir.<br/><br/>
 
![labels](https://user-images.githubusercontent.com/77435563/209437433-be95afad-bcfc-4654-bde2-21c913d844fd.jpg) <br/><br/>

### Verisetinin Parçalanması
Modelin başarısını doğru şekilde ölçebilmek için modeli eğittiğimiz veriler ile test ettiğimiz veriler farklı olmalıdır. Modelin eğitilmiş olduğu verileri tekrar modele gönderirsek model bu veriler ile eğitildiği için başarısı yüksek ve yanıltıcı olacaktır. Bu yüzden verisetini parçalamamız gerekir. Bu projede verisetinin %80 'i modeli eğitmek için, %20 'si de modeli test etmek için kullanılacaktır. <br/>
Aşağıda verisetinin nasıl parçalanacağının bir örneği gösterilmiştir.<br/><br/>
![train_test](https://user-images.githubusercontent.com/77435563/209437930-439dc4dd-c353-4205-b437-3d6d9aa5762d.jpg)
<br/>
Verisetini parçalamak için train_test_split() fonksiyonu kullanılmıştır.<br/>

### Tweetlerin Vektörel Matrisinin Çıkarılması
Tweetler metinden oluştuğu için bunun bilgisayar ortamında işlenmesi mümkün değildir bu yüzden veriler sayısal değerlere dönüştürülmelidir. Bir sözlük oluşturularak dökümandaki her kelime için bir indexleme yapılır. Daha sonra hangi index numarasına sahip kelimenin hangi tweette kaç kere geçtiği hesaplanarak sayma matrisi oluşturulur. Bu işlemi yaparken tf-idf vectorizer kullanılarak bir kelimenin döküman içindeki önemi istatistiksel olarak hesaplanmıştır. Bu sayede her tweette geçen model için anlamsız kelimelerin önemi düşürülmüştür yani stopwordsler tekrardan ayıklanmıştır. <br/><br/>

### Modellerin Eğitilmesi
Daha önceden parçalanmış olan X_train ve y_train verileri Naive Bayes ve Support Vector Machine modeline gönderilerek modeller eğitilmiştir. Eğitim sonucunda modellerin accuracy ve f1 score değerleri hesaplanmıştır. Modelleri eğitmek için sklearn kütüphanesi kullanılmıştır.<br/><br/>

### Modellerin Başarısının Hesaplanması
Modelin başarısı hem train hem test verileri üzerinden Accuracy ve F1 score ile ölçülmüştür. Alınan sonuçlar aşağıda bulunmaktadır.<br/><br/>

Naive Bayes             |  Support Vector Machine
:-------------------------:|:-------------------------:
![BayesClassifier](https://user-images.githubusercontent.com/77435563/209438951-be49cbe3-1135-444f-8c19-84211fd98c0f.jpg)  |  ![SVMClassifier](https://user-images.githubusercontent.com/77435563/209439020-68b78a8a-a88d-4410-b5bb-7f1869a8fb2c.jpg) <br/><br/><br/>

### Kategorilere Göre Başarı Dağılımları
Kategorilere göre modellerin test verisetindeki başarı dağılımları classification_report() fonksiyonu kullanılarak hesaplanmıştır. Alınan sonuçlar aşağıda bulunmaktadır.<br/><br/>

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
