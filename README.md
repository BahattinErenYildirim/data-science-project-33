# Data Science Project 33 — Bank Marketing (Imbalanced Classification)

**Modül**: ML-03 (Sınıflandırma 1) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir Portekizli bankada **junior data scientist** olarak işe başladın. Banka, tele-pazarlama kampanyalarıyla müşterilere **vadeli mevduat hesabı (term deposit)** satıyor. Operatörler günde yüzlerce çağrı yapıyor — ama dönüşüm oranı düşük: aranan her 100 kişiden sadece ~11'i ürünü satın alıyor.

Senin görevin: hangi müşterilerin teklifi kabul edeceğini **önceden tahmin etmek**. Böylece operasyon ekibi çağrı listesini önceliklendirebilir, bütçeyi yüksek olasılıklı müşterilere harcayabilir.

Ama burada bir tuzak var: veride **sadece %11'i pozitif sınıf**. Bu **dengesiz (imbalanced)** bir problem. "Hep 'hayır' tahmin et" diyen aptal model bile %89 accuracy alır — ama hiçbir müşteriyi yakalayamaz. Burada **accuracy yalan söyler**; **Precision-Recall + AUC** doğruyu söyler.

Bu projede ML-03 dersinde öğrendiklerini birleştirip uygulayacaksın:
- ✅ **Gerçek dünya veri çekme** (URL → zip → CSV cache)
- ✅ **Class imbalance** tespiti ve etkisi
- ✅ **Categorical encoding** (one-hot, high cardinality)
- ✅ **LogisticRegression** + `class_weight='balanced'`
- ✅ **ROC Curve + AUC** hesaplama
- ✅ **PR Curve + Average Precision**
- ✅ **Dengesizde ROC vs PR farkı** (PR daha dürüst)
- ✅ **Threshold optimization** (F1 max veya recall hedefli)
- ✅ **Target leakage** tespiti ve önleme

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-33

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 713 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Dataset — UCI Bank Marketing

### Kaynak
- **UCI Machine Learning Repository** — [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Akademik kaynak: Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22-31.

### Veri Çekme Yöntemi (önemli)
Veri seti **repo'da YOK** — projeyi clone ettiğin gün **runtime'da indireceksin**. Gerçek dünyada veriler S3 / API / FTP üzerinden gelir; statik dosya olarak gönderilmez. Bu yüzden `fetch_bank_data()` fonksiyonun:

1. Hedef URL: `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`
2. `data/` klasörüne zip'i indirir (`urllib.request.urlretrieve`)
3. Zip'i açar → içinde **nested zip** var: `bank-additional.zip` 
4. O zip'i de açar → `bank-additional/bank-additional-full.csv` (41188 satır × 21 sütun)
5. Cache: dosya zaten varsa **tekrar indirme**, hemen path döndür

### Boyut & Target
- **41188 satır × 21 sütun**
- Target: `y` (`yes` / `no`) — müşteri vadeli mevduat hesabı açtı mı?
- Class dağılımı:
  - `no`: ~36548 (%88.7)
  - `yes`: ~4640 (%11.3) ← **azınlık sınıf**
- **Dengesiz** — naive accuracy yanıltıcı. PR + AUC kullan.

### CSV Formatı
- Separator: `;` (noktalı virgül, virgül DEĞİL)
- `pd.read_csv(path, sep=';')` ile oku.

### Feature Dictionary (20 feature)

**Kategorik (10):**
| Feature | Açıklama | Örnek Değerler |
|---|---|---|
| `job` | meslek | admin., blue-collar, entrepreneur, ... |
| `marital` | medeni durum | married, single, divorced |
| `education` | eğitim | basic.4y, high.school, university.degree, ... |
| `default` | kredi kart borcu var mı | yes, no, unknown |
| `housing` | konut kredisi var mı | yes, no, unknown |
| `loan` | kişisel kredi var mı | yes, no, unknown |
| `contact` | iletişim tipi | cellular, telephone |
| `month` | son iletişim ayı | jan, feb, ..., dec |
| `day_of_week` | son iletişim günü | mon, tue, wed, thu, fri |
| `poutcome` | önceki kampanya sonucu | success, failure, nonexistent |

**Numerik (10):**
| Feature | Açıklama |
|---|---|
| `age` | müşteri yaşı |
| `duration` | son çağrı süresi (sn) — ⚠️ **LEAKAGE** |
| `campaign` | bu kampanyada yapılan çağrı sayısı |
| `pdays` | önceki kampanyadan beri geçen gün (999 = hiç) |
| `previous` | önceki kampanyalardaki çağrı sayısı |
| `emp.var.rate` | istihdam değişim oranı (çeyreklik) |
| `cons.price.idx` | tüketici fiyat endeksi (aylık) |
| `cons.conf.idx` | tüketici güven endeksi (aylık) |
| `euribor3m` | euribor 3-aylık faiz |
| `nr.employed` | istihdam edilen sayı (binler) |

### Örnek Satır (3 örnek)
```
age | job          | marital  | education          | ... | y
56  | housemaid    | married  | basic.4y           | ... | no
57  | services     | married  | high.school        | ... | no
37  | services     | married  | high.school        | ... | yes
```

### Domain Notu
Veriler **2008-2010 arası bir Portekiz bankasından** toplanmış. O dönemde Avrupa borç krizi var, faiz oranları (`euribor3m`) ve istihdam göstergeleri (`emp.var.rate`, `nr.employed`) modelin önemli sinyalleri. Yani makroekonomik değişkenler **özellik mühendisliği fırsatı** sunuyor.

### ⚠️ Target Leakage — `duration` Özelliği

`duration` (son çağrı süresi, saniye) **çağrı bittikten sonra ölçülür**. Yani modeli **gerçek production'da** çağrı yapmadan önce çalıştırmaya kalktığında bu değer **bilinmez**. Modelin trainde mükemmel performans alır ama production'da işe yaramaz — buna **target leakage** denir.

UCI'nin orijinal dökümanı bile bunu söylüyor: *"This attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed... should be discarded if the intention is to have a realistic predictive model."*

Bu yüzden `drop_leakage_features()` fonksiyonun `duration` sütununu **siler**.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **16 fonksiyonu** sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`fetch_bank_data(cache_dir='data')`** — URL'den zip indir, nested zip aç, CSV path döndür (cache'li)
2. **`load_bank_data(path)`** — `pd.read_csv(path, sep=';')`
3. **`explore_data(df)`** — shape, class dağılımı, positive rate, kategorik/numerik kolonlar
4. **`drop_leakage_features(df)`** — `duration` sil
5. **`encode_target(df)`** — `y`: yes→1, no→0
6. **`prepare_features(df)`** — one-hot encode (drop_first=True), X ve y dön
7. **`split_data(X, y)`** — 80/20, stratify=y, random_state=42
8. **`build_balanced_pipeline()`** — StandardScaler + LogisticRegression(class_weight='balanced')
9. **`train_model(pipeline, X_train, y_train)`** — fit + dön
10. **`compute_roc_metrics(pipeline, X_test, y_test)`** — fpr, tpr, thresholds, AUC
11. **`compute_pr_metrics(pipeline, X_test, y_test)`** — precision, recall, AP, baseline (y_test.mean())
12. **`find_best_threshold_f1(pipeline, X_test, y_test)`** — PR eğrisi üzerinde F1 max
13. **`evaluate_at_threshold(pipeline, X_test, y_test, threshold)`** — accuracy, precision, recall, f1, confusion matrix
14. **`predict_customer(pipeline, customer_features, threshold=0.5)`** — tek müşteri için tahmin + olasılık
15. **`compare_with_without_balance(X_train, X_test, y_train, y_test)`** — class_weight='balanced' vs None karşılaştır
16. **`run_pipeline()`** — uçtan uca akış, özet dict dön

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] Gerçek dünyada **veri çekme** (URL + zip + cache) yapabileceksin
- [x] **Class imbalance** tespit edip uygun metrik seçebileceksin
- [x] **One-hot encoding** ile yüksek kardinaliteli kategorik feature'ları işleyebileceksin
- [x] **LogisticRegression** + `class_weight='balanced'` etkisini ölçebileceksin
- [x] **ROC Curve & AUC** hesaplayabileceksin
- [x] **PR Curve & Average Precision** hesaplayabileceksin
- [x] **Dengesizde ROC vs PR farkını** anlayabileceksin (PR daha dürüst)
- [x] **Threshold tuning** ile F1'i optimize edebileceksin
- [x] **Target leakage** tespit edip elimine edebileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (17 test)

Tümü pass olmalı:
- Dataset fetch + nested zip açma (internet gerekli, ilk test ~5MB indirir)
- Class dağılımı doğru tespit edilmiş mi
- `duration` silinmiş mi
- One-hot encoding sonrası kolon sayısı artmış mı
- Stratified split korunmuş mu
- Pipeline tipi doğru mu
- AUC > 0.7 (model işe yarıyor)
- Average Precision > baseline * 2 (PR'da baseline'ı geç)
- `class_weight='balanced'` recall'ı artırmış mı (vs default)

## 📊 Beklenen Sonuçlar

```
Positive rate: ~%11.3
AUC: ~0.78-0.82 (LogReg + duration'sız feature'lar)
Average Precision: ~0.35-0.45 (baseline ~0.11)
Best F1 threshold: ~0.4-0.5 arası (default 0.5'ten farklı olabilir)
Balanced vs Default recall farkı: pozitif (balanced çok daha fazla pozitif yakalar)
```

## 💡 İpuçları

- **İlk testte internet** gerekli (zip indirme). Sonraki testler cache'den okur.
- `pd.read_csv(path, sep=';')` — `;` ayırıcısını unutma
- One-hot encoding sonrası **sütun sayısı artar** (~20 → ~50+)
- `class_weight='balanced'` — minoritye otomatik daha fazla ağırlık verir
- ROC AUC dengesizde **iyimser** görünür → PR-AUC daha dürüsttür
- F1 max threshold'u **0.5 değildir** — class imbalance'ta optimum kayar
- `roc_curve`, `precision_recall_curve`, `roc_auc_score`, `average_precision_score` — hepsi `sklearn.metrics`

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olur)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- `data/` klasörü repo'ya **gitmez** (.gitignore'da exclude — büyük dosya)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
