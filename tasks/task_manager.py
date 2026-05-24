"""
DS-33 — Bank Marketing (Imbalanced Classification)
Tele-pazarlama kampanyasında hangi müşterilerin vadeli mevduat hesabı açacağını
tahmin edeceksin. Veri dengesiz (%11 pozitif) — bu yüzden ROC + PR + threshold
tuning kritik.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""


# 1. UCI Bank Marketing veri setini indir (cache'li)
def fetch_bank_data(cache_dir='data'):
    """
    UCI'den bank+marketing.zip indir, nested zip'i aç, CSV path döndür.

    Akış:
    1. cache_dir mevcut değilse oluştur (os.makedirs(cache_dir, exist_ok=True))
    2. Hedef CSV path: cache_dir/bank-additional/bank-additional-full.csv
    3. Bu dosya zaten varsa → hemen path'i döndür (cache hit)
    4. Yoksa:
       a. URL'den zip indir: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
          (urllib.request.urlretrieve kullan)
       b. Zip'i cache_dir'a aç (zipfile.ZipFile + extractall)
       c. İçinde bank-additional.zip var — onu da cache_dir'a aç (nested zip)
       d. Artık cache_dir/bank-additional/bank-additional-full.csv hazır
    5. CSV path'ini döndür

    Args:
        cache_dir: lokal cache klasörü (default 'data')

    Returns:
        str: CSV dosyasının tam yolu

    İpucu: import os, urllib.request, zipfile
    URL: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
    """
    pass


# 2. CSV'yi DataFrame olarak yükle
def load_bank_data(path):
    """
    CSV'yi oku. DİKKAT: separator ';' (noktalı virgül), virgül DEĞİL.

    Args:
        path: CSV dosya yolu (fetch_bank_data'dan dönen)

    Returns:
        pd.DataFrame: 41188 satır × 21 sütun
    """
    pass


# 3. Veriyi keşfet — class dağılımı, feature tipleri
def explore_data(df):
    """
    Temel keşif metriği üret.

    Returns:
        dict: {
            'shape': tuple (rows, cols),
            'class_distribution': {'no': int, 'yes': int},
            'positive_rate': float (yes / total),
            'categorical_cols': list (dtype='object' sütunlar — 'y' HARİÇ),
            'numeric_cols': list ('y' HARİÇ numerik sütunlar)
        }

    İpucu:
    - df['y'].value_counts() → 'no' ve 'yes' sayıları
    - df.select_dtypes(include='object').columns.tolist() → kategorik
    - df.select_dtypes(include='number').columns.tolist() → numerik
    - 'y' her iki listeden çıkartılmalı
    """
    pass


# 4. Target leakage'lı feature'ları sil
def drop_leakage_features(df):
    """
    'duration' sütunu çağrı bittikten sonra ölçülür → leakage.
    Production'da kullanılamaz, eğitimden önce silinmeli.

    Args:
        df: ham DataFrame

    Returns:
        pd.DataFrame: 'duration' kolonu olmayan kopya

    İpucu: df.drop(columns=['duration']) — orijinali mutate etme
    """
    pass


# 5. Target'ı encode et (yes/no → 1/0)
def encode_target(df):
    """
    'y' sütununu binary'ye çevir: 'yes' → 1, 'no' → 0.

    Args:
        df: 'y' sütunu olan DataFrame

    Returns:
        pd.DataFrame: 'y' artık int (0/1)

    İpucu: df = df.copy(); df['y'] = df['y'].map({'yes': 1, 'no': 0})
    """
    pass


# 6. Feature'ları hazırla (one-hot encoding)
def prepare_features(df):
    """
    Kategorik feature'ları one-hot encode et. Numeric'lere dokunma.
    Target ('y') feature'lardan ayrıştır.

    Args:
        df: encode_target sonrası DataFrame (y artık 0/1)

    Returns:
        tuple: (X, y)
            X: pd.DataFrame (one-hot encoded features)
            y: pd.Series (0/1 target)

    İpucu:
    - y = df['y']
    - features = df.drop(columns=['y'])
    - X = pd.get_dummies(features, drop_first=True)
      (drop_first multicollinearity önler)
    """
    pass


# 7. Train/test split (stratified)
def split_data(X, y):
    """
    train_test_split kullan:
    - test_size=0.2
    - random_state=42 (tekrarlanabilirlik)
    - stratify=y (sınıf dengesi train ve test'te korunsun)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    pass


# 8. Balanced pipeline kur
def build_balanced_pipeline():
    """
    sklearn Pipeline:
    - 'scaler': StandardScaler()
    - 'lr': LogisticRegression(max_iter=2000, class_weight='balanced',
                                random_state=42)

    class_weight='balanced' → minoritye otomatik daha fazla ağırlık verir
    max_iter=2000 → convergence için (default 100 yetmez)

    Returns:
        sklearn.pipeline.Pipeline
    """
    pass


# 9. Modeli eğit
def train_model(pipeline, X_train, y_train):
    """
    Pipeline'ı fit et ve döndür.

    Returns:
        Pipeline: fit edilmiş pipeline
    """
    pass


# 10. ROC metriklerini hesapla
def compute_roc_metrics(pipeline, X_test, y_test):
    """
    ROC eğrisi ve AUC hesapla.

    Returns:
        dict: {
            'fpr': np.array (false positive rate),
            'tpr': np.array (true positive rate),
            'thresholds': np.array,
            'auc': float (0-1 arası)
        }

    İpucu:
    - y_proba = pipeline.predict_proba(X_test)[:, 1]  # pozitif sınıf olasılığı
    - fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    - auc = roc_auc_score(y_test, y_proba)
    """
    pass


# 11. PR (Precision-Recall) metriklerini hesapla
def compute_pr_metrics(pipeline, X_test, y_test):
    """
    Precision-Recall eğrisi + Average Precision + baseline hesapla.

    Returns:
        dict: {
            'precision': np.array,
            'recall': np.array,
            'thresholds': np.array,
            'average_precision': float,
            'baseline': float (y_test.mean() — PR'ın rastgele referansı)
        }

    Baseline neden y_test.mean()? Çünkü PR eğrisinde rastgele bir model'in
    AP'si pozitif sınıf oranına eşittir (bizim case: ~0.11).

    İpucu:
    - y_proba = pipeline.predict_proba(X_test)[:, 1]
    - precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    - ap = average_precision_score(y_test, y_proba)
    """
    pass


# 12. F1'i maksimize eden threshold'u bul
def find_best_threshold_f1(pipeline, X_test, y_test):
    """
    PR eğrisi üzerinde her threshold için F1 hesapla, en iyiyi bul.

    F1 = 2 * (precision * recall) / (precision + recall)

    Returns:
        dict: {
            'best_threshold': float,
            'best_f1': float,
            'precision_at_best': float,
            'recall_at_best': float
        }

    İpucu:
    - precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    - precision/recall len = len(thresholds) + 1 → son elemanı atla
    - f1 = 2 * p * r / (p + r + 1e-12)  # epsilon: 0'a bölme koruması
    - np.argmax(f1) → en iyi index
    """
    pass


# 13. Belirli threshold'da modeli değerlendir
def evaluate_at_threshold(pipeline, X_test, y_test, threshold):
    """
    Verilen threshold'da accuracy, precision, recall, F1, confusion matrix
    hesapla.

    Default threshold 0.5 — ama dengesizde optimum farklı olabilir.

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': np.array (2x2)
        }

    İpucu:
    - y_proba = pipeline.predict_proba(X_test)[:, 1]
    - y_pred = (y_proba >= threshold).astype(int)
    - sklearn.metrics'ten: accuracy_score, precision_score, recall_score,
      f1_score, confusion_matrix
    """
    pass


# 14. Tek bir müşteri için tahmin yap
def predict_customer(pipeline, customer_features, threshold=0.5):
    """
    Bir müşterinin feature'larını al, tahmin + olasılık döndür.

    Args:
        pipeline: eğitilmiş pipeline
        customer_features: pd.DataFrame (1 satır, X_train ile aynı kolonlar)
        threshold: karar eşiği (default 0.5)

    Returns:
        dict: {
            'predicted': int (0 veya 1),
            'probability': float (pozitif sınıf olasılığı),
            'will_subscribe': bool (predicted == 1)
        }

    İpucu:
    - proba = pipeline.predict_proba(customer_features)[0, 1]
    - predicted = int(proba >= threshold)
    """
    pass


# 15. Balanced vs Unbalanced karşılaştırması
def compare_with_without_balance(X_train, X_test, y_train, y_test):
    """
    İki model eğit:
    - Balanced:   class_weight='balanced'
    - Unbalanced: class_weight=None (default)

    Her ikisini de default threshold (0.5) ile değerlendir.

    Returns:
        dict: {
            'balanced':   {'recall':, 'precision':, 'f1':, 'auc':},
            'unbalanced': {'recall':, 'precision':, 'f1':, 'auc':}
        }

    Beklenti: 'balanced' recall'ı çok daha yüksek, precision biraz daha düşük.
    Çünkü minoritye fazla ağırlık verince daha fazla pozitif yakalıyor
    ama yanlış pozitif de artıyor.

    İpucu: İki ayrı pipeline kur, fit et, metrikleri hesapla.
    """
    pass


# 16. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Uçtan uca akış:
    1. fetch_bank_data → load_bank_data
    2. drop_leakage_features → encode_target → prepare_features
    3. split_data
    4. build_balanced_pipeline → train_model
    5. compute_roc_metrics, compute_pr_metrics, find_best_threshold_f1
    6. compare_with_without_balance

    Returns:
        dict: {
            'positive_rate': float,
            'auc': float,
            'average_precision': float,
            'best_threshold_f1': float,
            'best_f1': float,
            'balanced_vs_unbalanced_recall_diff': float
                (balanced.recall - unbalanced.recall, pozitif olmalı)
        }
    """
    pass


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Positive rate           : {result['positive_rate']:.2%}")
    print(f"  AUC                     : {result['auc']:.4f}")
    print(f"  Average Precision (AP)  : {result['average_precision']:.4f}")
    print(f"  Best F1 threshold       : {result['best_threshold_f1']:.4f}")
    print(f"  Best F1                 : {result['best_f1']:.4f}")
    print(f"  Recall lift (balanced)  : +{result['balanced_vs_unbalanced_recall_diff']:.4f}")
