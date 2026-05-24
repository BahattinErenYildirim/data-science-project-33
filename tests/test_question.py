import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    fetch_bank_data, load_bank_data, explore_data, drop_leakage_features,
    encode_target, prepare_features, split_data, build_balanced_pipeline,
    train_model, compute_roc_metrics, compute_pr_metrics,
    find_best_threshold_f1, evaluate_at_threshold, predict_customer,
    compare_with_without_balance, run_pipeline,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar indirme/eğitme yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def csv_path():
    """İlk testte internet'ten indirir, sonraki tüm testler cache'den okur."""
    return fetch_bank_data()


@pytest.fixture(scope="module")
def raw_df(csv_path):
    return load_bank_data(csv_path)


@pytest.fixture(scope="module")
def prepared(raw_df):
    df = drop_leakage_features(raw_df)
    df = encode_target(df)
    X, y = prepare_features(df)
    return X, y


@pytest.fixture(scope="module")
def split(prepared):
    X, y = prepared
    return split_data(X, y)


@pytest.fixture(scope="module")
def trained_pipeline(split):
    X_train, X_test, y_train, y_test = split
    pipe = build_balanced_pipeline()
    return train_model(pipe, X_train, y_train)


# 1. fetch_bank_data
def test_fetch_bank_data_returns_path(csv_path):
    assert isinstance(csv_path, str)
    assert os.path.exists(csv_path)
    assert csv_path.endswith('bank-additional-full.csv')


# 2. load_bank_data
def test_load_bank_data_shape(raw_df):
    assert isinstance(raw_df, pd.DataFrame)
    assert raw_df.shape == (41188, 21)
    assert 'y' in raw_df.columns


# 3. explore_data
def test_explore_data_structure(raw_df):
    info = explore_data(raw_df)
    assert set(info.keys()) >= {
        'shape', 'class_distribution', 'positive_rate',
        'categorical_cols', 'numeric_cols'
    }
    assert info['shape'] == (41188, 21)
    assert info['class_distribution']['no'] > info['class_distribution']['yes']
    assert 0.10 < info['positive_rate'] < 0.13  # ~11.3%
    assert 'y' not in info['categorical_cols']
    assert 'y' not in info['numeric_cols']


# 4. drop_leakage_features
def test_drop_leakage_features_removes_duration(raw_df):
    df = drop_leakage_features(raw_df)
    assert 'duration' not in df.columns
    assert df.shape[1] == raw_df.shape[1] - 1
    # Orijinal mutate edilmemiş
    assert 'duration' in raw_df.columns


# 5. encode_target
def test_encode_target_binary(raw_df):
    df = encode_target(raw_df)
    assert set(df['y'].unique()) == {0, 1}
    assert df['y'].dtype in (np.int64, np.int32, int)


# 6. prepare_features
def test_prepare_features_onehot(raw_df):
    df = drop_leakage_features(raw_df)
    df = encode_target(df)
    X, y = prepare_features(df)
    # One-hot encoding sonrası sütun sayısı artmalı
    assert X.shape[1] > 19
    # Hiçbir 'object' sütun kalmamalı
    obj_cols = X.select_dtypes(include='object').columns
    assert len(obj_cols) == 0
    # y target olarak ayrılmış
    assert 'y' not in X.columns
    assert len(y) == len(X)


# 7. split_data
def test_split_data_stratified(prepared):
    X, y = prepared
    X_train, X_test, y_train, y_test = split_data(X, y)
    # 80/20
    assert abs(len(X_train) / len(X) - 0.8) < 0.01
    # Stratify: train ve test positive rate yakın olmalı
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.005


# 8. build_balanced_pipeline
def test_build_balanced_pipeline_type():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pipe = build_balanced_pipeline()
    assert isinstance(pipe, Pipeline)
    # Step isimleri 'scaler' ve 'lr'
    names = dict(pipe.steps)
    assert 'scaler' in names and 'lr' in names
    assert isinstance(names['scaler'], StandardScaler)
    assert isinstance(names['lr'], LogisticRegression)
    assert names['lr'].class_weight == 'balanced'


# 9. train_model
def test_train_model_fits(trained_pipeline, split):
    X_train, X_test, y_train, y_test = split
    # Eğitilmiş pipeline predict yapabilmeli
    preds = trained_pipeline.predict(X_test[:5])
    assert len(preds) == 5


# 10. compute_roc_metrics
def test_compute_roc_metrics(trained_pipeline, split):
    _, X_test, _, y_test = split
    roc = compute_roc_metrics(trained_pipeline, X_test, y_test)
    assert set(roc.keys()) >= {'fpr', 'tpr', 'thresholds', 'auc'}
    assert 0 <= roc['auc'] <= 1
    assert roc['auc'] > 0.7  # Model işe yarıyor
    assert len(roc['fpr']) == len(roc['tpr'])


# 11. compute_pr_metrics
def test_compute_pr_metrics(trained_pipeline, split):
    _, X_test, _, y_test = split
    pr = compute_pr_metrics(trained_pipeline, X_test, y_test)
    assert set(pr.keys()) >= {
        'precision', 'recall', 'thresholds', 'average_precision', 'baseline'
    }
    # Baseline ≈ positive rate (~0.11)
    assert 0.10 < pr['baseline'] < 0.13
    # AP baseline'dan çok daha iyi olmalı
    assert pr['average_precision'] > pr['baseline'] * 2


# 12. find_best_threshold_f1
def test_find_best_threshold_f1(trained_pipeline, split):
    _, X_test, _, y_test = split
    out = find_best_threshold_f1(trained_pipeline, X_test, y_test)
    assert set(out.keys()) >= {
        'best_threshold', 'best_f1', 'precision_at_best', 'recall_at_best'
    }
    assert 0 < out['best_threshold'] < 1
    assert 0 < out['best_f1'] < 1


# 13. evaluate_at_threshold
def test_evaluate_at_threshold(trained_pipeline, split):
    _, X_test, _, y_test = split
    res = evaluate_at_threshold(trained_pipeline, X_test, y_test, threshold=0.5)
    assert set(res.keys()) >= {
        'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'
    }
    assert 0 <= res['accuracy'] <= 1
    assert res['confusion_matrix'].shape == (2, 2)


# 14. predict_customer
def test_predict_customer(trained_pipeline, split):
    _, X_test, _, _ = split
    sample = X_test.iloc[[0]]  # 1-satır DataFrame
    out = predict_customer(trained_pipeline, sample, threshold=0.5)
    assert set(out.keys()) >= {'predicted', 'probability', 'will_subscribe'}
    assert out['predicted'] in (0, 1)
    assert 0 <= out['probability'] <= 1
    assert isinstance(out['will_subscribe'], bool)


# 15. compare_with_without_balance
def test_compare_with_without_balance(split):
    X_train, X_test, y_train, y_test = split
    cmp = compare_with_without_balance(X_train, X_test, y_train, y_test)
    assert 'balanced' in cmp and 'unbalanced' in cmp
    for label in ('balanced', 'unbalanced'):
        assert set(cmp[label].keys()) >= {'recall', 'precision', 'f1', 'auc'}
    # class_weight='balanced' recall'ı arttırmalı
    assert cmp['balanced']['recall'] > cmp['unbalanced']['recall']


# 16. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    assert set(result.keys()) >= {
        'positive_rate', 'auc', 'average_precision',
        'best_threshold_f1', 'best_f1',
        'balanced_vs_unbalanced_recall_diff'
    }
    assert result['auc'] > 0.7
    assert result['average_precision'] > result['positive_rate'] * 2
    assert result['balanced_vs_unbalanced_recall_diff'] > 0


# 17. End-to-end sanity (positive rate within expected range)
def test_run_pipeline_positive_rate():
    result = run_pipeline()
    assert 0.10 < result['positive_rate'] < 0.13


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
