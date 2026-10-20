# Data eksperimen

German Credit diambil saat OpenML tersedia. Bila koneksi tidak tersedia, data cadangan sintetis dibuat saat runtime menggunakan `sklearn.datasets.make_classification` dengan konfigurasi `fallback_samples`, `minority_class_weight`, fitur, dan seed pada `config.yaml`. Tidak ada klaim bahwa file sintetis ini merupakan catatan nasabah nyata. File hasil laporan menyimpan `source_used` dan metadata agar sumber yang benar-benar dipakai dapat dibedakan.
