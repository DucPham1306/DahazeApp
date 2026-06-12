# Ứng dụng Khử Sương Mù Ảnh Số

Đồ án / khoá luận tốt nghiệp — xây dựng ứng dụng khử sương mù (image dehazing) sử dụng các phương pháp truyền thống dựa trên prior:

| Thuật toán | Tên đầy đủ | Mô tả ngắn |
|---|---|---|
| **DCP** | Dark Channel Prior | Ước lượng transmission map dựa trên kênh tối nhất |
| **DCP Improved** | Dark Channel Prior (cải tiến) | Kết hợp guided filter để làm mịn transmission map |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization | Tăng cường tương phản cục bộ |
| **Hybrid** | CLAHE + DCP | Pipeline kết hợp: CLAHE tiền xử lý, DCP khử sương mù |

---

## Yêu cầu hệ thống

- Python **3.9+**
- Hệ điều hành: Windows / Linux / macOS
- RAM: tối thiểu 4 GB (khuyến nghị 8 GB khi chạy benchmark toàn bộ SOTS)

---

## 1. Cài đặt

**Bước 1 — Tạo môi trường ảo (khuyến nghị):**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**Bước 2 — Cài các thư viện cần thiết:**

```bash
pip install -r requirements.txt
```

**Bước 3 (tuỳ chọn) — Cài thêm nếu muốn tính chỉ số NIQE:**

```bash
pip install pyiqa>=0.1.10 torch>=2.0
```

> Nếu không cài `pyiqa`, các chỉ số còn lại (PSNR, SSIM, Entropy, Edge visibility) vẫn hoạt động bình thường.

---

## 2. Chuẩn bị dữ liệu SOTS

Tải bộ **RESIDE-SOTS** (Synthetic Objective Testing Set) tại [RESIDE Benchmark](https://sites.google.com/view/reside-dehaze-datasets) và giải nén vào thư mục `data/SOTS/` theo đúng cấu trúc sau:

```
data/SOTS/
├── indoor/
│   ├── hazy/        # ảnh có sương mù,  ví dụ: 1400_1.png, 1400_2.png
│   └── gt/          # ảnh ground truth,  ví dụ: 1400.png
└── outdoor/
    ├── hazy/
    └── gt/
```

> **Lưu ý về tên file:** ảnh hazy có dạng `<id>_<số>.png`, ảnh GT chỉ có `<id>.png`. Module `src/utils/io.py` tự động ghép cặp theo phần prefix trước dấu `_`.

---

## 3. Cấu trúc thư mục

```
DehazeApp/
├── main.py                        # Entry point — khởi chạy GUI
├── requirements.txt
├── README.md
├── data/
│   └── SOTS/
│       ├── indoor/{hazy, gt}/
│       └── outdoor/{hazy, gt}/
├── results/                       # Kết quả benchmark (sinh tự động)
│   ├── metrics.csv                # Metric từng ảnh
│   ├── summary.csv                # Trung bình theo (subset, algorithm)
│   └── images/<subset>/<algo>/    # Ảnh đã khử sương mù (nếu --save-images)
└── src/
    ├── algorithms/
    │   ├── dcp.py                 # Dark Channel Prior
    │   ├── dcp_improved.py        # DCP + guided filter cải tiến
    │   ├── clahe.py               # CLAHE baseline
    │   ├── cap.py                 # Color Attenuation Prior
    │   └── hybrid.py              # Pipeline Hybrid (CLAHE → DCP)
    ├── metrics/
    │   ├── psnr_ssim.py           # Full-reference: PSNR, SSIM
    │   ├── niqe.py                # No-reference: NIQE (cần pyiqa)
    │   ├── no_reference.py        # No-reference tổng hợp
    │   └── edge_entropy.py        # Entropy + edge visibility
    ├── utils/
    │   ├── io.py                  # Đọc/ghi ảnh, ghép cặp SOTS
    │   ├── datasets.py            # Dataset loader
    │   └── guided_filter.py       # Guided filter dùng cho DCP cải tiến
    ├── gui/
    │   ├── main_window.py         # Cửa sổ chính GUI (PyQt5)
    │   ├── benchmark_tab.py       # Tab chạy benchmark trong GUI
    │   └── style.py               # Stylesheet của giao diện
    └── benchmark.py               # Script benchmark dòng lệnh
```

---

## 4. Sử dụng

### 4.1 Chạy giao diện đồ hoạ (GUI)

```bash
python main.py
```

Các bước thao tác trong GUI:

1. Nhấn **Mở ảnh hazy** (hoặc kéo–thả file ảnh vào cửa sổ).
2. *(Tuỳ chọn)* Nhấn **Mở Ground Truth** — app sẽ tự động tính PSNR/SSIM so sánh.
3. Chọn thuật toán: `DCP` / `DCP Improved` / `CLAHE` / `Hybrid`.
4. Điều chỉnh tham số nếu cần, rồi nhấn **Khử sương mù**.
5. Xem kết quả hiển thị song song với ảnh gốc.
6. Nhấn **Lưu kết quả** để xuất ảnh ra file.
7. Chuyển sang tab **Benchmark** để chạy đánh giá hàng loạt ngay trong GUI.

### 4.2 Chạy benchmark từ dòng lệnh

```bash
python -m src.benchmark \
    --sots-root data/SOTS \
    --out-dir   results \
    --algorithms DCP DCP_Improved CLAHE Hybrid \
    --save-images
```

Các tham số quan trọng:

| Tham số | Mô tả |
|---|---|
| `--sots-root` | Đường dẫn đến thư mục `SOTS/` |
| `--out-dir` | Thư mục lưu kết quả (mặc định: `results/`) |
| `--algorithms` | Danh sách thuật toán cần chạy |
| `--limit N` | Chỉ chạy N ảnh đầu mỗi subset (dùng để kiểm tra nhanh) |
| `--save-images` | Lưu ảnh kết quả ra file (tốn thêm dung lượng ổ cứng) |

Sau khi chạy xong, kết quả nằm trong `results/`:

- `metrics.csv` — metric của từng ảnh.
- `summary.csv` — trung bình theo cặp (subset, algorithm).
- `images/<subset>/<algo>/` — ảnh đã khử sương mù (nếu bật `--save-images`).

---

## 5. Ý nghĩa các chỉ số đánh giá

| Chỉ số | Loại | Ý nghĩa | Tốt khi |
|---|---|---|---|
| **PSNR** (dB) | Full-reference | Sai khác cường độ pixel so với GT | Càng **cao** càng tốt |
| **SSIM** | Full-reference | Tương đồng cấu trúc so với GT, thang 0–1 | Càng **gần 1** càng tốt |
| **NIQE** | No-reference | Chất lượng tự nhiên, không cần GT | Càng **thấp** càng tốt |
| **Entropy** | No-reference | Mức độ giàu thông tin của ảnh | Càng **cao** (trong mức hợp lý) |
| **Edge visibility** | No-reference | Tỷ lệ cạnh hiển thị rõ sau khử sương | Càng **cao** càng tốt |

> PSNR và SSIM yêu cầu ảnh ground truth. NIQE, Entropy, Edge visibility có thể dùng với ảnh thực tế không có GT.

---

## 6. Gợi ý cấu trúc báo cáo

- **Chương 1 — Giới thiệu:** Bài toán khử sương mù, ứng dụng thực tế (giao thông, giám sát, ảnh vệ tinh), phạm vi đồ án.
- **Chương 2 — Cơ sở lý thuyết:** Mô hình Atmospheric Scattering, nguyên lý của từng thuật toán (DCP, CLAHE, Hybrid), ưu và nhược điểm. Lấy code trong `src/algorithms/` làm minh hoạ.
- **Chương 3 — Thiết kế hệ thống:** Kiến trúc module, Use Case diagram, thiết kế giao diện (chụp màn hình từ `main.py`).
- **Chương 4 — Thực nghiệm:** Chạy `src/benchmark.py` trên SOTS, vẽ biểu đồ so sánh từ `metrics.csv` (dùng matplotlib/pandas). Phân tích kết quả theo indoor/outdoor, trade-off chất lượng ↔ tốc độ xử lý.

---

## 7. Hướng phát triển

- DCP với patch size thích nghi theo từng vùng ảnh.
- Tích hợp cân bằng trắng vào pipeline Hybrid.
- Tăng tốc bằng vectorization, Numba, hoặc CUDA.
- Bổ sung chỉ số FADE (Fog Aware Density Evaluator).
- Thêm phương pháp học sâu (DehazeNet, AOD-Net) để so sánh.

---

## 8. Giấy phép

Mã nguồn được phát triển phục vụ mục đích **học thuật** (đồ án / khoá luận tốt nghiệp). Không sử dụng cho mục đích thương mại.
