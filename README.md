# Ứng dụng hỗ trợ khử sương mù ảnh số

Đề tài đồ án/khoá luận tốt nghiệp — triển khai ứng dụng khử sương mù ảnh số
bằng các phương pháp **truyền thống / prior-based**: **DCP**, **CLAHE**, **CAP**,
và pipeline lai **Hybrid (CLAHE + DCP)**.

## 1. Cấu trúc thư mục

```
dehaze_app/
├── main.py                 # Entry point chạy GUI
├── requirements.txt
├── README.md
├── data/
│   └── SOTS/
│       ├── indoor/{hazy, gt}/
│       └── outdoor/{hazy, gt}/
├── results/                # Kết quả benchmark
└── src/
    ├── algorithms/
    │   ├── dcp.py          # Dark Channel Prior
    │   ├── clahe.py        # CLAHE baseline
    │   ├── cap.py          # Color Attenuation Prior
    │   └── hybrid.py       # CLAHE + DCP
    ├── metrics/
    │   ├── psnr_ssim.py    # Full-reference
    │   ├── niqe.py         # No-reference (cần pyiqa)
    │   └── edge_entropy.py # Entropy + edge visibility
    ├── utils/
    │   ├── io.py           # Đọc/ghi ảnh, ghép cặp SOTS
    │   └── guided_filter.py
    ├── gui/
    │   └── main_window.py  # GUI PyQt5
    └── benchmark.py        # Script chạy toàn bộ SOTS
```

## 2. Cài đặt

```bash
# Khuyến nghị dùng venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Cài thêm (tuỳ chọn) nếu muốn đo **NIQE**:

```bash
pip install pyiqa torch
```

## 3. Chuẩn bị dữ liệu SOTS

Tải bộ **RESIDE-SOTS** (Synthetic Objective Testing Set) và giải nén vào
`data/SOTS/` với cấu trúc:

```
data/SOTS/
├── indoor/
│   ├── hazy/   # ví dụ 1400_1.png, 1400_2.png, ...
│   └── gt/     # ví dụ 1400.png, ...
└── outdoor/
    ├── hazy/
    └── gt/
```

Lưu ý tên file: hazy có dạng `<id>_xxx.png`, gt chỉ có `<id>.png`. Module
`src/utils/io.py` ghép cặp tự động theo prefix trước dấu `_`.

## 4. Sử dụng

### 4.1 Chạy GUI

```bash
python main.py
```

Trong GUI:

1. Nhấn **📂 Mở ảnh hazy** (hoặc kéo-thả ảnh vào cửa sổ).
2. (Tuỳ chọn) Mở **Ground Truth** tương ứng → app tự tính PSNR/SSIM.
3. Chọn thuật toán DCP / CLAHE / CAP / Hybrid, điều chỉnh tham số.
4. Nhấn **▶ Khử sương mù** → xem kết quả song song với ảnh gốc.
5. Nhấn **💾 Lưu kết quả** để xuất ảnh.

### 4.2 Benchmark trên toàn bộ SOTS

```bash
python -m src.benchmark \
    --sots-root data/SOTS \
    --out-dir results \
    --algorithms DCP CLAHE CAP Hybrid \
    --save-images
```

Tham số:

- `--limit N`: chỉ chạy N ảnh đầu mỗi subset (test nhanh).
- `--save-images`: lưu ảnh kết quả (tốn ổ cứng).

Sau khi chạy xong, kết quả nằm ở:

- `results/metrics.csv` — metric từng ảnh.
- `results/summary.csv` — trung bình theo (subset, algorithm).
- `results/images/<subset>/<algo>/…png` — ảnh đã khử sương mù.

## 5. Ý nghĩa các chỉ số

| Chỉ số          | Loại           | Ý nghĩa                           | Xu hướng tốt                |
| --------------- | -------------- | --------------------------------- | --------------------------- |
| PSNR            | Full-reference | Sai khác cường độ so với GT       | Càng cao                    |
| SSIM            | Full-reference | Tương đồng cấu trúc với GT        | Càng gần 1                  |
| NIQE            | No-reference   | Chất lượng tự nhiên, không cần GT | Càng thấp                   |
| Entropy         | No-reference   | Độ giàu thông tin của ảnh         | Càng cao (trong mức hợp lý) |
| Edge visibility | No-reference   | Tỷ lệ cạnh nhìn thấy              | Càng cao                    |

## 6. Gợi ý viết báo cáo

- **Chương 1**: Tổng quan bài toán, ứng dụng thực tế.
- **Chương 2**: Mô hình Atmospheric Scattering, chi tiết DCP/CLAHE/CAP,
  ưu/nhược của mỗi phương pháp. Dùng code trong `src/algorithms/` làm minh hoạ.
- **Chương 3**: Kiến trúc hệ thống, Use Case diagram, thiết kế GUI
  (chụp màn hình từ `main.py`).
- **Chương 4**: Thực nghiệm — chạy `src/benchmark.py`, dùng `metrics.csv`
  dựng biểu đồ so sánh (matplotlib). Phân tích theo indoor/outdoor,
  trade-off chất lượng ↔ tốc độ.

## 7. Hướng phát triển (đã nêu trong đề cương)

- DCP với cửa sổ thích nghi (patch size thay đổi theo vùng).
- Pipeline Hybrid với post-processing cân bằng trắng.
- Tối ưu tốc độ bằng vectorization / Numba / CUDA.
- Bổ sung thêm chỉ số FADE (Fog Aware Density Evaluator).

## 8. Giấy phép

Mã nguồn chỉ phục vụ mục đích học thuật, luận văn tốt nghiệp.
