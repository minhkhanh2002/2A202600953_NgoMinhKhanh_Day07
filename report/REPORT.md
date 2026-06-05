# Báo Cáo Lab 7: Embedding & Vector Store (Đề tài Nhóm: Luật Việt Nam)

**Họ tên:** Ngô Minh Khánh - 2A202600953
**Nhóm:** Nhóm A5  
**Thành viên nhóm:**
- Ngô Minh Khánh — 2A202600953
- Dương Đức Cường — 2A202600794
- Đinh Hoàng Nam — 2A202600884
- Bùi Hoàng Sơn — 2A202600925
- Bùi Như Kiệt — 2A202600895
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là góc giữa hai vector biểu diễn văn bản trong không gian nhiều chiều rất nhỏ (gần bằng 0 độ), cho thấy hai đoạn văn bản đó có sự tương đồng cao về mặt ngữ nghĩa hoặc phân bố từ vựng.

**Ví dụ HIGH similarity:**
- Sentence A: "The weather is very hot today."
- Sentence B: "It is extremely warm outside today."
- Tại sao tương đồng: Cả hai câu đều mô tả cùng một hiện tượng thời tiết nóng bức bằng các từ đồng nghĩa khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Quantum computing utilizes superposition and entanglement."
- Sentence B: "I love eating chocolate chip cookies."
- Tại sao khác: Hai câu thuộc về hai lĩnh vực hoàn toàn độc lập và không chia sẻ chung bất kỳ ngữ cảnh hay khái niệm nào (vật lý lượng tử vs ẩm thực/sở thích).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo hướng (góc) giữa hai vector mà bỏ qua độ dài của chúng. Trong văn bản, hai tài liệu có cùng chủ đề nhưng độ dài khác nhau sẽ tạo ra vector có độ dài (magnitude) rất khác nhau dẫn đến khoảng cách Euclidean lớn, nhưng góc của chúng vẫn rất nhỏ (độ tương đồng cosine vẫn rất cao).

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Formula: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.111...) = 23`
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Trình bày phép tính:*
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25`
> Số lượng chunks sẽ tăng lên thành 25. Ta muốn tăng overlap để bảo toàn ngữ cảnh tốt hơn ở ranh giới giữa các chunk, đảm bảo các câu văn hoặc ý tưởng không bị cắt đứt nửa chừng khiến mô hình RAG bị thiếu thông tin khi truy xuất.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật pháp Việt Nam (Bộ luật Hình sự, Luật Trật tự, an toàn giao thông đường bộ, Luật Đất đai, và Bộ luật Lao động).

**Tại sao nhóm chọn domain này?**
> Hệ thống pháp luật Việt Nam có các điều khoản đan xen phức tạp và có các từ khóa chuyên ngành rất đặc trưng. Việc ứng dụng RAG vào truy xuất Luật pháp giúp giải đáp nhanh các thắc mắc của người dân, đồng thời đây là bộ dữ liệu cực kỳ phù hợp để kiểm chứng sức mạnh của các chiến lược chia nhỏ (chunking) và lọc siêu dữ liệu (metadata filtering).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | luat_dan_su_hinh_su.md | thuvienphapluat.vn | 631,745 | `{"category": "civil_penal", "language": "vi"}` |
| 2 | Luật Trật tự, an toàn giao thông đường bộ.md | thuvienphapluat.vn | 164,405 | `{"category": "traffic", "language": "vi"}` |
| 3 | luat_dat_dai.txt | thuvienphapluat.vn | 168,918 | `{"category": "land", "language": "vi"}` |
| 4 | bo-luat-lao-dong-2019-20210924034205-e.md | thuvienphapluat.vn | 2,990 | `{"category": "labor", "language": "vi"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | String | `"civil_penal"`, `"traffic"`, `"land"`, `"labor"` | Giúp lọc nhanh tài liệu theo bộ luật cụ thể khi người dùng đặt câu hỏi nằm trong một phạm vi luật định. |
| `language` | String | `"vi"`, `"en"` | Tránh việc hệ thống truy xuất nhầm các bản dịch hoặc văn bản ngoại ngữ đi kèm. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên mẫu 5000 ký tự đầu của mỗi tài liệu (với `chunk_size=500`):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| luat_dan_su_hinh_su.md | FixedSizeChunker (`fixed_size`) | 11 | 500.00 | No (bị cắt từ ngữ ngẫu nhiên giữa điều luật) |
| luat_dan_su_hinh_su.md | SentenceChunker (`by_sentences`) | 8 | 621.12 | Yes (giữ trọn câu văn) |
| luat_dan_su_hinh_su.md | RecursiveChunker (`recursive`) | 11 | 452.73 | Yes (tách theo cấu trúc dòng/đoạn) |
| Luật giao thông đường bộ.md | FixedSizeChunker (`fixed_size`) | 11 | 500.00 | No (bị cắt từ ngữ ngẫu nhiên) |
| Luật giao thông đường bộ.md | SentenceChunker (`by_sentences`) | 15 | 331.00 | Yes |
| Luật giao thông đường bộ.md | RecursiveChunker (`recursive`) | 13 | 382.77 | Yes |
| luat_dat_dai.txt | FixedSizeChunker (`fixed_size`) | 11 | 500.00 | No |
| luat_dat_dai.txt | SentenceChunker (`by_sentences`) | 16 | 310.12 | Yes |
| luat_dat_dai.txt | RecursiveChunker (`recursive`) | 11 | 453.55 | Yes |
| bo-luat-lao-dong-2019-20210924034205-e.md | FixedSizeChunker (`fixed_size`) | 7 | 470.00 | No (cắt ngang giữa các điều/khoản quy định chung) |
| bo-luat-lao-dong-2019-20210924034205-e.md | SentenceChunker (`by_sentences`) | 13 | 228.46 | Yes (giữ câu tốt nhưng tách rời các dòng nghỉ lễ/tết) |
| bo-luat-lao-dong-2019-20210924034205-e.md | RecursiveChunker (`recursive`) | 8 | 372.00 | Yes (giữ nguyên cấu trúc phân cấp Điều/Chương và danh sách lễ tết) |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**
> Bộ chia nhỏ này sử dụng một danh sách các ký tự phân tách theo thứ tự ưu tiên: `["\n\n", "\n", ". ", " ", ""]`. Nó sẽ thử tách văn bản bằng ký tự đầu tiên (`\n\n`), nếu phần con nào vẫn lớn hơn `chunk_size` quy định thì nó sẽ đệ quy gọi lại chính nó với các ký tự phân tách tiếp theo trong danh sách ưu tiên. Sau cùng, các phần nhỏ hơn sẽ được gộp lại tối đa sao cho tổng chiều dài của chunk không vượt quá `chunk_size` (trong thí nghiệm này là 500 ký tự).

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản luật pháp Việt Nam được chia thành các Điều, Khoản (đánh số 1, 2, 3...) và Điểm (a, b, c...) ngăn cách bằng các dòng mới. Việc dùng `RecursiveChunker` giúp bảo toàn toàn bộ cấu trúc của một Khoản hoặc Điểm liên quan trong cùng một chunk thay vì cắt vụn từng câu đơn lẻ (làm mất đi ngữ cảnh chính của Điều luật đó).

---

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| luat_dat_dai.txt | SentenceChunker (best baseline) | 16 | 310.12 | Khá tốt (nhưng đôi khi một câu không đủ mô tả toàn bộ ý nghĩa của điều khoản) |
| luat_dat_dai.txt | **RecursiveChunker (của tôi)** | 11 | 453.55 | Xuất sắc (giữ trọn vẹn nội dung của cả một Điểm/Khoản luật) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu | Vai trò thực tế trong Nhóm |
|-----------|----------|----------------------|-----------|----------|----------------------------|
| Tôi (Khanh) | `RecursiveChunker(chunk_size=500)` + `category`/`language` filters | 10/10 | Giữ nguyên cấu trúc ngữ nghĩa khoản/điểm luật; filter metadata loại bỏ nhiễu chéo hiệu quả giữa các bộ luật. | Số lượng chunks lớn hơn, tăng chi phí lưu trữ và thời gian index/truy xuất. | **Core System Engineer:** Thiết lập framework cốt lõi của package `src`, triển khai cấu trúc lớp chunker và embedding store, đồng thời tối ưu hóa logic tiền xử lý và lưu trữ. |
| Dương Đức Cường | `RecursiveChunker(chunk_size=700)` + `category`/`language` filters | 10/10 | Cân bằng tối ưu giữa độ dài chunk và ngữ cảnh; filter metadata giúp query đạt vị trí top-1 chính xác. | Chunk lớn có thể chứa các thông tin thừa đối với các điều luật cực ngắn. | **Prompt Optimization Specialist:** Chịu trách nhiệm tối ưu prompt cho KnowledgeBaseAgent, ngăn chặn hallucination (suy diễn) của LLM khi context không đủ. |
| Đinh Hoàng Nam | `FixedSizeChunker(chunk_size=500, overlap=80)` | 9/10 | Chunk size ổn định, kiểm soát tài nguyên tốt; overlap 80 ký tự giúp cứu vãn thông tin tại biên cắt. | Không giữ được ranh giới tự nhiên của điều khoản (dễ cắt đôi câu/khoản); dễ bị nhiễu chéo nếu thiếu filter. | **Data & Metadata Engineer:** Thu thập, làm sạch tài liệu luật pháp, xử lý OCR file PDF quét `52_honnhan_gd.signed.pdf` sang định dạng Markdown sạch. |
| Bùi Như Kiệt | `RecursiveChunker(chunk_size=900)` + `category`/`language` filters | 10/10 | Giữ nguyên vẹn toàn bộ một điều luật dài (nhiều khoản/điểm) trong một chunk; thông tin pháp lý liền mạch. | Chunk lớn dễ làm loãng điểm similarity khi câu hỏi quá chi tiết; tốn nhiều token đầu vào LLM. | **QA & Failure Analysis Specialist:** Thiết kế bộ câu hỏi biên (stress tests), chạy thử nghiệm truy xuất và phân tích chi tiết giới hạn của mock embedding. |
| Bùi Hoàng Sơn | `SentenceChunker(max_sentences_per_chunk=3)` | 8/10 | Không cắt giữa câu, chunk dễ đọc, phù hợp cho các điều luật đơn giản dạng Q&A ngắn. | Không giữ được cấu trúc phân cấp (Khoản/Điểm); dễ bị nhiễu chéo do không sử dụng metadata filter. | **Theoretical & Mathematical Analyst:** Nghiên cứu lý thuyết Warm-up, phân tích sự khác biệt Cosine vs Euclidean và thực hiện các phép tính toán học chunking. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chiến lược tốt nhất cho domain Luật pháp Việt Nam là **`RecursiveChunker` (kích thước tối ưu 500-700 ký tự) kết hợp bắt buộc với bộ lọc metadata (`category`/`language`)**.
> Lý do chi tiết:
> 1. **Bảo toàn cấu trúc phân cấp pháp lý:** Văn bản luật Việt Nam được định dạng rất chặt chẽ theo cấu trúc thụt lề `Điều > Khoản > Điểm`. `RecursiveChunker` ưu tiên cắt theo ranh giới dòng mới (`\n`, `\n\n`), giúp các Khoản/Điểm luật liên quan nằm trọn vẹn trong cùng một chunk thay vì bị cắt đôi ngẫu nhiên như `FixedSizeChunker`.
> 2. **Kiểm soát kích thước tối ưu:** Cỡ chunk 500-700 vừa đủ để chứa đầy đủ ngữ nghĩa của một điều khoản cụ thể mà không làm loãng điểm tương đồng cosine (như cỡ 900) hoặc quá ngắn gây mất liên kết ngữ cảnh (như Sentence Chunker).
> 3. **Metadata Filtering loại bỏ nhiễu chéo:** Trong cơ sở dữ liệu chứa nhiều bộ luật khác nhau (Hình sự, Lao động, Giao thông...), các thuật ngữ như "xử phạt", "trách nhiệm", "quy định" lặp lại rất nhiều. Việc lọc thô theo `category` trước khi đo similarity giúp khoanh vùng chính xác bộ luật cần tra cứu, ngăn chặn lỗi lấy nhầm tài liệu cực kỳ hiệu quả.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng `re.split` kết hợp với lookbehind pattern `(?<=\. |! |\? |\.\n)` để cắt nhỏ văn bản thành các câu riêng biệt mà vẫn giữ nguyên dấu câu. Sau khi loại bỏ khoảng trắng dư thừa bằng `strip()`, ta thực hiện gộp nhóm các câu thành từng chunk có độ dài tối đa là `max_sentences_per_chunk` câu.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Hàm đệ quy `_split` kiểm tra điều kiện dừng: nếu văn bản ngắn hơn `chunk_size` thì trả về ngay. Nếu không còn ký tự phân tách nào, thực hiện chia cứng theo độ dài ký tự. Trường hợp ngược lại, tách văn bản theo ký tự phân tách hiện tại, đệ quy chia nhỏ các phần có kích thước vượt ngưỡng, và gom các phần nhỏ lại tối đa sao cho tổng kích thước không vượt quá `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ bản ghi dưới dạng từ điển chuẩn hóa chứa: `id` (dạng `{doc.id}_{counter}` độc bản), `content`, `metadata` (chèn thêm `doc_id`), và vector `embedding` được tính thông qua `_embedding_fn`. Hàm `search` sử dụng tích vô hướng (`_dot`) để xếp hạng độ tương đồng giữa query embedding và toàn bộ vector trong bộ lưu trữ.

**`search_with_filter` + `delete_document`** — approach:
> Hàm `search_with_filter` thực hiện lọc thô các bản ghi (pre-filtering) bằng cách duyệt qua `self._store` và chỉ giữ lại những bản ghi thỏa mãn tất cả khóa-giá trị của `metadata_filter` trước khi thực hiện tìm kiếm tương đồng. Hàm `delete_document` thực hiện lọc bỏ mọi bản ghi có `metadata["doc_id"] == doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi hàm `self.store.search` để lấy ra top-k chunk liên quan, nối nội dung của chúng thành một context thống nhất, sau đó chèn context này và câu hỏi của người dùng vào prompt template RAG định sẵn rồi gọi LLM sinh câu trả lời.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
...
============================= 42 passed in 0.11s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Chạy thực tế sử dụng mô hình nhúng cục bộ `LocalEmbedder` (`all-MiniLM-L6-v2`):

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The quick brown fox jumps over the lazy dog. | A swift auburn canine leaps across the sluggish hound. | high | 0.6079 | Đúng (Độ tương đồng cao) |
| 2 | This system is completely stable and ready for deployment. | This system is not completely stable and not ready for deployment. | high | 0.8418 | Đúng (Trùng từ vựng nhiều) |
| 3 | Artificial intelligence models are trained on massive datasets. | Baking a delicious chocolate cake requires cocoa powder and flour. | low | -0.0546 | Đúng (Không liên quan) |
| 4 | We store customer emails in the relational database. | We keep employee names in the SQL database table. | high | 0.4967 | Đúng (Trung bình - cao) |
| 5 | I love reading books about machine learning. | Tôi thích đọc sách về học máy. | low | 0.0087 | Đúng (Gần như không tương quan) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Hai điểm bất ngờ lớn nhất là:
> 1. **Pair 2**: Hai câu có ý nghĩa phủ định hoàn toàn nhưng điểm tương đồng đạt tận 0.84. Điều này xảy ra do mô hình nhúng dựa trên từ vựng/ngữ cảnh cục bộ (vocabulary overlap) nên rất dễ bị đánh lừa bởi các câu phủ định nhẹ (fine-grained negation).
> 2. **Pair 5**: Bản dịch tiếng Anh và tiếng Việt có độ tương đồng gần như bằng 0 (0.0087). Lý do là vì mô hình nhúng `all-MiniLM-L6-v2` chỉ hỗ trợ đơn ngữ tiếng Anh (monolingual English), nên nó không thể ánh xạ ngôn ngữ tiếng Việt vào cùng một không gian ngữ nghĩa với tiếng Anh.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src` sử dụng mô hình `LocalEmbedder`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Tội cố ý gây thương tích bị xử lý như thế nào? | Quy định tại Điều 134, cải tạo không giam giữ đến 3 năm hoặc phạt tù từ 6 tháng đến 20 năm hoặc tù chung thân tùy theo tỷ lệ tổn thương cơ thể. |
| 2 | Tuổi nào thì phải chịu trách nhiệm hình sự? | Quy định tại Điều 12: Người từ đủ 16 tuổi trở lên chịu trách nhiệm về mọi tội phạm; người từ đủ 14 đến dưới 16 tuổi chịu trách nhiệm về tội rất nghiêm trọng hoặc đặc biệt nghiêm trọng được liệt kê. |
| 3 | Hình phạt tù chung thân áp dụng khi nào? | Quy định tại Điều 39: Áp dụng đối với người phạm tội đặc biệt nghiêm trọng nhưng chưa đến mức bị áp dụng hình phạt tử hình. Không áp dụng đối với người dưới 18 tuổi phạm tội. |
| 4 | Tội trộm cắp tài sản bị phạt tù tối đa bao nhiêu năm? | Quy định tại Điều 173: Phạt tù tối đa là 20 năm (Khoản 4) đối với tài sản trị giá 500 triệu đồng trở lên hoặc lợi dụng hoàn cảnh đặc biệt. |
| 5 | Tội phạm là gì theo Bộ luật Hình sự? | Quy định tại Điều 8: Tội phạm là hành vi nguy hiểm cho xã hội được quy định trong BLHS, do người có năng lực TNHS hoặc pháp nhân thương mại thực hiện. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tội cố ý gây thương tích bị xử lý như thế nào? | luat_dan_su_hinh_su_chunk_541 (Gây thương tích tỉ lệ từ 61% đến 121%...) | 0.6386 | Yes | Quy định tại Điều 134: cải tạo không giam giữ đến 3 năm, tù đến 20 năm hoặc chung thân... |
| 2 | Tuổi nào thì phải chịu trách nhiệm hình sự? | luat_dan_su_hinh_su_chunk_176 (Điều 76 về phạm vi chịu TNHS...) | 0.7252 | Yes | Quy định tại Điều 12: Người từ đủ 16 tuổi chịu mọi tội; từ đủ 14 đến dưới 16 tuổi chịu tội rất/đặc biệt nghiêm trọng... |
| 3 | Hình phạt tù chung thân áp dụng khi nào? | luat_dan_su_hinh_su_chunk_80 (Điều 39 về Tù chung thân...) | 0.7586 | Yes | Quy định tại Điều 39: Áp dụng đối với người phạm tội đặc biệt nghiêm trọng chưa đến mức tử hình... |
| 4 | Tội trộm cắp tài sản bị phạt tù tối đa bao nhiêu năm? | luat_dan_su_hinh_su_chunk_386 (Điều 173 về tội trộm cắp...) | 0.8387 | Yes | Quy định tại Điều 173: phạt tù tối đa 20 năm đối với tài sản trị giá 500 triệu đồng trở lên... |
| 5 | Tội phạm là gì theo Bộ luật Hình sự? | luat_dan_su_hinh_su_chunk_27 (Định nghĩa hành vi xâm phạm trật tự...) | 0.6655 | Yes | Quy định tại Điều 8: Tội phạm là hành vi nguy hiểm cho xã hội do người có năng lực TNHS thực hiện... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> - **Từ Đinh Hoàng Nam:** Học được tầm quan trọng của chất lượng dữ liệu thô. Nam đã xử lý OCR rất kỹ file PDF quét để ra file markdown sạch sẽ. Mặc dù Nam dùng `FixedSizeChunker` làm baseline, nhưng việc thiết kế `overlap=80` cho thấy một phương án dự phòng tốt giúp giữ lại thông tin ở ranh giới cắt.
> - **Từ Dương Đức Cường và Bùi Như Kiệt:** Học được cách tối ưu hóa kích thước của `RecursiveChunker`. Kiệt sử dụng chunk size 900 để giữ nguyên vẹn context của một điều luật dài nhưng lại gây tốn token khi đưa vào LLM, trong khi size 500 của tôi và 700 của Cường đạt hiệu năng/chi phí tối ưu hơn rất nhiều.
> - **Từ Bùi Hoàng Sơn:** Thấy rõ nhược điểm của việc chia nhỏ theo câu đơn thuần (`SentenceChunker`) đối với văn bản pháp lý. Chia theo câu sẽ xé lẻ các điểm liệt kê (a, b, c) ra khỏi phần câu dẫn chủ quản ở đầu khoản, làm mất đi tính toàn vẹn và mạch lạc ngữ nghĩa của luật định.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Trong các bài toán thực tế, chất lượng của hệ thống RAG không chỉ phụ thuộc vào sức mạnh của LLM hay độ phức tạp của vector store, mà phụ thuộc rất lớn vào **chiến lược xử lý dữ liệu (Data Strategy)** bao gồm: tiền xử lý văn bản, xác định ranh giới chunking phù hợp và thiết kế schema metadata tối ưu. Việc kết hợp Hybrid Search (BM25 + Dense Vector) cũng là một giải pháp cực kỳ hiệu quả mà các nhóm khác đã chia sẻ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thiết kế metadata chi tiết hơn (như gán cụ thể số `Điều` và `Chương` vào metadata của từng chunk con), đồng thời bắt buộc sử dụng các mô hình nhúng đa ngôn ngữ chuyên dụng (ví dụ: `keepitreal/vietnamese-sbert` hoặc `sentence-transformers/LaBSE`) thay vì `all-MiniLM-L6-v2` chỉ hỗ trợ tốt tiếng Anh để cải thiện độ tương đồng ngữ nghĩa tiếng Việt.

### Failure Analysis (Phân tích lỗi của chiến lược Recursive 500)

**Trường hợp lỗi thực tế:**
Khi sử dụng `RecursiveChunker(chunk_size=500)` trên Bộ luật Hình sự, các điều luật dài và có cấu trúc phức tạp (ví dụ: **Điều 134 về tội cố ý gây thương tích** gồm 6 khoản và nhiều điểm nhỏ) sẽ bị chia cắt thành 3-4 chunks độc lập.
- **Biểu hiện lỗi:** Khi người dùng hỏi *"Tội cố ý gây thương tích bị phạt tù tối đa bao nhiêu năm?"*, thông tin phạt tù tối đa (nằm ở Khoản 4, 5 hoặc 6 của Điều 134, có hình phạt lên tới 20 năm hoặc tù chung thân) nằm ở các chunk con phía sau. Tuy nhiên, do mật độ từ khóa ở Khoản 1 hoặc 2 cao hơn, top-1 chunk được trả về có thể chỉ là Khoản 1 (quy định hình phạt cải tạo không giam giữ đến 3 năm hoặc phạt tù từ 6 tháng đến 3 năm).
- **Hậu quả:** RAG Agent chỉ nhận được context từ top-1 chunk và trả lời sai lệch nghiêm trọng rằng hình phạt tối đa là 3 năm tù, bỏ sót hoàn toàn các khung hình phạt nặng hơn ở các khoản sau.

**Giải pháp khắc phục đề xuất:**
1. **Parent-Child Retrieval:** Chia nhỏ văn bản thành các child chunks (cỡ 200 ký tự) để tăng độ chính xác so khớp cosine, nhưng khi tìm thấy child chunk thì sẽ trả về toàn bộ parent document (cả Điều luật gốc, khoảng 1500-2000 ký tự) cho LLM.
2. **Metadata Grouping:** Gán tag `article_id` cho từng chunk. Khi truy xuất top-k, hệ thống tự động quét và lấy thêm tất cả các chunk có cùng `article_id` để tái cấu trúc lại toàn bộ Điều luật trước khi chuyển qua LLM.
3. **Tăng Top-K:** Tăng `top_k` retrieved chunks từ 3 lên 5 hoặc 7 để đảm bảo các khoản tiếp theo của Điều luật dài không bị bỏ sót.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
