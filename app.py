import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Thesis Dashboard - RAG IUH",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; text-transform: uppercase;}
    .sub-title { font-size: 1.1rem; color: #4B5563; text-align: center; margin-bottom: 30px;}
    .section-header { font-size: 1.6rem; font-weight: bold; color: #1E40AF; margin-top: 40px; margin-bottom: 20px; border-bottom: 2px solid #93C5FD; padding-bottom: 5px;}
    .sub-header { font-size: 1.2rem; font-weight: bold; color: #047857; margin-top: 15px;}
    .highlight-box { background-color: #F8FAFC; border-left: 5px solid #3B82F6; padding: 15px; border-radius: 4px; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU TỰ ĐỘNG
# ==========================================
METRICS_P1 =['bleu_score', 'rougel_f1', 'bert_score_f1', 'ans_ctx_bert_f1']
METRICS_P2 =['geval_correctness', 'geval_faithfulness', 'geval_helpfulness']
METRICS_RETRIEVAL =['hit_at_k', 'mrr_at_k', 'context_recall', 'retriever_f1']
METRICS_TIME =['retrieval_time', 'generation_time', 'total_time']

@st.cache_data
def load_and_process_data():
    data_dir = "data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        st.error(f"❌ Không tìm thấy file CSV nào trong thư mục '{data_dir}'.")
        return pd.DataFrame()

    summary_data =[]
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=';', decimal=',', on_bad_lines='skip')
            df.columns = df.columns.str.strip().str.lower()
            
            all_metrics = METRICS_P1 + METRICS_P2 + METRICS_RETRIEVAL + METRICS_TIME
            for col in all_metrics:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            mean_scores = df[[c for c in all_metrics if c in df.columns]].mean().to_dict()
            
            fname = os.path.basename(file).lower()
            if "baseline" in fname:
                mean_scores['Model'] = "1. Baseline (Qwen 1.5B)"
            elif "lite" in fname:
                mean_scores['Model'] = "2. Lite (VinaLlama 7B)"
            elif "sota" in fname:
                mean_scores['Model'] = "3. SOTA (Llama 3.1 8B)"
            elif "prune" in fname:
                mean_scores['Model'] = "4. SOTA Pruned (Gemma 3 4B)"
            else:
                mean_scores['Model'] = fname.replace('.csv', '')
                
            summary_data.append(mean_scores)
        except Exception as e:
            st.error(f"Lỗi đọc file {file}: {e}")
            
    if summary_data:
        res_df = pd.DataFrame(summary_data)
        res_df = res_df.sort_values(by="Model").reset_index(drop=True)
        return res_df
    return pd.DataFrame()

df_summary = load_and_process_data()

# Hàm highlight giá trị tốt nhất trong dataframe
def highlight_best(s):
    # Thời gian thì nhỏ nhất là tốt nhất, các chỉ số khác lớn nhất là tốt nhất
    is_time = s.name in METRICS_TIME
    if is_time:
        is_best = s == s.min()
    else:
        is_best = s == s.max()
    return['background-color: #D1FAE5; font-weight: bold; color: #065F46' if v else '' for v in is_best]

# ==========================================
# 3. HEADER & THÔNG TIN CHUNG
# ==========================================
st.markdown('<p class="main-title">HỆ THỐNG TRẢ LỜI TỰ ĐỘNG CÂU HỎI CỦA SINH VIÊN TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP TPHCM SỬ DỤNG MÔ HÌNH NGÔN NGỮ LỚN</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">BÁO CÁO TIẾN ĐỘ VÀ KẾT QUẢ THỰC NGHIỆM ĐỀ TÀI LUẬN VĂN THẠC SĨ</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# PHẦN 1: TỔNG QUAN PHƯƠNG PHÁP LUẬN
# ==========================================
st.markdown('<p class="section-header">PHẦN 1: TỔNG QUAN PHƯƠNG PHÁP LUẬN VÀ XÂY DỰNG DỮ LIỆU</p>', unsafe_allow_html=True)

st.markdown('<p class="sub-header">1.1. Chiến lược xử lý văn bản (Data Engineering & Chunking)</p>', unsafe_allow_html=True)
st.markdown("""
Thay vì sử dụng các bộ chia văn bản (Text Splitters) thông thường (như `RecursiveCharacterTextSplitter`) vốn dễ làm đứt gãy ngữ cảnh pháp lý, em đã tiền xử lý toàn bộ Quy chế đào tạo bằng phương pháp **Manual Tagging & Semantic Chunking**:
* **Sử dụng thẻ định ranh giới (Boundary Tags):** Văn bản được chèn thủ công các thẻ `(-----<<break >>-----)` tại các vị trí ngắt đoạn ngữ nghĩa tự nhiên.
* **Kế thừa Siêu dữ liệu (Metadata Inheritance):** Sử dụng Regex để quét các thẻ phân cấp như `document, chapter, article, clause`. Mỗi chunk sinh ra sẽ mang theo một bộ Dictionary Metadata hoàn chỉnh (Ví dụ: thuộc Chương nào, Điều mấy, Khoản nào).
* **Lợi ích khoa học:** Đảm bảo khi LLM sinh câu trả lời, mô hình có thể trích dẫn ngược lại chính xác Điều/Khoản gốc, giải quyết triệt để bài toán Hallucination về nguồn gốc tài liệu trong RAG.
""")

st.markdown('<p class="sub-header">1.2. Xây dựng bộ Dữ liệu Kiểm thử (Ground Truth Dataset)</p>', unsafe_allow_html=True)
st.markdown("""
Bộ dữ liệu gồm 1000 mẫu được xây dựng với sự hỗ trợ của **GPT-4o** dưới sự **giám sát và tinh chỉnh thủ công 100%**:
* **Cấu trúc chặt chẽ:** Gồm `Question`, `Context`, `Source ID`, và `Answer`.
* **Bổ sung tri thức ngầm (Implicit/Internal Knowledge):** Trong Ground Truth Answer, em cố tình đưa vào các thông tin thực tế mang tính đặc thù nội bộ của nhà trường (những thông tin không hề xuất hiện trên internet hoặc sinh viên lâu năm, giảng viên biết, nhưng văn bản quy chế không ghi chi tiết).
* **Mục đích:** Đây là dụng ý nghiên cứu để phục vụ bài toán *External Knowledge vs. Contextual Knowledge* sau này. Điều này cũng lý giải vì sao các mô hình RAG hiện tại (chỉ dựa vào Context) **không thể và không nên đạt điểm tuyệt đối (9-10)** khi so khớp với Ground Truth. Nếu LLM tự bịa ra thông tin nội bộ mà không có trong Context, nó sẽ bị phạt nặng ở tiêu chí Faithfulness.
""")

st.markdown('<p class="sub-header">1.3. Thiết lập Đánh giá (Evaluation Framework)</p>', unsafe_allow_html=True)
st.markdown("""<div class="highlight-box">
Việc đánh giá sử dụng <b>LLM-as-a-Judge (DeepSeek-R1-8B-Qwen3)</b> chạy cục bộ, kết hợp parsing chuỗi <code>&lt;think&gt;</code> để đảm bảo tính minh bạch trong suy luận của mô hình giám khảo. Thang đo G-EVAL (1-10) được định nghĩa chặt chẽ trong Prompt:
</div>""", unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    st.markdown("""**1. Correctness (Độ chính xác thông tin - So với Ground Truth)**
* *Score 9-10 (Xuất sắc):* Thông tin hoàn toàn trùng khớp về ý nghĩa, bao gồm cả các chi tiết quan trọng.
* *Score 7-8 (Tốt):* Đúng ý chính, nhưng có thể thiếu một vài chi tiết nhỏ.
* *Score 5-6 (Khá):* Đúng một phần, nhưng bỏ sót thông tin quan trọng.
* *Score 3-4 (Kém):* Sai lệch ý nghĩa đáng kể.
* *Score 1-2 (Rất tệ):* Sai hoàn toàn, hoặc trả lời "Không tìm thấy" trong khi GT có thông tin.
""")
with col_e2:
    st.markdown("""**2. Faithfulness (Độ trung thực - So với Retrieved Context)** - *(Chỉ số quan trọng nhất)*
* *Score 9-10 (Tuyệt đối):* Mọi tuyên bố trong câu trả lời đều có dẫn chứng cụ thể từ Context.
* *Score 7-8 (Cao):* Đa số thông tin từ Context, các suy luận là hợp lý.
* *Score 5-6 (Trung bình):* Có sử dụng Context nhưng pha trộn kiến thức bên ngoài.
* *Score 3-4 (Thấp):* Trả lời đúng câu hỏi nhưng thông tin KHÔNG nằm trong Context (Hallucination dạng đúng).
* *Score 1-2 (Ảo giác):* Bịa đặt thông tin, mâu thuẫn trực tiếp với Context.
""")
with col_e3:
    st.markdown("""**3. Helpfulness (Độ hữu ích - So với User Query)**
* *Score 9-10 (Chuyên nghiệp):* Văn phong lịch sự, cấu trúc mạch lạc (bullet points), trích dẫn nguồn rõ ràng.
* *Score 7-8 (Hữu ích):* Câu trả lời dễ hiểu, đầy đủ câu cú, nhưng thiếu trích dẫn hoặc format chưa đẹp.
* *Score 5-6 (Tạm được):* Đủ ý nhưng trình bày dạng khối (block text) khó đọc.
* *Score 3-4 (Sơ sài):* Trả lời cộc lốc, thiếu chủ ngữ vị ngữ.
* *Score 1-2 (Vô dụng):* Gây khó hiểu, rắc rối.
""")

# ==========================================
# PHẦN 2: CHI TIẾT 4 PHIÊN BẢN (ABLATION)
# ==========================================
st.markdown('<p class="section-header">PHẦN 2: CHI TIẾT KIẾN TRÚC 4 PHIÊN BẢN (ABLATION STUDY)</p>', unsafe_allow_html=True)
st.markdown("Các tham số và mô hình được tuyển chọn thông qua nhiều vòng thử nghiệm, dựa trên các công bố khoa học về tiếng Việt và RAG.")

exp1, exp2, exp3, exp4 = st.columns(4)
with exp1:
    st.info("**2.1. Phiên bản 1: Baseline**\n*(Vanilla RAG Tối ưu hóa)*")
    st.markdown("""
    * **Bản chất:** Đây là mô hình đối chứng. Mặc dù khá giống với Vanilla RAG nhưng các tham số đã được tinh chỉnh.
    * **Embedding:** `paraphrase-multilingual-MiniLM-L12-v2` (Đa ngữ, nhẹ, nhanh).
    * **Chunking:** `RecursiveCharacterTextSplitter` (chunk=512, overlap=50).
    * **Truy xuất:** Cosine Similarity thuần túy (Top-K = 3).
    * **LLM:** `Qwen2.5-1.5B-Instruct` (GGUF Q4). Baseline nhẹ nhất xử lý tiếng Việt tốt.
    """)
with exp2:
    st.success("**2.2. Phiên bản 2: Lite Version**\n*(Nhanh nhất)*")
    st.markdown("""
    * **Bản chất:** Kiến trúc Hybrid RAG kết hợp Parent-Child Chunking và Reranking.
    * **Chunking:** Parent (400, overlap=40) và Child (128, overlap=16). Truy xuất trên Child, đưa Parent vào LLM.
    * **Embedding:** Dense (`jina-embeddings-v3`) + Sparse (`BM25L`).
    * **Reranking:** `BAAI/bge-reranker-base` chấm điểm lại top 60.
    * **LLM:** `VinaLlama-7B-chat` (GGUF Q5) chuyên biệt tiếng Việt.
    """)
with exp3:
    st.warning("**2.3. Phiên bản 3: SOTA Version**\n*(Đề xuất cải tiến dựa trên ý tưởng mới)*")
    st.markdown("""
    * **Bản chất:** Advanced RAG tích hợp HyDE và Ensemble.
    * **Augmentation (HyDE):** LLM sinh thêm 1 "Câu hỏi giả định" và 1 "Tóm tắt 1 câu" trước khi nhúng vector.
    * **Ensemble Retrieval:** Chạy song song 5 Embedding Models (PhoBERT, v-bi-encoder, v-sbert, BGE-m3, e5).
    * **Fusion & Rerank:** Thuật toán RRF (k=60) + Cross-Encoder `mmarco-mMiniLMv2`.
    * **LLM:** `Meta-Llama-3.1-8B-Instruct` (GGUF Q4).
    """)
with exp4:
    st.error("**2.4. Phiên bản 4: SOTA Pruned**\n*(Tối ưu hóa từ SOTA)*")
    st.markdown("""
    * **Bản chất:** Cắt tỉa SOTA để tìm điểm cân bằng giữa thời gian và độ chính xác.
    * **Thay đổi:** Giảm Ensemble từ 5 xuống 3 Embeddings. Đổi LLM sinh văn bản sang `Gemma-3-4B-It` (GGUF Q4) để giảm tải VRAM nhưng vẫn giữ tư duy logic xuất sắc của họ Gemma. Ép Gemma sinh văn bản khó khăn nhưng ít lỗi hơn.
    """)

# ==========================================
# PHẦN 3: BẢNG ĐIỀU KHIỂN & BIỂU ĐỒ
# ==========================================
st.markdown('<p class="section-header">PHẦN 3: BẢNG THỐNG KÊ KẾT QUẢ (DASHBOARD)</p>', unsafe_allow_html=True)

if not df_summary.empty:
    st.markdown("**📋 3.1. BẢNG TỔNG HỢP SỐ LIỆU CHI TIẾT (Đã Highlight giá trị tốt nhất)**")
    
    display_df = df_summary.copy().set_index('Model')
    # Áp dụng hàm highlight_best đã định nghĩa ở trên (Tự động bôi đậm giá trị Max, riêng Time bôi Min)
    styled_df = display_df.style.format("{:.4f}").apply(highlight_best, axis=0)
    
    st.dataframe(styled_df, use_container_width=True, height=180)

    st.markdown("**📊 3.2. TRỰC QUAN HÓA DỮ LIỆU ĐÁNH GIÁ**")
    
    # --- ROW 1: Context Recall & Trade-off (Yêu cầu mới của user) ---
    col_new1, col_new2 = st.columns(2)
    
    with col_new1:
        # Biểu đồ Context Recall theo đúng hình mẫu
        fig_recall = px.line(
            df_summary, x='Model', y='context_recall', markers=True,
            title="Khả năng truy xuất đúng ngữ cảnh (Context Recall)",
            labels={'context_recall': 'Context Recall', 'Model': 'Version'}
        )
        fig_recall.update_traces(line_color='#1E88E5', marker=dict(size=10, color='#1E88E5'))
        fig_recall.update_layout(yaxis_range=[0.2, 0.65], margin=dict(t=40, b=10))
        st.plotly_chart(fig_recall, use_container_width=True)

    with col_new2:
        # Biểu đồ Trade-off (Sự đánh đổi giữa Thời gian và Độ hữu ích)
        fig_tradeoff = px.scatter(
            df_summary, x='total_time', y='geval_helpfulness', text='Model',
            size=[10, 15, 20, 25], color='Model', # Kích thước điểm tăng dần
            title="Sự đánh đổi: Thời gian phản hồi vs Độ hữu ích (Trade-off Analysis)",
            labels={'total_time': 'Tổng thời gian phản hồi (Giây) ➡️ (Càng thấp càng tốt)', 
                    'geval_helpfulness': 'Độ hữu ích G-Eval ➡️ (Càng cao càng tốt)'}
        )
        fig_tradeoff.update_traces(textposition="bottom right", textfont=dict(size=10))
        # Thêm các đường phân chia vùng
        fig_tradeoff.add_hline(y=6.0, line_dash="dot", line_color="gray", opacity=0.5)
        fig_tradeoff.add_vline(x=10.0, line_dash="dot", line_color="gray", opacity=0.5)
        fig_tradeoff.update_layout(margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    # --- ROW 2: G-EVAL & NLP Metrics ---
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_geval = px.bar(
            df_summary, x='Model', y=METRICS_P2, barmode='group',
            title="Đánh giá bằng LLM-as-a-Judge (G-EVAL Metrics)",
            labels={'value': 'Score (Thang 1 - 10)', 'variable': 'Tiêu chí'}
        )
        fig_geval.update_layout(yaxis_range=[0, 10], margin=dict(t=40, b=10))
        st.plotly_chart(fig_geval, use_container_width=True)

    with col_c2:
        fig_nlp = px.bar(
            df_summary, x='Model', y=METRICS_P1, barmode='group',
            title="Chỉ số NLP Truyền thống (BLEU, ROUGE, BERTScore)",
            labels={'value': 'Score (0.0 - 1.0)', 'variable': 'Tiêu chí'},
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_nlp.update_layout(yaxis_range=[0, 1.1], margin=dict(t=40, b=10))
        st.plotly_chart(fig_nlp, use_container_width=True)


# ==========================================
# PHẦN 4: PHÂN TÍCH CHUYÊN SÂU
# ==========================================
st.markdown('<p class="section-header">PHẦN 4: BẢN PHÂN TÍCH CHUYÊN SÂU (CRITICAL ANALYSIS)</p>', unsafe_allow_html=True)

st.markdown("""
**4.1. Giải mã Nghịch lý: Tại sao Traditional Metrics (BLEU/ROUGE) thấp nhưng G-EVAL lại cao?**
* **Kết quả:** Ở bản SOTA, ROUGE giảm (0.24) nhưng Correctness tăng vọt (7.2). Đây gần như không phải là lỗi, mà là minh chứng cho sự sụp đổ của N-gram metrics trong đánh giá LLM hiện đại (đã được đề cập trong các bài báo ACL gần đây).
* **Nguyên nhân:** Ground Truth được viết ngắn gọn, súc tích (Human-style). Trong khi đó, Llama 3.1 hay Gemma 3 (SOTA) có xu hướng sinh câu trả lời rất chi tiết, phân tích từng ý (Chat-style). BLEU và ROUGE đo sự trùng lặp từ vựng (Lexical overlap), nên việc LLM sinh dài khiến Recall bị loãng và Precision sụt giảm nghiêm trọng.
* **Ngược lại:** BERTScore (đo Semantic) và LLM-Judge hiểu được rằng *"tuy câu trả lời dài hơn, nhưng ý nghĩa cốt lõi hoàn toàn trùng khớp với Ground Truth"*, do đó điểm Correctness phản ánh chính xác hiệu năng thực tế.

**4.2. Đánh giá sự ảnh hưởng của Lượng tử hóa (Quantization)**
* Toàn bộ hệ thống được chạy trên các mô hình GGUF chuẩn `Q4_K_M` và `Q5`. Dù tối ưu VRAM rất tốt, nhưng các công bố cho thấy việc lượng tử hóa làm giảm khả năng "In-context Reasoning" (suy luận dựa trên ngữ cảnh) khoảng 2-5% so với bản Full Precision (FP16).
* Điều này giải thích một phần vì sao **Faithfulness cao nhất chỉ đạt 7.0**. Đôi khi LLM Q4 bị "ảo giác" nhẹ, bỏ sót các tiểu tiết trong Context rườm rà. Dù vậy, kết quả này là khá tin cậy và phù hợp với dự liệu.

**4.3. Tại sao điểm không đạt 9-10?**
* Đây là kết quả trung thực trên bộ dữ liệu chứa Tri thức nội bộ (Internal Knowledge) kết hợp với LLM thương mại mạnh nhất tại thời điểm tạo bộ dữ liệu là ChatGPT-4o. 
* Các mô hình sử dụng là bản Quantized (nén lượng tử hóa) để chạy trên tài nguyên hạn chế, nên việc mất mát một phần nhỏ độ chính xác là đánh đổi bắt buộc. Điểm 7.25/10 cho bài toán thực tế là một kết quả hợp lý.

**4.4. Hiệu quả của Pruning (Cắt tỉa)**
* Bản **Pruned SOTA (Gemma-3-4B + 3 Embeddings)** dù giảm số lượng mô hình (giảm thời gian tính toán) nhưng vẫn giữ được các thông số gần ngang bằng SOTA và thậm chí **GEval Helpfulness cao nhất (7.44)**. Điều này chứng minh Gemma 3 có khả năng giao tiếp, format văn bản và cấu trúc logic cực kỳ xuất sắc.

**4.5. Baseline vs. Advanced RAG**
* Sự chênh lệch **Context Recall (0.28 vs 0.60)** chứng minh rằng Vanilla RAG (Baseline) hoàn toàn thất bại trong việc tìm kiếm chi tiết trong văn bản pháp quy. Các kỹ thuật Reranking và Ensemble trong Lite/SOTA/Prune là yếu tố **bắt buộc** để hệ thống hoạt động chính xác.
""")

st.markdown("---")
st.caption("Báo cáo được trích xuất tự động từ hệ thống đánh giá thực nghiệm - Đại học Công nghiệp TPHCM.")