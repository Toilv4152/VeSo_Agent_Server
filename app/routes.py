"""
API Routes — FastAPI endpoints cho dò vé số.
"""

import json
import time
import asyncio
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.config import LLM_BACKEND, OCR_MODEL, LLM_MODEL, executor, logger
from app.backends import LLMBackend, create_backend
from app.prompts import OCR_PROMPT, OCR_SCHEMA, CHECK_PROMPT_TEMPLATE, LOTTERY_PRIZE_SCHEMA
from app.utils import robust_json_parser, preprocess_image
from app.lottery_service import get_lottery_results_xosoapi

PRIZE_VALUES = {
    "DB": "2.000.000.000đ",
    "G1": "30.000.000đ",
    "G2": "15.000.000đ",
    "G3": "10.000.000đ",
    "G4": "3.000.000đ",
    "G5": "1.000.000đ",
    "G6": "400.000đ",
    "G7": "200.000đ",
    "G8": "100.000đ"
}

# ==============================================================================
# ROUTER
# ==============================================================================
router = APIRouter()

# Khởi tạo backend (lazy, chỉ tạo khi cần)
_backend: Optional[LLMBackend] = None


def get_backend() -> LLMBackend:
    global _backend
    if _backend is None:
        _backend = create_backend()
    return _backend


def sse_event(step: str, message: str = None, data=None, status: str = None) -> str:
    """Format một SSE event thành chuỗi."""
    event = {"step": step}
    if message is not None:
        event["message"] = message
    if data is not None:
        event["data"] = data
    if status is not None:
        event["status"] = status
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def process_lottery_image(image_bytes: bytes, filename: str) -> AsyncGenerator[str, None]:
    """
    Pipeline xử lý ảnh vé số, yield SSE events từng bước.
    Chạy các bước blocking trong thread pool để không block event loop.
    """
    loop = asyncio.get_event_loop()
    backend = get_backend()

    # ========== BƯỚC 1: TIỀN XỬ LÝ ẢNH ==========
    #yield sse_event("upload", status="processing", message=f"Đã nhận ảnh: {filename}. Đang tiền xử lý...")

    try:
        img_root, base64_data, width, height = await loop.run_in_executor(
            executor, preprocess_image, image_bytes
        )
    except ValueError as e:
        yield sse_event("error", message=str(e))
        yield sse_event("done")
        return

    #yield sse_event("upload", status="done", message=f"Ảnh {filename} ({width}x{height}) đã sẵn sàng.")

    # ========== BƯỚC 2: OCR ==========
    yield sse_event("ocr", status="processing", message="🔍 Đang trích xuất thông tin vé số...")

    start_time = time.time()
    try:
        ocr_result = await loop.run_in_executor(
            executor, backend.generate_ocr, base64_data, OCR_PROMPT, OCR_SCHEMA
        )
    except Exception as e:
        logger.error(f"OCR error: {e}")
        yield sse_event("error", message=f"❌ Lỗi OCR: {e}")
        yield sse_event("done")
        return

    ocr_time = time.time() - start_time

    # Parse JSON từ OCR result
    res_json = None
    if isinstance(ocr_result, dict):
        res_json = ocr_result
    else:
        try:
            res_json = json.loads(ocr_result)
        except (json.JSONDecodeError, TypeError):
            res_json = robust_json_parser(str(ocr_result))
            if not isinstance(res_json, dict):
                res_json = None

    if res_json is None:
        yield sse_event("ocr", status="error", message=f"❌ Không parse được kết quả OCR. Raw: {ocr_result}")
        yield sse_event("done")
        return

    yield sse_event("ocr", status="done", data=res_json,
                     message=f"✅ Thông tin vé số")

    # ========== BƯỚC 3: KIỂM TRA THÔNG TIN ==========
    province = res_json.get("Tên tỉnh")
    date = res_json.get("Ngày tháng năm phát hành")
    ticket_number = res_json.get("Dãy số trúng thưởng")

    if not all([province, date, ticket_number]) or \
       any(v == "None" for v in [province, date, ticket_number]):
        yield sse_event("error",
                         message="⚠️ Không trích xuất được thông tin Tỉnh, Ngày hoặc Dãy số từ ảnh để dò vé.",
                         data=res_json)
        yield sse_event("done")
        return

    # ========== BƯỚC 4: TRA CỨU KQXS ==========
    yield sse_event("lookup", status="processing",
                     message=f"📡 Đang tra cứu KQXS đài {province} ngày {date}...")

    try:
        search_context = await loop.run_in_executor(
            executor, get_lottery_results_xosoapi, province, date
        )
        #search_context = """Đài: Bình Thuận - Ngày: 06/02/2025\n| Giải thưởng          | Số trúng thưởng                |\n|----------------------|--------------------------------|\n| Giải đặc biệt (DB)   | 926731                         |\n| Giải nhất (G1)       | 90400                          |\n| Giải nhì (G2)        | 06702                          |\n| Giải 3 (G3)          | 48778 - 07648                  |\n| Giải 4 (G4)          | 83690 - 92115 - 67667 - 11594 - 77844 - 20510 - 22332 |\n| Giải 5 (G5)          | 4195                           |\n| Giải 6 (G6)          | 0420 - 2614 - 7633             |\n| Giải 7 (G7)          | 978                            |\n| Giải 8 (G8)          | 54                             |\n"""
        #search_context = """Đài: Thừa Thiên Huế - Ngày: 14/10/2024\n| Giải thưởng          | Số trúng thưởng                |\n|----------------------|--------------------------------|\n| Giải đặc biệt (DB)   | 386552                         |\n| Giải nhất (G1)       | 97595                          |\n| Giải nhì (G2)        | 80048                          |\n| Giải 3 (G3)          | 94734 - 32999                  |\n| Giải 4 (G4)          | 74464 - 03611 - 20031 - 88447 - 98461 - 48671 - 24039 |\n| Giải 5 (G5)          | 8476                           |\n| Giải 6 (G6)          | 0262 - 4629 - 7874             |\n| Giải 7 (G7)          | 665                            |\n| Giải 8 (G8)          | 76                             |\n"""
    except Exception as e:
        logger.error(f"Lookup error: {e}")
        yield sse_event("error", message=f"❌ Lỗi tra cứu KQXS: {e}")
        yield sse_event("done")
        return

    if not search_context:
        yield sse_event("error", message="❌ Không lấy được kết quả xổ số từ web.")
        yield sse_event("done")
        return

    yield sse_event("lottery_table", status="done", data=search_context,
                     message="📋 Đã lấy được bảng KQXS.")

    # ========== BƯỚC 5: DÒ VÉ SỐ ==========
    yield sse_event("check", status="processing", message="🎯 Đang dò vé số...")

    prompt = CHECK_PROMPT_TEMPLATE.format(
        province=province,
        date=date,
        ticket_number=ticket_number,
        search_context=search_context,
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        llm_answer = await loop.run_in_executor(
            executor, backend.generate_text, messages, None, LOTTERY_PRIZE_SCHEMA
        )
    except Exception as e:
        logger.error(f"LLM check error: {e}")
        yield sse_event("error", message=f"❌ Lỗi khi gọi LLM dò vé: {e}")
        yield sse_event("done")
        return

    # Parse and process prize values
    try:
        parsed_json = json.loads(llm_answer)
    except json.JSONDecodeError:
        parsed_json = robust_json_parser(str(llm_answer))

    if not isinstance(parsed_json, dict):
        yield sse_event("error", message=f"❌ Không parse được kết quả dò vé: {llm_answer}")
        yield sse_event("done")
        return

    # Compute total prize money
    parsed_json["tien_thuong"] = "0đ"
    if parsed_json.get("status") == "Trúng":
        giai_str = parsed_json.get("giai_trung", "")
        if giai_str and giai_str != "None":
            danh_sach_giai = [g.strip() for g in giai_str.split(",") if g.strip()]
            tong_tien = 0
            cac_giai_trung = []
            for g in danh_sach_giai:
                if g in PRIZE_VALUES:
                    cac_giai_trung.append(g)
                    tien_int = int(PRIZE_VALUES[g].replace(".", "").replace("đ", ""))
                    tong_tien += tien_int
                    
            if tong_tien > 0:
                parsed_json["tien_thuong"] = f"{tong_tien:,.0f}".replace(",", ".") + "đ"
                parsed_json["giai_trung"] = ", ".join(cac_giai_trung)

    # Note that data should be the parsed_json dictionary, which converts to valid JSON in sse_event
    yield sse_event("result", status="done", data=parsed_json,
                     message="🎉 Đã có kết quả dò vé!")

    # ========== HOÀN TẤT ==========
    yield sse_event("done")


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "backend": LLM_BACKEND,
        "ocr_model": OCR_MODEL,
        "llm_model": LLM_MODEL,
    }


@router.post("/api/check-lottery")
async def check_lottery(file: UploadFile = File(...)):
    """
    Upload ảnh vé số để dò kết quả.
    Trả về Server-Sent Events (SSE) streaming từng bước xử lý.
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh (JPG, PNG, ...)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")

    filename = file.filename or "unknown.jpg"

    return StreamingResponse(
        process_lottery_image(image_bytes, filename),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Tắt buffering trên nginx
        }
    )
