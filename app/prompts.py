"""
Prompts và JSON Schema cho OCR vé số và kiểm tra trúng thưởng.
"""

# ==============================================================================
# OCR PROMPT - Trích xuất thông tin từ ảnh vé số
# ==============================================================================
OCR_PROMPT = """Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
Bạn được cung cấp 1 (một) hình ảnh tờ vé số.
Bạn phải thực hiện nhiệm vụ trích xuất các thông tin sau từ tờ vé số.

## Tham khảo danh sách các tỉnh, thành phố của Việt Nam (các đài phát hành vé số)
- Các tỉnh, thành phố ở Việt Nam: An Giang, Bà Rịa-Vũng Tàu, Bắc Giang, Bắc Kạn, Bạc Liêu, Bắc Ninh, Bến Tre, Bình Định, Bình Dương, Bình Phước, Bình Thuận, Cà Mau, Cần Thơ, Cao Bằng, Đà Nẵng, Đắk Lắk, Đắk Nông, Điện Biên, Đồng Nai, Đồng Tháp, Gia Lai, Hà Giang, Hà Nam, Hà Nội, Hà Tĩnh, Hải Dương, Hải Phòng, Hậu Giang, TP. Hồ Chí Minh, Hòa Bình, Hưng Yên, Khánh Hòa, Kiên Giang, Kon Tum, Lai Châu, Lâm Đồng, Lạng Sơn, Lào Cai, Long An, Nam Định, Nghệ An, Ninh Bình, Ninh Thuận, Phú Thọ, Phú Yên, Quảng Bình, Quảng Nam, Quảng Ngãi, Quảng Ninh, Quảng Trị, Sóc Trăng, Sơn La, Tây Ninh, Thái Bình, Thái Nguyên, Thanh Hóa, Thừa Thiên - Huế, Tiền Giang, Trà Vinh, Tuyên Quang, Vĩnh Long, Vĩnh Phúc, Yên Bái.

## Yêu cầu
- Chỉ trích xuất những thông tin sau từ hình ảnh tờ vé số được cung cấp:
    1. Tên tỉnh: Tên tỉnh hoặc đài phát hành tờ vé số.
    2. Ngày tháng năm phát hành: Ngày xổ số hoặc ngày mở thưởng được in trên vé (định dạng linh hoạt nhưng ưu tiên dd/mm/yyyy nếu có).
    3. Dãy số trúng thưởng: Dãy số tham gia dự thưởng (thường gồm 6 chữ số) in lớn và nổi bật trên vé số.
- KHÔNG bịa đặt, không đưa thêm thông tin hay diễn giải ngoài nội dung trong ảnh.

## QUAN TRỌNG
Kết quả trả về bằng tiếng Việt dưới dạng JSON có format sau:
```json
{
    "Tên tỉnh": "<str Nếu không có thì để None.>",
    "Ngày tháng năm phát hành": "<str Nếu không có thì để None.>",
    "Dãy số trúng thưởng": "<str Nếu không có thì để None.>"
}
```"""

# ==============================================================================
# OCR JSON SCHEMA - Structured output cho OCR
# ==============================================================================
OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "Tên tỉnh": {"type": "string", "description": "Tên tỉnh hoặc đài phát hành vé số. Nếu không có thì để 'None'."},
        "Ngày tháng năm phát hành": {"type": "string", "description": "Ngày tháng năm phát hành hoặc ngày mở thưởng của vé số. Nếu không có thì để 'None'."},
        "Dãy số trúng thưởng": {"type": "string", "description": "Dãy số tham gia dự thưởng trên vé số. Nếu không có thì để 'None'."}
    },
    "required": ["Tên tỉnh", "Ngày tháng năm phát hành", "Dãy số trúng thưởng"],
    "additionalProperties": False
}

# ==============================================================================
# CHECK LOTTERY PROMPT - Dò vé số trúng thưởng
# ==============================================================================
CHECK_PROMPT_TEMPLATE = """Bạn là một trợ lý ảo chuyên tra cứu kết quả xổ số.
Người dùng có một tờ vé số đài {province}, mở thưởng ngày {date}, với dãy số dự thưởng là {ticket_number}.

Dưới đây là thông tin kết quả xổ số thu thập được từ Internet:
---
{search_context}
---

QUY TẮC TRÚNG THƯỞNG:
Vé số của người dùng chỉ trúng một giải nếu CÁC CHỮ SỐ CUỐI CÙNG của tờ vé số KHỚP HOÀN TOÀN với số trúng thưởng của giải đó (theo đúng thứ tự).
Ví dụ:
- Trúng giải đặc biệt (6 số): Khớp hoàn toàn cả 6 số.
- Trúng giải nhất, nhì, ba, tư (5 số): 5 chữ số cuối cùng của vé số phải khớp hoàn toàn với số trúng thưởng.
- Trúng giải năm, sáu (4 số): 4 chữ số cuối cùng của vé số phải khớp hoàn toàn với số trúng thưởng.
- Trúng giải bảy (3 số): 3 chữ số cuối cùng của vé số phải khớp hoàn toàn với số trúng thưởng.
- Trúng giải tám (2 số): 2 chữ số cuối cùng của vé số phải khớp hoàn toàn với số trúng thưởng.
CHÚ Ý QUAN TRỌNG: 
Tuyệt đối KHÔNG xét trường hợp khớp phần đầu của tờ vé số. Chỉ quan tâm đến phần đuôi (các chữ số cuối cùng). 
Ví dụ: Vé số 992280 có chữ số cuối là 80, nếu giải 3 là 80514 thì KHÔNG trúng (vì đuôi vé số 280 khác đuôi giải ba 514).

Dựa vào thông tin trên, hãy cho biết tờ vé số có trúng giải nào không?

QUAN TRỌNG:
Kết quả trả về phải là một chuỗi JSON theo định dạng sau:
{{
\t"status": "Trúng hoặc Không trúng",
\t"giai_trung": "Mã giải trúng thưởng: DB, G1, G2, G3, G4, G5, G6, G7, G8. Chỉ trả lời ngắn gọn mã giải tương ứng, không suy diễn dài dòng. Có thể trúng nhiều giải liệt kê đầy đủ các mã giải trúng. Nếu không trúng thì để None"
}}
"""

# ==============================================================================
# LOTTERY PRIZE SCHEMA - Structured output cho dò vé số
# ==============================================================================
LOTTERY_PRIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "description": "Trạng thái trúng thưởng: Trúng hoặc Không trúng"
        },
        "giai_trung": {
            "type": "string",
            "description": "Mã giải trúng thưởng: DB, G1, G2, G3, G4, G5, G6, G7, G8. Chỉ trả lời ngắn gọn mã giải tương ứng, không suy diễn dài dòng. Có thể trúng nhiều giải liệt kê đầy đủ các mã giải trúng. Nếu không trúng thì để None"
        }
    },
    "required": ["status", "giai_trung"],
    "additionalProperties": False
}
