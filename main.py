import requests
import json
import logging
import time
import asyncio
import json
import os
import cv2
import base64
import re
import matplotlib.pyplot as  plt
import numpy as np
import ast
import math
import unicodedata
from bs4 import BeautifulSoup

from json_repair import repair_json

def _salvage_malformed_dict(text: str, fix_func):
    """
    Trích xuất Key-Value bằng Regex nghiêm ngặt.
    Nếu Value lỗi -> Trả về "" (chuỗi rỗng).
    """
    # Chuẩn bị chuỗi: thay thế xuống dòng để regex hoạt động tốt trên 1 dòng
    text_flat = text.replace('\n', ' ') 
    
    # Xóa ngoặc nhọn bao quanh
    inner_text = text_flat.strip()
    if inner_text.startswith('{'): inner_text = inner_text[1:]
    if inner_text.endswith('}'): inner_text = inner_text[:-1]
    
    # REGEX TÌM KEY (QUAN TRỌNG NHẤT)
    # Giải thích:
    # (?:^|[,\{\[]) : Key phải bắt đầu ở đầu dòng, hoặc sau dấu phẩy, hoặc sau dấu { [
    # \s*           : Khoảng trắng tùy ý
    # (['"])        : Dấu mở quote (Group 1)
    # (.*?)         : Tên Key (Group 2) - Non-greedy
    # \1            : Dấu đóng quote tương ứng
    # \s*:          : Dấu hai chấm
    # Regex này ngăn chặn việc bắt nhầm dấu : bên trong Value (ví dụ "Nốt ruồi C: ...")
    key_pattern = re.compile(r"(?:^|[,\{\[])\s*(['\"])(.*?)\1\s*:")
    
    matches = list(key_pattern.finditer(inner_text))
    result = {}
    
    for i, match in enumerate(matches):
        key = match.group(2) # Lấy tên key
        
        # Xác định vùng chứa Value
        start_val_idx = match.end()
        
        if i < len(matches) - 1:
            # Value kết thúc tại nơi Key tiếp theo bắt đầu
            end_val_idx = matches[i+1].start()
            raw_value = inner_text[start_val_idx:end_val_idx]
            
            # Cắt bỏ dấu phẩy ngăn cách ở cuối (nếu có)
            # Tìm dấu phẩy cuối cùng và cắt tại đó
            last_comma = raw_value.rfind(',')
            if last_comma != -1:
                raw_value = raw_value[:last_comma]
        else:
            # Key cuối cùng
            raw_value = inner_text[start_val_idx:]
        
        raw_value = raw_value.strip()
        
        # Xử lý Value
        final_value = ""
        
        # Nếu value rỗng
        if not raw_value:
            result[key] = ""
            continue

        # Nếu là List/Dict
        if raw_value.startswith('[') or raw_value.startswith('{'):
            try:
                # Thử fix và parse
                fixed_val = fix_func(raw_value)
                final_value = ast.literal_eval(fixed_val)
            except:
                # Lỗi -> Trả về rỗng (list thì [], dict thì {})
                final_value = [] if raw_value.startswith('[') else {}
        else:
            # Nếu là String/Number
            try:
                # Thử parse chuẩn
                fixed_val = fix_func(raw_value)
                final_value = ast.literal_eval(fixed_val)
            except:
                # Nếu parse thất bại (thường do quote không đóng, hoặc ký tự lạ)
                # Ta sẽ cố gắng lấy nội dung thô (raw string)
                
                # 1. Nếu nhìn giống string ('...') nhưng lỗi cú pháp bên trong
                if (raw_value.startswith("'") and raw_value.endswith("'")) or \
                   (raw_value.startswith('"') and raw_value.endswith('"')):
                    # Cắt bỏ quote bao quanh và lấy ruột
                    clean_val = raw_value[1:-1]
                    # Nếu ruột chứa dấu : hoặc { (dấu hiệu của việc cắt sai hoặc rác), gán rỗng
                    if "':" in clean_val or '":' in clean_val:
                        final_value = ""
                    else:
                        final_value = clean_val
                else:
                    # Nếu không có quote bao quanh (ví dụ: Nguyen Van A), lấy luôn
                    # Nhưng nếu nó chứa ký tự nguy hiểm của code, gán rỗng cho an toàn
                    if any(c in raw_value for c in ['{', '}', '[', ']']):
                        final_value = "" 
                    else:
                        # Xóa các dấu nháy thừa nếu có
                        final_value = raw_value.replace("'", "").replace('"', "")

        result[key] = final_value

    if result:
        return result
    return None

def robust_json_parser(text: str):
    """
    Phân tích chuỗi JSON/Dict mạnh mẽ, đặc biệt tối ưu cho OCR tiếng Việt.
    Xử lý tốt các trường hợp tên dân tộc (K' HA, H' Hen), lỗi thiếu ngoặc,
    và tự động gán rỗng nếu value bị lỗi không thể cứu.
    """
    if not isinstance(text, str):
        if isinstance(text, (dict, list)):
            return text
        return None

    # 1. Dọn dẹp sơ bộ
    text_r = re.sub(r'^\s*```(json)?\s*|\s*```\s*$', '', text.strip(), flags=re.DOTALL)
    
    # Hàm tìm khối { ... } hoặc [ ... ]
    def find_valid_block(text_to_scan):
        start_idx = -1
        for i, char in enumerate(text_to_scan):
            if char in ['{', '[']:
                start_idx = i
                break
        if start_idx == -1: return None
        
        # Đếm ngoặc để lấy khối cân bằng nhất có thể
        start_char = text_to_scan[start_idx]
        end_char = '}' if start_char == '{' else ']'
        stack = []
        for i, char in enumerate(text_to_scan[start_idx:], start=start_idx):
            if char == start_char:
                stack.append(char)
            elif char == end_char:
                if stack: stack.pop()
            if not stack:
                return text_to_scan[start_idx : i + 1]
        return text_to_scan[start_idx:] # Trả về hết nếu thiếu ngoặc đóng

    raw_block = find_valid_block(text_r)
    if not raw_block:
        return None

    # ==================================================================
    # BƯỚC 2: SỬA LỖI CÚ PHÁP (QUAN TRỌNG)
    # ==================================================================
    
    def fix_syntax_errors(block):
        # 1. Escape dấu nháy đơn trong tên riêng (K' HA, H' Hen, M'gar)
        # Logic: Dấu ' nằm sau chữ cái VÀ (nằm trước chữ cái HOẶC khoảng trắng + chữ cái)
        # (?<=[a-zA-Z]) : Ký tự trước là chữ
        # '             : Dấu nháy cần escape
        # (?=[a-zA-Z\s]): Ký tự sau là chữ hoặc khoảng trắng
        block = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z\s])", r"\\'", block)
        
        # 2. Xử lý số bắt đầu bằng 0 bị lỗi (ví dụ: : 0123 -> : '0123')
        # Tìm số bắt đầu bằng 0, không nằm trong quote, sau dấu : hoặc , hoặc [
        block = re.sub(r'(?<=[:\[,]\s)0(\d+)', r"'0\1'", block)
        
        # 3. Đóng ngoặc nếu thiếu
        open_braces = block.count('{')
        close_braces = block.count('}')
        if close_braces < open_braces: block += '}' * (open_braces - close_braces)
        
        open_brackets = block.count('[')
        close_brackets = block.count(']')
        if close_brackets < open_brackets: block += ']' * (open_brackets - close_brackets)
        
        return block

    processed_block = fix_syntax_errors(raw_block)

    # ==================================================================
    # BƯỚC 3: THỬ PARSE BẰNG AST (LẦN 1)
    # ==================================================================
    try:
        return ast.literal_eval(processed_block)
    except Exception:
        pass

    # ==================================================================
    # BƯỚC 4: XỬ LÝ NÂNG CAO (Bọc value thiếu quote) VÀ THỬ LẠI (LẦN 2)
    # ==================================================================
    def _wrap_unquoted_values(match):
        colon = match.group(1)
        whitespace = match.group(2)
        value = match.group(3)
        value_stripped = value.strip()
        # Nếu đã có quote hoặc là số/bool/none/list/dict thì giữ nguyên
        if (value_stripped.startswith(("'", '"', '[', '{')) or
            value_stripped.lower() in ('none', 'true', 'false') or
            re.fullmatch(r'-?\d+(\.\d+)?', value_stripped)):
            return match.group(0)
        # Escape quote trong value và bọc lại
        escaped_value = value_stripped.replace("'", "\\'")
        return f"{colon}{whitespace}'{escaped_value}'"

    # Regex tìm value sau dấu : (tránh match nhầm bên trong string)
    processed_block_2 = re.sub(r'(:)(\s*)([^,}\]]+)(?=\s*[,}\]])', _wrap_unquoted_values, processed_block)
    
    # Chạy lại fix syntax lần nữa cho chắc (vì regex trên có thể làm lộ ra lỗi mới)
    processed_block_2 = fix_syntax_errors(processed_block_2)

    try:
        return ast.literal_eval(processed_block_2)
    except Exception as e:
        print(f"--- Chuỗi đã được xử lý nhưng vẫn lỗi. Lỗi: {e} ---")
        #print(processed_block_2)

    try:
        content = repair_json(processed_block_2, return_objects=True)
        return content
    except Exception as e:
        print(f"--- Chuỗi đã được xử lý repair_json nhưng vẫn lỗi. Lỗi: {e} ---")
        print(f"Chuyển sang chế độ cứu dữ liệu từng phần.")
        
    # ==================================================================
    # BƯỚC 5: CHẾ ĐỘ CỨU DỮ LIỆU (SALVAGE MODE)
    # ==================================================================
    if raw_block.strip().startswith('{'):
        return _salvage_malformed_dict(raw_block, fix_syntax_errors)
    
    return None

def get_images(dir_data):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(dir_data):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

NEW_GCN = """Bạn là một hệ thống AI đẳng cấp thế giới hỗ trợ nhận diện ký tự quang học (Optical Character Recognition - OCR) từ hình ảnh.
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

NEW_SCHEMA = {
    "type": "object",
    "properties": {
        "Tên tỉnh": {"type": "string", "description": "Tên tỉnh hoặc đài phát hành vé số. Nếu không có thì để 'None'."},
        "Ngày tháng năm phát hành": {"type": "string", "description": "Ngày tháng năm phát hành hoặc ngày mở thưởng của vé số. Nếu không có thì để 'None'."},
        "Dãy số trúng thưởng": {"type": "string", "description": "Dãy số tham gia dự thưởng trên vé số. Nếu không có thì để 'None'."}
    },
    "required": ["Tên tỉnh", "Ngày tháng năm phát hành", "Dãy số trúng thưởng"],
    "additionalProperties": False
}

url = "http://103.9.156.145:8000/v1"
url_llm = "http://10.225.0.28:8000/v1"

from openai import OpenAI

print(url)
client = OpenAI(
    base_url=url,  # Local Ollama API
    api_key="EMPTY"
)

client_llm = OpenAI(
    base_url=url_llm,  # Local Ollama API
    api_key="EMPTY"
)

def generate(prompt, schema=None):
    kwargs = {
        "model": "aisingapore/Qwen-SEA-LION-v4-8B-VL",
        "messages": prompt,
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "extra_body": {
            "top_k": 1,
            "min_p": 0.1,
            "repetition_penalty": 1.06,
        }
    }
    
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "gcn_schema",
                "schema": schema,
                "strict": True
            }
        }

    stream = client.chat.completions.create(**kwargs)
    return stream.choices[0].message.content

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def get_province_slug(province):
    if not province:
        return ""
    province = province.lower().strip()
    if province == "tp. hồ chí minh" or province == "hồ chí minh" or "hcm" in province:
        return "tp-hcm"
    if province == "thừa thiên - huế" or province == "thừa thiên huế":
        return "thua-thien-hue"
    
    slug = remove_accents(province)
    slug = slug.replace("đ", "d")
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    return slug

def format_date_for_url(date_str):
    if not date_str:
        return ""
    match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}-{int(month):02d}-{year}"
    return date_str.replace("/", "-")

def scrape_lottery_results_minhngoc(province, date):
    slug = get_province_slug(province)
    fmt_date = format_date_for_url(date)
    
    if not slug or not fmt_date:
        print("Không thể tạo URL hợp lệ từ Tỉnh và Ngày.")
        return ""
        
    url = f"https://www.minhngoc.net.vn/ket-qua-xo-so/{slug}/{fmt_date}.html"
    print(f"Đang lấy dữ liệu từ trang: {url} ...")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Loại bỏ các tag không cần thiết (menu, quảng cáo, script)
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()
            
        result_text = ""
        
        # Thử tìm bảng kết quả (Minh Ngọc thường dùng chữ "ĐB" hoặc "Giải")
        tables = soup.find_all('table')
        for table in tables:
            text_content = table.get_text(separator=' ', strip=True)
            if "ĐB" in text_content or "Giải" in text_content or "G.8" in text_content or "Đặc biệt" in text_content:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    row_data = [col.get_text(separator=' ', strip=True) for col in cols]
                    if any(row_data):
                        result_text += " | ".join(row_data) + "\n"
                
                # Nếu đã lấy được dữ liệu bảng thì thoát vòng lặp
                if len(result_text) > 50:
                    break
        
        # Fallback: Nếu không tìm thấy bảng hợp lệ, lấy toàn bộ text (đã xóa menu)
        if not result_text or len(result_text) < 50:
            content_div = soup.find(class_='content')
            if content_div:
                result_text = content_div.get_text(separator='\n', strip=True)
            else:
                result_text = soup.body.get_text(separator='\n', strip=True)
                
            # Cắt bớt nếu quá dài (lấy phần giữa thường chứa nội dung chính)
            if len(result_text) > 4000:
                result_text = result_text[-4000:] 
                
        return result_text
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu từ Minh Ngọc: {e}")
        return ""

def format_date_for_api(date_str):
    if not date_str:
        return ""
    match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        # Định dạng yyyy-mm-dd
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return date_str.replace("/", "-")

def format_lottery_table(draw):
    lines = []
    lines.append(f"Đài: {draw.get('province', {}).get('name', '')} - Ngày: {draw.get('formatted_date', '')}")
    lines.append(f"| {'Giải thưởng':<20} | {'Số trúng thưởng':<30} |")
    lines.append(f"|{'-'*22}|{'-'*32}|")
    
    prize_names = {
        "DB": "Giải đặc biệt (DB)",
        "G1": "Giải nhất (G1)",
        "G2": "Giải nhì (G2)",
        "G3": "Giải 3 (G3)",
        "G4": "Giải 4 (G4)",
        "G5": "Giải 5 (G5)",
        "G6": "Giải 6 (G6)",
        "G7": "Giải 7 (G7)",
        "G8": "Giải 8 (G8)"
    }
    
    results = draw.get("results", [])
    results.sort(key=lambda x: x.get("prizeOrder", 99))
    
    for r in results:
        code = r.get("prizeCode", "")
        if code not in prize_names:
            continue
            
        name = prize_names.get(code, code)
        values = r.get("values", [])
        values_str = " - ".join(values)
        lines.append(f"| {name:<20} | {values_str:<30} |")
        
    return "\n".join(lines)

def get_lottery_results_xosoapi(province, date_str):
    slug = get_province_slug(province)
    fmt_date = format_date_for_api(date_str)
    
    if not slug or not fmt_date:
        print("Không thể tạo ID Tỉnh và Ngày hợp lệ cho API.")
        return ""
        
    api_key = os.getenv("XOSOAPI_KEY", "sk_live_f4d14e2c0a820dfdd70f203fb58bf4f5f9288765c3be37a109bf5cd140301a5b")
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    province_id = None
    print(f"Đang lấy danh sách các tỉnh từ XoSoAPI...")
    try:
        prov_resp = requests.get("https://xosoapi.online/api/v1/vietnam/provinces", headers=headers, timeout=10)
        prov_resp.raise_for_status()
        provinces_data = prov_resp.json()
        
        # Tìm code của đài dựa vào so sánh name (hoặc slug của name)
        prov_list = provinces_data.get("data", []) if isinstance(provinces_data, dict) else provinces_data
        if not isinstance(prov_list, list):
            prov_list = []
            
        for p in prov_list:
            p_name = str(p.get("name", ""))
            p_slug = get_province_slug(p_name)
            
            if slug == p_slug or slug in p_slug or p_slug in slug:
                province_id = p.get("code")
                break
                
    except Exception as e:
        print(f"❌ Lỗi khi lấy danh sách tỉnh từ XoSoAPI: {e}")
        
    if not province_id:
        # Nếu không tìm thấy, fallback về slug (trong trường hợp slug vô tình trùng với code)
        province_id = slug
        print(f"Không tìm thấy mapping cho tỉnh {province}, dùng mặc định: {slug}")

    # Gọi API draws
    url = "https://xosoapi.online/api/v1/vietnam/draws"
    querystring = {"code": province_id, "date": fmt_date, "limit": 1}
    
    print(f"Đang gọi XoSoAPI kqxs cho đài {province_id} ngày {fmt_date}...")
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Lọc client-side nếu API trả về danh sách lớn mà không xử lý query parameters
        draws_list = data.get("data", []) if isinstance(data, dict) else data
        filtered = []
        if isinstance(draws_list, list):
            for d in draws_list:
                d_date = str(d.get("date", ""))
                d_prov = str(d.get("province", {}).get("code", ""))
                
                # match date
                if fmt_date in d_date:
                    if str(province_id) == d_prov:
                        filtered.append(d)
                        
            # Fallback in case query parameters worked and filtering is too strict
            if not filtered and len(draws_list) > 0:
                filtered = draws_list

        if filtered:
            # Chuyển dữ liệu xổ số thành bảng format Markdown
            return format_lottery_table(filtered[0])
            
        return "Không tìm thấy dữ liệu kết quả xổ số phù hợp."
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu từ XoSoAPI: {e}")
        return ""

def check_lottery_prize(province, date, ticket_number, search_context):
    prompt = f"""Bạn là một trợ lý ảo chuyên tra cứu kết quả xổ số.
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
Kết quả trả về phải là một chuỗi JSON theo định dạng sau:
{{
\t"status": "Trúng hoặc Không trúng",
\t"giai_trung": "Giải trúng thưởng: DB, G1, G2, G3, G4, G5, G6, G7, G8. Nếu không trúng thì để None"
}}
"""

    schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Trạng thái trúng thưởng: Trúng hoặc Không trúng"
            },
            "giai_trung": {
                "type": "string",
                "description": "Giải trúng thưởng: DB, G1, G2, G3, G4, G5, G6, G7, G8. Nếu không trúng thì để None"
            }
        },
        "required": ["status", "giai_trung"],
        "additionalProperties": False
    }

    messages = [{"role": "user", "content": prompt}]
    print(messages)
    kwargs = {
        "model": "openai/gpt-oss-20b",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lottery_prize_schema",
                "schema": schema,
                "strict": True
            }
        }
    }
    
    print("Đang gọi LLM để kiểm tra vé số trúng thưởng...")
    try:
        response = client_llm.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Lỗi khi gọi LLM: {e}")
        return "Lỗi khi kiểm tra giải thưởng."

Data = 'images'
output = 'results_tmp/'

im_fn_list = get_images(Data)
for im_fn in im_fn_list:
    print('===============')
    print(im_fn)
    base64_data = ""   

    encoded_image = ""
    with open(im_fn, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    decoded_image_text = encoded_image.decode('utf-8')

    #decoded_image_text = enhance_contrast_base64(decoded_image_text, method="clahe")

    # 1️⃣ Giải mã base64 thành dữ liệu nhị phân
    image_data = base64.b64decode(decoded_image_text)

    # 2️⃣ Đưa dữ liệu này vào NumPy array
    nparr = np.frombuffer(image_data, np.uint8)

    # 3️⃣ Giải mã ảnh từ mảng nhị phân bằng OpenCV
    img_root = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_root is None:
        print("Error decoding image {}!".format(im_fn))
        continue

    img = img_root.copy()

    height, width = img.shape[:2]
    max_width, max_height = 4096, 4096
    #max_width, max_height = 1920, 1080
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # 4️⃣ Chuyển sang trắng đen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 5️⃣ Nếu muốn mã hóa lại thành base64
    _, buffer = cv2.imencode('.jpg', gray)
    encoded_image = base64.b64encode(buffer)
    decoded_image_text = encoded_image.decode('utf-8')

    base64_data = f"data:image;base64,{decoded_image_text}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_data
                    }
                },
                {
                    "type": "text",
                    "text": NEW_GCN
                },
            ],
        }
    ]

    start = time.time()
    result = generate(messages, schema = NEW_SCHEMA)
    print(time.time() - start)

    isJson = False
    res_json_2 = None
    try:
        res_json = json.loads(result)
        res_json_2 = res_json
        isJson = True
    except json.JSONDecodeError as e:
        print("❌ JSON không hợp lệ: ", e)

    if isJson == False:
        print("robust_json_parser .... ")
        res_json = robust_json_parser(str(result))
        if type(res_json) == dict:
            res_json_2 = res_json

    res_json = res_json_2

    res_json_str = json.dumps(res_json, ensure_ascii=False, indent=4)
    print(res_json_str)

    # Lưu kết quả ra ảnh
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Hiển thị ảnh gốc
    img_rgb = cv2.cvtColor(img_root, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.axis('off')
    ax1.set_title(f"Image: {os.path.basename(im_fn)}")
    
    # Hiển thị JSON text
    ax2.axis('off')
    ax2.text(0.05, 0.5, res_json_str, fontsize=16, va='center', ha='left', family='monospace')
    
    # Lưu hình ảnh
    save_path = os.path.join('results', f"result_{os.path.basename(im_fn)}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Đã lưu ảnh kết quả tại: {save_path}")

    # ===== TRA CỨU VÀ KIỂM TRA TRÚNG THƯỞNG =====
    if res_json_2 and isinstance(res_json_2, dict):
        province = res_json_2.get("Tên tỉnh")
        date = res_json_2.get("Ngày tháng năm phát hành")
        ticket_number = res_json_2.get("Dãy số trúng thưởng")
        
        if province and date and ticket_number and province != "None" and date != "None" and ticket_number != "None":
            #search_context = get_lottery_results_xosoapi(province, date)
            search_context = """Đài: Thừa Thiên Huế - Ngày: 14/10/2024\n| Giải thưởng          | Số trúng thưởng                |\n|----------------------|--------------------------------|\n| Giải đặc biệt (DB)   | 386552                         |\n| Giải nhất (G1)       | 97595                          |\n| Giải nhì (G2)        | 80048                          |\n| Giải 3 (G3)          | 94734 - 32999                  |\n| Giải 4 (G4)          | 74464 - 03611 - 20031 - 88447 - 98461 - 48671 - 24039 |\n| Giải 5 (G5)          | 8476                           |\n| Giải 6 (G6)          | 0262 - 4629 - 7874             |\n| Giải 7 (G7)          | 665                            |\n| Giải 8 (G8)          | 76                             |\n"""
            
            if search_context:
                llm_answer = check_lottery_prize(province, date, ticket_number, search_context)
                print("\n================== KẾT QUẢ XỔ SỐ ==================")
                print(f"{search_context}\n")

                isJson = False
                res_json_2 = None
                try:
                    res_json = json.loads(llm_answer)
                    res_json_2 = res_json
                    isJson = True
                except json.JSONDecodeError as e:
                    print("❌ JSON không hợp lệ: ", e)

                if isJson == False:
                    print("robust_json_parser .... ")
                    res_json = robust_json_parser(str(llm_answer))
                    if type(res_json) == dict:
                        res_json_2 = res_json

                res_json = res_json_2

                res_json_str = json.dumps(res_json, ensure_ascii=False, indent=4)
                print(res_json_str)
                
                # Lưu text dò vé số
                result_txt_path = os.path.join('results', f"check_result_{os.path.basename(im_fn)}.md")
                with open(result_txt_path, "w", encoding="utf-8") as f:
                    f.write(llm_answer)
                print(f"Đã lưu chi tiết đối chiếu tại: {result_txt_path}")
            else:
                print("Không lấy được kết quả từ trang web minhngoc.")
        else:
            print("Không trích xuất đủ thông tin Tỉnh, Ngày hoặc Dãy số từ ảnh để dò vé.")
        time.sleep(10)
