"""
Utility functions — JSON parser, province slug, date formatter, image processing.

Các hàm tiện ích dùng chung cho toàn bộ app.
"""

import re
import ast
import base64
import unicodedata

import cv2
import numpy as np
from json_repair import repair_json


# ==============================================================================
# ROBUST JSON PARSER
# ==============================================================================

def _salvage_malformed_dict(text: str, fix_func):
    """
    Trích xuất Key-Value bằng Regex nghiêm ngặt.
    Nếu Value lỗi -> Trả về "" (chuỗi rỗng).
    """
    text_flat = text.replace('\n', ' ')
    inner_text = text_flat.strip()
    if inner_text.startswith('{'): inner_text = inner_text[1:]
    if inner_text.endswith('}'): inner_text = inner_text[:-1]

    key_pattern = re.compile(r"""(?:^|[,\{\[])\ *(['"])(.*?)\1\s*:""")
    matches = list(key_pattern.finditer(inner_text))
    result = {}

    for i, match in enumerate(matches):
        key = match.group(2)
        start_val_idx = match.end()

        if i < len(matches) - 1:
            end_val_idx = matches[i + 1].start()
            raw_value = inner_text[start_val_idx:end_val_idx]
            last_comma = raw_value.rfind(',')
            if last_comma != -1:
                raw_value = raw_value[:last_comma]
        else:
            raw_value = inner_text[start_val_idx:]

        raw_value = raw_value.strip()
        final_value = ""

        if not raw_value:
            result[key] = ""
            continue

        if raw_value.startswith('[') or raw_value.startswith('{'):
            try:
                fixed_val = fix_func(raw_value)
                final_value = ast.literal_eval(fixed_val)
            except Exception:
                final_value = [] if raw_value.startswith('[') else {}
        else:
            try:
                fixed_val = fix_func(raw_value)
                final_value = ast.literal_eval(fixed_val)
            except Exception:
                if (raw_value.startswith("'") and raw_value.endswith("'")) or \
                   (raw_value.startswith('"') and raw_value.endswith('"')):
                    clean_val = raw_value[1:-1]
                    if "':" in clean_val or '":' in clean_val:
                        final_value = ""
                    else:
                        final_value = clean_val
                else:
                    if any(c in raw_value for c in ['{', '}', '[', ']']):
                        final_value = ""
                    else:
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

    text_r = re.sub(r'^\s*```(json)?\s*|\s*```\s*$', '', text.strip(), flags=re.DOTALL)

    def find_valid_block(text_to_scan):
        start_idx = -1
        for i, char in enumerate(text_to_scan):
            if char in ['{', '[']:
                start_idx = i
                break
        if start_idx == -1:
            return None

        start_char = text_to_scan[start_idx]
        end_char = '}' if start_char == '{' else ']'
        stack = []
        for i, char in enumerate(text_to_scan[start_idx:], start=start_idx):
            if char == start_char:
                stack.append(char)
            elif char == end_char:
                if stack:
                    stack.pop()
            if not stack:
                return text_to_scan[start_idx: i + 1]
        return text_to_scan[start_idx:]

    raw_block = find_valid_block(text_r)
    if not raw_block:
        return None

    def fix_syntax_errors(block):
        block = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z\s])", r"\\'", block)
        block = re.sub(r'(?<=[:[\[,]\s)0(\d+)', r"'0\1'", block)
        open_braces = block.count('{')
        close_braces = block.count('}')
        if close_braces < open_braces:
            block += '}' * (open_braces - close_braces)
        open_brackets = block.count('[')
        close_brackets = block.count(']')
        if close_brackets < open_brackets:
            block += ']' * (open_brackets - close_brackets)
        return block

    processed_block = fix_syntax_errors(raw_block)

    try:
        return ast.literal_eval(processed_block)
    except Exception:
        pass

    def _wrap_unquoted_values(match):
        colon = match.group(1)
        whitespace = match.group(2)
        value = match.group(3)
        value_stripped = value.strip()
        if (value_stripped.startswith(("'", '"', '[', '{')) or
                value_stripped.lower() in ('none', 'true', 'false') or
                re.fullmatch(r'-?\d+(\.\d+)?', value_stripped)):
            return match.group(0)
        escaped_value = value_stripped.replace("'", "\\'")
        return f"{colon}{whitespace}'{escaped_value}'"

    processed_block_2 = re.sub(r'(:)(\s*)([^,}\]]+)(?=\s*[,}\]])', _wrap_unquoted_values, processed_block)
    processed_block_2 = fix_syntax_errors(processed_block_2)

    try:
        return ast.literal_eval(processed_block_2)
    except Exception:
        pass

    try:
        content = repair_json(processed_block_2, return_objects=True)
        return content
    except Exception:
        pass

    if raw_block.strip().startswith('{'):
        return _salvage_malformed_dict(raw_block, fix_syntax_errors)

    return None


# ==============================================================================
# PROVINCE & DATE HELPERS
# ==============================================================================

def remove_accents(input_str: str) -> str:
    """Xóa dấu tiếng Việt."""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def get_province_slug(province: str) -> str:
    """Chuyển tên tỉnh thành slug URL-friendly."""
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


def format_date_for_api(date_str: str) -> str:
    """Chuyển date string sang yyyy-mm-dd cho API."""
    if not date_str:
        return ""
    match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return date_str.replace("/", "-")


# ==============================================================================
# IMAGE PROCESSING
# ==============================================================================

def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, str, int, int]:
    """
    Tiền xử lý ảnh: decode → resize → grayscale → base64.
    Trả về (img_root, base64_data_uri, width, height).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_root = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_root is None:
        raise ValueError("Không thể decode ảnh. Vui lòng gửi file JPG/PNG hợp lệ.")

    img = img_root.copy()
    height, width = img.shape[:2]

    max_width, max_height = 4096, 4096
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # Chuyển sang trắng đen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mã hóa thành base64
    _, buffer = cv2.imencode('.jpg', gray)
    encoded = base64.b64encode(buffer).decode('utf-8')
    base64_data = f"data:image;base64,{encoded}"

    return img_root, base64_data, width, height
