"""
Lottery Service — Tra cứu kết quả xổ số từ XoSoAPI.
"""

import requests

from app.config import XOSOAPI_KEY, logger
from app.utils import get_province_slug, format_date_for_api


def format_lottery_table(draw: dict) -> str:
    """Chuyển dữ liệu xổ số từ API thành bảng Markdown."""
    lines = []
    lines.append(f"Đài: {draw.get('province', {}).get('name', '')} - Ngày: {draw.get('formatted_date', '')}")
    lines.append(f"| {'Giải thưởng':<20} | {'Số trúng thưởng':<30} |")
    lines.append(f"|{'-' * 22}|{'-' * 32}|")

    prize_names = {
        "DB": "Giải đặc biệt (DB)", "G1": "Giải nhất (G1)", "G2": "Giải nhì (G2)",
        "G3": "Giải 3 (G3)", "G4": "Giải 4 (G4)", "G5": "Giải 5 (G5)",
        "G6": "Giải 6 (G6)", "G7": "Giải 7 (G7)", "G8": "Giải 8 (G8)"
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


def get_lottery_results_xosoapi(province: str, date_str: str) -> str:
    """Tra cứu kết quả xổ số từ XoSoAPI."""
    slug = get_province_slug(province)
    fmt_date = format_date_for_api(date_str)

    if not slug or not fmt_date:
        return ""

    headers = {"X-API-Key": XOSOAPI_KEY, "Content-Type": "application/json"}

    # Tìm province code
    province_id = None
    try:
        prov_resp = requests.get("https://xosoapi.online/api/v1/vietnam/provinces", headers=headers, timeout=10)
        prov_resp.raise_for_status()
        provinces_data = prov_resp.json()
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
        logger.warning(f"Lỗi khi lấy danh sách tỉnh từ XoSoAPI: {e}")

    if not province_id:
        province_id = slug

    # Gọi API draws
    url = "https://xosoapi.online/api/v1/vietnam/draws"
    querystring = {"code": province_id, "date": fmt_date, "limit": 1}

    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        data = response.json()

        draws_list = data.get("data", []) if isinstance(data, dict) else data
        filtered = []
        if isinstance(draws_list, list):
            for d in draws_list:
                d_date = str(d.get("date", ""))
                d_prov = str(d.get("province", {}).get("code", ""))
                if fmt_date in d_date:
                    if str(province_id) == d_prov:
                        filtered.append(d)
            if not filtered and len(draws_list) > 0:
                filtered = draws_list

        if filtered:
            return format_lottery_table(filtered[0])

        return "Không tìm thấy dữ liệu kết quả xổ số phù hợp."
    except Exception as e:
        logger.warning(f"Lỗi khi tải dữ liệu từ XoSoAPI: {e}")
        return ""
