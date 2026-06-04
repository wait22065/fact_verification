"""
FEVER事实验证系统 - 金牌数据加载模块 (模块化整理版)
"""
import glob
import json
import os
import random
from pathlib import Path
from tqdm import tqdm

from src.config import (
    DUMP_DIR,
    GOLDEN_EVIDENCE_CACHE,
    GOLDEN_FEVER_FILE,
    RANDOM_SEED,
    REBUILD_GOLDEN_EVIDENCE_CACHE,
    SAMPLE_SIZE,
    VALID_LABELS,
)
from src.retriever import check_and_download_wiki_dump


def _clean_text(text: str) -> str:
    """还原维基转义括号"""
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("-LCB-", "{").replace("-RCB-", "}")
    return text.strip()


def _parse_lines_field(lines_str: str) -> dict[int, str]:
    """解析每一行的句子对应表"""
    result = {}
    if not isinstance(lines_str, str):
        return result

    for line in lines_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        sent_id_str, sent_text = parts
        sent_text = _clean_text(sent_text)
        if not sent_text:
            continue
        try:
            result[int(sent_id_str)] = sent_text
        except ValueError:
            continue
    return result


def _load_json_or_jsonl(file_path: str) -> list[dict]:
    """解析标注文件 (支持 JSON 数组和 JSONL)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"找不到标注数据文件：{file_path}\n"
            "请确认标注数据集放置于项目 data 目录下。"
        )

    if file_path.lower().endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data

    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"不支持的格式：{file_path}")


def _extract_evidence_items(raw_item: dict) -> list[dict]:
    """提取真实证据坐标"""
    evidence_items = []
    seen = set()

    for group in raw_item.get("evidence", []):
        if not isinstance(group, list):
            continue
        for evidence in group:
            if not isinstance(evidence, list) or len(evidence) < 4:
                continue
            page = evidence[2]
            sentence_id = evidence[3]
            if not isinstance(page, str) or not isinstance(sentence_id, int):
                continue

            key = (page, sentence_id)
            if key in seen:
                continue
            seen.add(key)
            evidence_items.append({
                "page": page,
                "sentence_id": sentence_id,
            })

    return evidence_items


def _sample_valid_fever_items(raw_data: list[dict], sample_size: int, seed: int) -> list[dict]:
    data_list = []
    for item in raw_data:
        claim = item.get("claim")
        label = item.get("label")
        if not claim or not isinstance(claim, str):
            continue
        if label not in VALID_LABELS:
            continue

        evidence_items = _extract_evidence_items(item)
        data_list.append({
            "id": item.get("id"),
            "claim": claim,
            "label": label,
            "evidence_items": evidence_items,
            "evidence_pages": sorted({
                ev["page"] for ev in evidence_items
            }),
        })

    if sample_size is None or sample_size <= 0 or sample_size >= len(data_list):
        return data_list

    rng = random.Random(seed)
    return rng.sample(data_list, sample_size)


def _load_sentence_cache(cache_path: str) -> dict:
    if REBUILD_GOLDEN_EVIDENCE_CACHE or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_sentence_cache(cache: dict, cache_path: str) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _collect_required_pages(data_list: list[dict]) -> set[str]:
    pages = set()
    for item in data_list:
        for evidence in item.get("evidence_items", []):
            page = evidence.get("page")
            if page:
                pages.add(page)
    return pages


def _resolve_pages_from_dump(required_pages: set[str], cache: dict) -> dict:
    """读取本地 Dump，解析需要的词条"""
    # 自动安全检查：没有维基原始数据时触发自动下载
    check_and_download_wiki_dump(DUMP_DIR)

    missing_pages = required_pages - set(cache.keys())
    if not missing_pages:
        return cache

    wiki_files = sorted(glob.glob(os.path.join(DUMP_DIR, "wiki-*.jsonl")))
    if not wiki_files:
        raise FileNotFoundError(f"未在 {DUMP_DIR} 目录下找到 wiki-*.jsonl。")

    print(f"开始扫描本地原始 Dump 还原金牌证据句：共有 {len(missing_pages)} 个页面待解析...")
    found_pages = set()

    for file_path in tqdm(wiki_files, desc="扫描原始维基数据", unit="file"):
        if not missing_pages:
            break

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = (record.get("id") or "").strip()
                if doc_id not in missing_pages:
                    continue

                sentences = _parse_lines_field(record.get("lines", ""))
                if not sentences:
                    text = _clean_text(record.get("text", ""))
                    if text:
                        sentences = {0: text}

                # 保存为字符串形式的 Key 方便序列化
                cache[doc_id] = {str(k): v for k, v in sentences.items()}
                missing_pages.remove(doc_id)
                found_pages.add(doc_id)

                if not missing_pages:
                    break

    return cache


def _attach_gold_evidence(data_list: list[dict], sentence_cache: dict) -> list[dict]:
    """将坐标和正文映射，生成格式化后的金牌证据"""
    for item in data_list:
        lines = []
        seen = set()
        for evidence in item.get("evidence_items", []):
            page = evidence.get("page")
            sentence_id = evidence.get("sentence_id")
            sentence = sentence_cache.get(page, {}).get(str(sentence_id))
            if not sentence:
                continue

            key = (page, sentence_id, sentence)
            if key in seen:
                continue
            seen.add(key)
            readable_title = page.replace("_", " ")
            lines.append(f"[{len(lines) + 1}] ({readable_title}, 句#{sentence_id}) {sentence}")

        item["gold_evidence"] = lines
    return data_list


def load_fever_gold_data(
    file_path: str = GOLDEN_FEVER_FILE,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    cache_path: str = GOLDEN_EVIDENCE_CACHE,
) -> list[dict]:
    """金牌证据数据加载器外部接口"""
    print(f"正在从标注文件加载金牌评估集：{file_path}")
    raw_data = _load_json_or_jsonl(file_path)
    data_list = _sample_valid_fever_items(raw_data, sample_size=sample_size, seed=seed)

    required_pages = _collect_required_pages(data_list)
    sentence_cache = _load_sentence_cache(cache_path)
    sentence_cache = _resolve_pages_from_dump(required_pages, sentence_cache)
    _save_sentence_cache(sentence_cache, cache_path)

    data_list = _attach_gold_evidence(data_list, sentence_cache)
    return data_list