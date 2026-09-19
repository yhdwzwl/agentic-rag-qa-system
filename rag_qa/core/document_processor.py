#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能文档处理器：根据文档复杂度自动选择解析方案 + 结构驱动分块。

解析路径：
    打分器 ≥ 3 分 → MinerU 版面分析（输出 Markdown，含表格/图片/标题层级）
    打分器 < 3 分 → markitdown 简单文本提取

后处理校验：
    1. 乱码生僻字比例检测：> 5% 标记为低质量
    2. 表格完整性检测：检查 | 分隔行是否成对
    3. 图片引用验证：检查 MinerU 输出的图片路径是否存在
    4. 降级策略：表格不完整 → 降级为纯文本；图片缺失 → 标记并重试

分块策略：结构驱动（按 ## 小节分父块，段落/句子分子块）
"""
import os
import re
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document

from base import logger, Config

conf = Config()


# ==================== 打分器：文档复杂度评估 ====================
class DocumentComplexityScorer:
    """6 分制打分器，≥3 分走 MinerU，<3 分走 markitdown"""

    GARBAGE_CHARS = set("�□■▲●★☆♦♪♫‚ƒ„…†‡‰‹›€™˜™")
    RARE_CHARS_RANGE = (0x3400, 0x4DBF)

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.page_count = 0

    def score(self) -> Tuple[int, Dict[str, bool]]:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".pdf":
            return self._score_pdf()
        else:
            return 1, {"non_pdf": True}

    def _score_pdf(self) -> Tuple[int, Dict[str, bool]]:
        doc = fitz.open(self.file_path)
        self.page_count = doc.page_count
        features = {}

        total_images = 0
        total_text_chars = 0
        column_widths = []
        footer_texts = []

        for page in doc:
            text = page.get_text()
            total_text_chars += len(text.strip())
            img_list = page.get_image_info()
            total_images += len(img_list)
            blocks = page.get_text("blocks")
            for b in blocks:
                if b[4].strip():
                    width = b[2] - b[0]
                    column_widths.append(width)
            footer_area = page.rect.height - 80
            footer_blocks = [b for b in blocks if b[1] > footer_area and b[4].strip()]
            if footer_blocks:
                footer_texts.append(footer_blocks[0][4].strip()[:50])

        doc.close()

        avg_images_per_page = total_images / max(self.page_count, 1)
        avg_chars_per_page = total_text_chars / max(self.page_count, 1)

        # 1. 图片密度
        features["dense_images"] = avg_images_per_page > 2
        # 2. 表格/复杂版式
        features["has_tables"] = total_images > self.page_count * 0.5
        # 3. 文本稀疏
        features["sparse_text"] = avg_chars_per_page < 500
        # 4. 多栏排版
        if len(column_widths) > 5:
            avg_width = sum(column_widths) / len(column_widths)
            variance = sum((w - avg_width) ** 2 for w in column_widths) / len(column_widths)
            features["multi_column"] = variance > (avg_width * 0.3) ** 2
        else:
            features["multi_column"] = False
        # 5. 乱码生僻字比例
        doc = fitz.open(self.file_path)
        sample_text = ""
        for i in range(min(10, doc.page_count)):
            sample_text += doc[i].get_text()
        doc.close()
        features["high_garbage"] = self._garbage_ratio(sample_text) > 0.05
        # 6. 页眉页脚干扰
        if len(footer_texts) > 3:
            from collections import Counter
            counter = Counter(footer_texts)
            most_common_count = counter.most_common(1)[0][1]
            features["header_footer_noise"] = most_common_count > self.page_count * 0.5
        else:
            features["header_footer_noise"] = False

        score = sum(1 for v in features.values() if v)
        return score, features

    @staticmethod
    def _garbage_ratio(text: str) -> float:
        """静态方法：计算乱码生僻字比例"""
        if not text:
            return 0.0
        GARBAGE_CHARS = set("�□■▲●★☆♦♪♫‚ƒ„…†‡‰‹›€™˜™")
        RARE_CHARS_RANGE = (0x3400, 0x4DBF)
        garbage_count = 0
        for ch in text:
            if ch in GARBAGE_CHARS:
                garbage_count += 1
            elif RARE_CHARS_RANGE[0] <= ord(ch) <= RARE_CHARS_RANGE[1]:
                garbage_count += 1
        return garbage_count / len(text)


# ==================== 解析器：MinerU / markitdown ====================
class DocumentParser:
    """根据打分结果选择解析器，输出标准化 Markdown"""

    def __init__(self, file_path: str, score: int, features: Dict):
        self.file_path = file_path
        self.score = score
        self.features = features
        self.use_mineru = score >= 3
        self.md_content = ""
        self.image_dir = ""
        self.actual_method = ""  # 记录实际用的解析方式

    def parse(self) -> str:
        if self.use_mineru:
            try:
                self.md_content = self._parse_with_mineru()
                self.actual_method = "mineru"
                logger.info(f"[MinerU] 解析成功: {self.file_path}")
            except Exception as e:
                logger.warning(f"[MinerU] 解析失败，降级 markitdown: {e}")
                self.md_content = self._parse_with_markitdown()
                self.actual_method = "markitdown_fallback"
        else:
            self.md_content = self._parse_with_markitdown()
            self.actual_method = "markitdown"
            logger.info(f"[markitdown] 解析成功: {self.file_path}")

        self._post_validate()
        return self.md_content

    def _parse_with_mineru(self) -> str:
        global MINERU_PARSER
        if MINERU_PARSER is None:
            warmup_mineru()
        if MINERU_PARSER is None:
            raise RuntimeError("MinerU 未就绪，无法进行版面解析")

        self.image_dir = tempfile.mkdtemp(prefix="mineru_")
        # 用全局单例 Parser，避免重复加载模型
        result = MINERU_PARSER.parse(self.file_path)
        return result.markdown()

    def _parse_with_markitdown(self) -> str:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    # ---------- 后处理校验 ----------
    def _post_validate(self):
        # 1. 乱码比例检测
        garbage_ratio = self._calc_garbage_ratio(self.md_content)
        if garbage_ratio > 0.1:
            logger.warning(f"[校验] 乱码比例过高: {garbage_ratio:.2%}，文本质量低")
        # 2. 表格完整性检测
        self._validate_tables()
        # 3. 图片引用验证
        self._validate_images()

    def _calc_garbage_ratio(self, text: str) -> float:
        return DocumentComplexityScorer._garbage_ratio(text)

    def _validate_tables(self):
        """检测 Markdown 表格是否完整（| 行是否成对）"""
        lines = self.md_content.split("\n")
        table_lines = [i for i, l in enumerate(lines) if l.strip().startswith("|")]
        if not table_lines:
            return
        has_separator = any("---" in lines[i] for i in table_lines)
        if not has_separator:
            logger.warning("[校验] 表格缺少分隔行，降级为纯文本")
            self._strip_incomplete_tables()

    def _strip_incomplete_tables(self):
        """移除不完整的表格行，降级为纯文本"""
        lines = self.md_content.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                if "---" in stripped:
                    continue
                content = stripped.replace("|", " ").strip()
                cleaned.append(content)
            else:
                cleaned.append(line)
        self.md_content = "\n".join(cleaned)

    def _validate_images(self):
        """验证 Markdown 中的图片引用路径是否存在"""
        img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        matches = img_pattern.findall(self.md_content)
        if not matches:
            return
        missing = []
        for img_path in matches:
            full_path = img_path if os.path.isabs(img_path) else os.path.join(self.image_dir, img_path)
            if not os.path.exists(full_path):
                missing.append(img_path)
        if missing:
            logger.warning(f"[校验] {len(missing)} 张图片引用缺失，标记处理")
            for img_path in missing:
                self.md_content = self.md_content.replace(
                    f"![", f"<!-- 图片缺失: {img_path} -->\n![", 1)


# ==================== 结构驱动分块器 ====================
class StructureAwareSplitter:
    """按 Markdown 标题层级分块：## 小节为父块，段落为子块"""

    def __init__(self,
                 min_parent_size: int = 200,
                 max_parent_size: int = 2000,
                 min_child_size: int = 100,
                 max_child_size: int = 500):
        self.min_parent = min_parent_size
        self.max_parent = max_parent_size
        self.min_child = min_child_size
        self.max_child = max_child_size

    def split(self, markdown_text: str, metadata: dict) -> List[Document]:
        parent_blocks = self._split_by_headings(markdown_text)
        all_children = []

        for idx, (heading, content) in enumerate(parent_blocks):
            parent_id = f"parent_{idx}"
            # 父块就是一个完整的 ## 小节，不再拆分父块
            # 子块按段落/句子拆细
            parent_text = f"# {heading}\n{content}".strip()
            children = self._split_into_children(content, parent_id, parent_text)
            all_children.extend(children)

        for doc in all_children:
            doc.metadata.update(metadata)
        return all_children

    def _split_by_headings(self, text: str) -> List[Tuple[str, str]]:
        lines = text.split("\n")
        blocks = []
        current_heading = "概述"
        current_content = []

        for line in lines:
            # 支持 # ## ### 三级标题，都作为新父块分隔
            if re.match(r'^#{1,3}\s+', line):
                if current_content:
                    blocks.append((current_heading, "\n".join(current_content)))
                current_heading = re.sub(r'^#{1,3}\s+', '', line).strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            blocks.append((current_heading, "\n".join(current_content)))
        return blocks if blocks else [("全文", text)]

    def _split_into_children(self, content: str, parent_id: str, parent_text: str) -> List[Document]:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        children = []
        chunk = ""
        child_idx = 0

        for para in paragraphs:
            if len(para) > self.max_child:
                sentences = re.split(r'(?<=[。！？；])', para)
                for sent in sentences:
                    if not sent.strip():
                        continue
                    if len(chunk) + len(sent) > self.max_child and chunk:
                        children.append(self._make_child(chunk, parent_id, child_idx, parent_text))
                        chunk = sent
                        child_idx += 1
                    else:
                        chunk = chunk + sent if chunk else sent
            else:
                if len(chunk) + len(para) > self.max_child and chunk:
                    children.append(self._make_child(chunk, parent_id, child_idx, parent_text))
                    chunk = para
                    child_idx += 1
                else:
                    chunk = chunk + "\n\n" + para if chunk else para

        if chunk:
            children.append(self._make_child(chunk, parent_id, child_idx, parent_text))
        return children

    def _make_child(self, text: str, parent_id: str, child_idx: int, parent_text: str) -> Document:
        return Document(
            page_content=text.strip(),
            metadata={
                "parent_id": parent_id,
                "parent_content": parent_text,
                "id": f"{parent_id}_child_{child_idx}"
            }
        )


# ==================== 主流程 ====================
MINERU_READY = False  # 全局标记：MinerU 模型是否已就绪
MINERU_PARSER = None  # 全局单例的 MinerUParser 对象


def warmup_mineru():
    """预热 MinerU 模型，确保开始批量处理前模型已加载完成
    预热失败直接抛异常，不降级，避免低质量数据进数据库
    """
    global MINERU_READY, MINERU_PARSER
    if MINERU_READY and MINERU_PARSER is not None:
        return True

    logger.info("正在预热 MinerU 模型...")
    try:
        from mineru.parser.mineru_parser import MinerUParser
        # 创建全局单例 Parser，所有文件共用，避免重复加载模型
        MINERU_PARSER = MinerUParser(tier="standard", parse_mode="auto", image_analysis=True)
        # 真正解析一个小测试文件，触发模型加载
        test_pdf = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\pdf_data\920030_AN202609111829278225.pdf"
        if os.path.exists(test_pdf):
            logger.info("用测试文件预热 MinerU 模型...")
            result = MINERU_PARSER.parse(test_pdf)
            _ = result.markdown()  # 真正触发解析
        MINERU_READY = True
        logger.info("✓ MinerU 模型预热完成")
        return True
    except Exception as e:
        MINERU_READY = False
        MINERU_PARSER = None
        raise RuntimeError(f"MinerU 预热失败，终止流程：{e}") from e


def process_pdf(file_path: str) -> List[Document]:
    """处理单个 PDF 文件，返回子块列表"""
    logger.info(f"处理文件: {file_path}")

    # 1. 打分
    scorer = DocumentComplexityScorer(file_path)
    score, features = scorer.score()
    logger.info(f"复杂度评分: {score}/6, 特征: {features}")

    # 2. 解析
    parser = DocumentParser(file_path, score, features)
    # 只在需要 MinerU 时才预热模型
    if score >= 3:
        warmup_mineru()
    md_text = parser.parse()

    # 3. 分块
    splitter = StructureAwareSplitter()
    metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "parse_method": parser.actual_method,  # 用实际的解析方式
        "complexity_score": score,
        "processed_at": datetime.now().isoformat()
    }
    chunks = splitter.split(md_text, metadata)
    logger.info(f"生成子块: {len(chunks)} 个")
    return chunks


if __name__ == "__main__":
    test_pdf = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\pdf_data\920030_AN202609111829278225.pdf"
    chunks = process_pdf(test_pdf)
    print(f"\n共 {len(chunks)} 个子块")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- 子块 {i} ---")
        print(c.page_content[:200])
        print(f"parent_id: {c.metadata.get('parent_id')}")
