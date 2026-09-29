#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能文档处理器：根据文档复杂度自动选择解析方案 + 结构驱动分块。

解析路径：
    打分器 ≥ 2 分 → MinerU 版面分析（输出 Markdown，含表格/图片/标题层级）
    打分器 < 2 分 → markitdown 简单文本提取

后处理校验：
    1. 乱码生僻字比例检测：> 5% 标记为低质量
    2. 表格完整性检测：检查 | 分隔行是否成对
    3. 图片引用验证：检查 MinerU 输出的图片路径是否存在
    4. 降级策略：表格不完整 → 降级为纯文本；图片缺失 → 标记并重试

分块策略：结构驱动（按 ## 小节分父块，段落/句子分子块）
    父块 ≤1200 字符（主体内容），相邻父块间 20% 重叠
    子块 ≤300 字符，20% 重叠
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
    """6 分制打分器，≥2 分走 MinerU，<2 分走 markitdown"""

    GARBAGE_CHARS = set("\ufffd\u25a1\u25a0\u25b2\u25cf\u2605\u2606\u2666\u266a\u266b\u201a\u0192\u201e\u2026\u2020\u2021\u2030\u2039\u203a\u20ac\u2122\u02dc")
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
        total_tables = 0
        garbage_count = 0
        tables_found = False

        for page_idx, page in enumerate(doc):
            text = page.get_text()
            total_text_chars += len(text.strip())
            garbage_count += self._count_garbage(text)
            # 图片统计：过滤装饰性小图（像素短边 <20px，或显示面积 < 页面 1%），
            # 避免页眉 logo/公司印章让 dense_images 误命中，把所有文档都推向 MinerU
            page_area = page.rect.width * page.rect.height
            for img in page.get_image_info():
                w = img.get("width", 0) or 0
                h = img.get("height", 0) or 0
                if min(w, h) < 20:
                    continue
                bbox = img.get("bbox")
                if bbox and len(bbox) == 4 and page_area > 0:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area / page_area < 0.01:
                        continue
                total_images += 1
            blocks = page.get_text("blocks")
            for b in blocks:
                if b[4].strip():
                    width = b[2] - b[0]
                    column_widths.append(width)
            footer_area = page.rect.height - 80
            footer_blocks = [b for b in blocks if b[1] > footer_area and b[4].strip()]
            if footer_blocks:
                footer_texts.append(footer_blocks[0][4].strip()[:50])
            # 真正检测表格（PyMuPDF 1.23+ API）；跳过封面页（封面多为装饰性公司信息表），
            # 找到第一张正文表即停止（has_tables 是布尔特征，无需统计全部表）
            if not tables_found:
                if page_idx == 0 and len(re.sub(r"\s", "", text)) < 300:
                    continue
                try:
                    tables = page.find_tables()
                    if tables and tables.tables:
                        total_tables += len(tables.tables)
                        tables_found = True
                except Exception:
                    pass

        doc.close()

        avg_chars_per_page = total_text_chars / max(self.page_count, 1)

        # 1. 只要有图片就走 MinerU
        features["dense_images"] = total_images > 0
        # 2. 表格/复杂版式（真正检测表格，不再用图片数近似）
        features["has_tables"] = total_tables > 0
        # 3. 文本稀疏
        features["sparse_text"] = avg_chars_per_page < 500
        # 4. 多栏排版
        if len(column_widths) > 5:
            avg_width = sum(column_widths) / len(column_widths)
            variance = sum((w - avg_width) ** 2 for w in column_widths) / len(column_widths)
            features["multi_column"] = variance > (avg_width * 0.3) ** 2
        else:
            features["multi_column"] = False
        # 5. 乱码生僻字比例（全量统计，已在循环中累加）
        features["high_garbage"] = garbage_count / max(total_text_chars, 1) > 0.05
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
    def _count_garbage(text: str) -> int:
        """数文本中乱码/生僻字个数"""
        GARBAGE_CHARS = DocumentComplexityScorer.GARBAGE_CHARS
        RARE_CHARS_RANGE = DocumentComplexityScorer.RARE_CHARS_RANGE
        cnt = 0
        for ch in text:
            if ch in GARBAGE_CHARS:
                cnt += 1
            elif RARE_CHARS_RANGE[0] <= ord(ch) <= RARE_CHARS_RANGE[1]:
                cnt += 1
        return cnt


# ==================== 解析器：MinerU / markitdown ====================
class DocumentParser:
    """根据打分结果选择解析器，输出标准化 Markdown"""

    def __init__(self, file_path: str, score: int, features: Dict):
        self.file_path = file_path
        self.score = score
        self.features = features
        self.use_mineru = score >= 2
        self.md_content = ""
        self.image_dir = ""
        self.actual_method = ""  # 记录实际用的解析方式
        self.extracted_images = []  # 提取的图片列表
        self.page_signatures = []  # 每页文本签名（去空白），用于子块→页码反查

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
        text = result.markdown()
        # 构建每页文本签名（blocks 原始文本），供子块切分后反查页码
        self._build_page_signatures_from_mineru(result)
        # 提取图片 base64，正文替换成占位符
        text = self._extract_images(text)
        return text

    def _parse_with_markitdown(self) -> str:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(self.file_path)
        text = result.text_content
        # 用 PDF 文本层构建每页签名（markitdown 无结构化页码，走文本层反查）
        self._build_page_signatures_from_pdf()
        # 提取图片 base64
        text = self._extract_images(text)
        return text

    def _collect_block_text(self, obj, out: list):
        """递归提取 block/span 中的全部文本字符串（含表格 HTML 内容）"""
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                self._collect_block_text(x, out)
        elif obj is not None and hasattr(obj, "content"):
            self._collect_block_text(obj.content, out)

    def _build_page_signatures_from_mineru(self, result):
        """从 MinerU 每页 blocks 提取文本，构建页签名（去空白）"""
        import re
        sigs = []
        try:
            for pg in result.pages:
                texts = []
                for b in pg.blocks:
                    self._collect_block_text(b, texts)
                sigs.append(re.sub(r"\s+", "", "".join(texts)))
        except Exception as e:
            logger.warning(f"[页码签名] MinerU 页签名构建失败: {e}")
            sigs = []
        self.page_signatures = sigs
        logger.info(f"[页码签名] 已构建 {len(sigs)} 页签名")

    def _build_page_signatures_from_pdf(self):
        """用 PyMuPDF 文本层构建页签名（markitdown 路径）"""
        import re
        try:
            import fitz
            doc = fitz.open(self.file_path)
            self.page_signatures = [re.sub(r"\s+", "", p.get_text()) for p in doc]
            doc.close()
            logger.info(f"[页码签名] PDF 文本层构建 {len(self.page_signatures)} 页签名")
        except Exception as e:
            logger.warning(f"[页码签名] PDF 文本层签名失败: {e}")
            self.page_signatures = []

    def _extract_images(self, text: str) -> str:
        """从 Markdown 中提取 base64 图片，存到 self.extracted_images，正文替换成占位符"""
        import re
        pattern = r'!\[([^\]]*)\]\(data:image/([a-zA-Z]+);base64,([^)]+)\)'
        
        idx = 0
        def replacer(match):
            nonlocal idx
            caption = match.group(1)
            img_type = match.group(2)
            img_data = match.group(3)
            self.extracted_images.append({
                'image_data': img_data,
                'image_type': img_type,
                'image_caption': caption[:500] if caption else ''
            })
            placeholder = f'[IMAGE_{idx}]'
            idx += 1
            return f'![{caption}]({placeholder})'
        
        text = re.sub(pattern, replacer, text)
        
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 100 and re.match(r'^[A-Za-z0-9+/=\s]+$', stripped):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

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
        garbage_count = DocumentComplexityScorer._count_garbage(text)
        return garbage_count / max(len(text), 1)

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
        """验证 Markdown 中的图片引用路径是否存在。

        - `[IMAGE_n]` 占位符是 _extract_images 已提取成功的 base64 图片，跳过（图片本体已入库）；
        - 真实路径引用缺失时删除整条引用标记，不留"图片缺失"占位文字污染正文与检索。
        """
        img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        matches = img_pattern.findall(self.md_content)
        if not matches:
            return
        # 已提取的 base64 图片占位符 [IMAGE_n] 不是路径引用，跳过
        real_refs = [p for p in matches if not re.match(r'^\[IMAGE_\d+\]$', p.strip())]
        if not real_refs:
            return
        missing = []
        for img_path in real_refs:
            full_path = img_path if os.path.isabs(img_path) else os.path.join(self.image_dir, img_path)
            if not os.path.exists(full_path):
                missing.append(img_path)
        if missing:
            logger.warning(f"[校验] {len(missing)} 张图片引用缺失，已从正文移除引用标记")
            for img_path in missing:
                # 删除 ![...](路径) 整条引用标记，图片本体未提取成功，不留无意义占位
                self.md_content = re.sub(
                    r'!\[[^\]]*\]\(' + re.escape(img_path) + r'\)', '', self.md_content)

# ==================== 结构驱动分块器 ====================
class StructureAwareSplitter:
    """按 Markdown 标题层级分块：## 小节为父块，段落为子块，父块最大1200字符"""

    def __init__(self,
                 min_parent_size: int = 200,
                 max_parent_size: int = 1200,
                 min_child_size: int = 100,
                 max_child_size: int = 350,
                 overlap_ratio: float = 0.2):
        self.min_parent = min_parent_size
        self.max_parent = max_parent_size
        self.min_child = min_child_size
        self.max_child = max_child_size
        self.overlap = int(max_child_size * overlap_ratio)
        # 父块间重叠量：max_parent * 20%
        self.parent_overlap = int(max_parent_size * overlap_ratio)

    def split(self, markdown_text: str, metadata: dict, cover_text: str = "") -> List[Document]:
        # 预处理0：封面页保护——PDF 第一页去空白 <300 字符视为封面（标题+图片为主），
        # 整页替换为独立"文件封面"父块，避免"关于/日期"碎片入库、发布机构行丢失
        if cover_text and cover_text.strip():
            markdown_text = self._protect_cover(markdown_text, cover_text)
        # 预处理1：清洗 markitdown 伪表格残留（裸分隔线行、普通文本里夹的 |）
        markdown_text = self._clean_pseudo_table_noise(markdown_text)
        # 预处理1.5：剔除目录（TOC）条目行——"第X章/节...点号+页码"导航行，
        # 无语义价值且点线是向量化噪声；章节名在正文中本就有 heading 块
        markdown_text = self._strip_toc_lines(markdown_text)
        # 预处理2：把完整表格块保护成占位符，避免父块切分把表格从中间截断
        # （截断后的表格碎片特征失真，伪表格会漏过判定）
        markdown_text = self._protect_tables(markdown_text)
        # 按标题层级切块：(heading, level, heading_path, content)
        parent_blocks = self._split_by_headings(markdown_text)
        # 第一步：超过max_parent的按段落拆成多个父块（保留层级）
        splited_parents = []
        for heading, level, path, content in parent_blocks:
            if len(content) > self.max_parent:
                # 按段落拆分超大父块
                paras = [p.strip() for p in content.split("\n\n") if p.strip()]
                current_para = ""
                for p in paras:
                    if len(current_para) + len(p) + 2 > self.max_parent and current_para:
                        splited_parents.append((heading, level, path, current_para))
                        current_para = p
                    else:
                        current_para = current_para + "\n\n" + p if current_para else p
                if current_para:
                    splited_parents.append((heading, level, path, current_para))
            else:
                splited_parents.append((heading, level, path, content))

        # 第二步：按句子兜底拆分，产出最终父块列表（主体内容均不超过max_parent）
        final_parents = []
        for heading, level, path, content in splited_parents:
            if len(content) > self.max_parent:
                # 按句子拆分（保留换行，避免句号行与 | 表格行被错误拼接）
                sentences = self._split_sentences_keep_newlines(content)
                current_para = ""
                sub_idx = 0
                for sent in sentences:
                    if not sent.strip():
                        continue
                    # 单句仍超长：按字符硬切，保证父块主体不超过 max_parent
                    if len(sent) > self.max_parent:
                        for piece in self._hard_split(sent, self.max_parent, 0):
                            if current_para:
                                final_parents.append((f"{heading}_part{sub_idx}", level, path, current_para))
                                sub_idx += 1
                            current_para = piece
                        continue
                    if len(current_para) + len(sent) > self.max_parent and current_para:
                        final_parents.append((f"{heading}_part{sub_idx}", level, path, current_para))
                        current_para = sent
                        sub_idx += 1
                    else:
                        current_para += sent
                if current_para:
                    final_parents.append((f"{heading}_part{sub_idx}", level, path, current_para))
            else:
                final_parents.append((heading, level, path, content))

        # 第三步：父块不再做 overlap（避免跨父块大段重复；子块层已有 overlap 足够）
        all_children = []
        for idx, (heading, level, path, content) in enumerate(final_parents):
            parent_id = f"parent_{idx}"
            parent_text = f"# {heading}\n{content}".strip()
            children = self._split_into_children(content, parent_id, parent_text, heading, level, path)
            all_children.extend(children)
        # 跨子块前缀去重：相邻块开头若与上一块尾部大段重复，截掉（双保险）
        all_children = self._dedup_prefix_overlap(all_children)

        # 表格去重：同一张表的 HTML/Markdown 多格式只保留一份（单元格指纹相似 >0.9 判定同表）
        all_children = self._dedup_table_chunks(all_children)
        # 短子块过滤：去空白后 <=4 字符的碎片（封面标题残片/孤立短词）不入库，避免污染检索
        all_children = [d for d in all_children if len(d.page_content.strip()) > 4]

        for doc in all_children:
            doc.metadata.update(metadata)
        return all_children

    def _protect_cover(self, md: str, cover_text: str) -> str:
        """把 md 开头的封面区整体替换为"文件封面"父块。

        - 封面区收集规则（兼容 MinerU 标题标记不稳定：同一 PDF 有时输出 # 标题、有时纯文本）：
          从 md 开头逐行收集标题行、图片行、空行，以及"去空白后出现在封面文本中"的短行；
          遇到长正文行或不属于封面的短行（如第 2 页标题"重要提示"）即结束；
        - 封面文本取 PyMuPDF 第一页全文（发布机构/标题/日期不丢失）；
        - 保留封面区图片占位符（[IMAGE_n]），印章/logo 仍可关联入库。"""
        lines = md.split("\n")
        cover_lines = {re.sub(r"\s", "", x) for x in cover_text.split("\n") if x.strip()}
        cover_region = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            s = re.sub(r"\s", "", ln)
            if re.search(r'\[IMAGE_\d+\]', ln):
                cover_region.append(ln)
                i += 1
                continue
            if re.match(r'^#\s+\S', ln):
                # 一级标题：封面标题（MinerU 对封面页标题常用 #）
                cover_region.append(ln)
                i += 1
                continue
            if re.match(r'^#{2,3}\s+\S', ln):
                # 二级/三级标题：正文标题（如"## 重要提示"），封面区到此为止
                break
            if not s:
                cover_region.append(ln)
                i += 1
                continue
            if len(s) <= 40 and s in cover_lines:
                cover_region.append(ln)
                i += 1
                continue
            # 长正文行，或不在封面文本中的短行（第 2 页标题等）：封面区到此为止
            break
        img_lines = [ln for ln in cover_region if re.search(r'\[IMAGE_\d+\]', ln)]
        new_block = "## 文件封面\n" + cover_text.strip()
        if img_lines:
            new_block += "\n\n" + "\n".join(img_lines)
        rest = lines[i:]
        return new_block + "\n\n" + "\n".join(rest)

    def _protect_tables(self, text: str) -> str:
        """把完整的 Markdown/HTML 表格块替换为占位符 [[TBLn]]，
        保证父块切分、句子切分、重叠都不会把表格从中间截断。
        原始表格块暂存到 self._protected_tables，由 _restore_tables 还原。"""
        lines = text.split("\n")
        result = []
        self._protected_tables = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if self._is_html_table(line) and "<table" in line:
                # HTML 表格：从 <table 到 </table>
                block = [line]
                if "</table>" not in line:
                    i += 1
                    while i < len(lines) and "</table>" not in lines[i]:
                        block.append(lines[i])
                        i += 1
                    if i < len(lines):
                        block.append(lines[i])
                self._protected_tables.append("\n".join(block))
                result.append(f"[[TBL{len(self._protected_tables) - 1}]]")
                i += 1
                continue
            if self._is_table_line(stripped):
                # Markdown 表格块：连续表格行（含分隔线行）
                block = [line]
                i += 1
                while i < len(lines):
                    s = lines[i].strip()
                    if self._is_table_line(s) or self._is_table_separator(s):
                        block.append(lines[i])
                        i += 1
                    else:
                        break
                self._protected_tables.append("\n".join(block))
                result.append(f"[[TBL{len(self._protected_tables) - 1}]]")
                continue
            result.append(line)
            i += 1
        return "\n".join(result)

    def _restore_tables(self, text: str) -> str:
        """把占位符 [[TBLn]] 还原为原始表格文本"""
        if "[[TBL" not in text:
            return text
        def repl(m):
            try:
                idx = int(m.group(1))
                return self._protected_tables[idx]
            except (IndexError, ValueError):
                return m.group(0)
        return re.sub(r'\[\[TBL(\d+)\]\]', repl, text)

    def _safe_tail(self, content: str, size: int) -> str:
        """取 content 尾部 size 字符作为父块重叠尾巴，供下一父块开头重叠。

        表格是完整语义单元，不随父块重叠重复入库：
        - 重叠区域内出现的表格占位符（[[TBLn]]）一律整行剔除；
        - 短父块（len <= size）整体作为尾巴时同样剔除其中的占位符；
        - 多个占位符同时出现时全部剔除（旧实现只处理第一个，后续占位符
          仍会留在尾巴里被复制到后续父块）。"""
        if len(content) <= size:
            tail = content
        else:
            tail = content[len(content) - size:]
        if "[[TBL" in tail:
            lines = tail.split("\n")
            tail = "\n".join(l for l in lines if "[[TBL" not in l)
        return tail
    def _split_by_headings(self, text: str) -> List[Tuple[str, int, str, str]]:
        """按 #/##/### 标题层级切块，返回 (heading, level, heading_path, content)。
        heading_path 为完整标题路径，如 '一、基本情况 > 1.1 公司概况'。"""
        lines = text.split("\n")
        blocks = []
        path_stack = []  # 标题路径栈
        current_heading = "概述"
        current_level = 0
        current_content = []

        for line in lines:
            m = re.match(r'^(#{1,3})\s+(.*)', line)
            if m:
                if current_content:
                    blocks.append((current_heading, current_level,
                                   " > ".join(path_stack) or current_heading,
                                   "\n".join(current_content)))
                level = len(m.group(1))
                title = m.group(2).strip()
                # 维护路径栈：遇到同层或更浅层的新标题，先弹出旧标题
                while path_stack and len(path_stack) >= level:
                    path_stack.pop()
                path_stack.append(title)
                current_heading = title
                current_level = level
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            blocks.append((current_heading, current_level,
                           " > ".join(path_stack) or current_heading,
                           "\n".join(current_content)))
        return blocks if blocks else [("全文", 0, "全文", text)]

    def _split_sentences_keep_newlines(self, text: str) -> List[str]:
        """按句子/换行拆分，但把换行符保留在上一片末尾。
        避免 re.split(r'(?<=[。！？；\\n])') 把 \n 单独切成一片后丢弃，
        导致"句号结尾的正文行"与"| 开头的表格行"被拼成一行（残留管道符）。"""
        parts = re.split(r'(?<=[。！？；\n])', text)
        merged = []
        for p in parts:
            if p in ('\n', '\r\n') and merged:
                merged[-1] += p
            else:
                merged.append(p)
        return merged

    def _is_table_line(self, line: str) -> bool:
        """判断一行是否是 Markdown 表格行。
        标准 Markdown 表格行必须以 | 开头（表头/分隔线/数据行都是），
        普通文本中间夹 |（如 markitdown 误拼的行）不算表格行。"""
        stripped = line.strip()
        return stripped.startswith("|") and "|" in stripped[1:]

    def _is_table_separator(self, line: str) -> bool:
        """判断一行是否是表格分隔线"""
        stripped = line.strip()
        if not stripped:
            return False
        # 分隔线格式：| --- | --- | 或 ---- | ---- |
        return bool(re.match(r'^\s*\|?[\s\-:|]+\|?\s*$', line)) and ("---" in line or "-" * 5 in line)

    def _is_html_table(self, line: str) -> bool:
        """判断一行是否包含 HTML 表格标签"""
        return "<table" in line or "<tr>" in line or "<td>" in line

    def _clean_pseudo_table_noise(self, text: str) -> str:
        """清洗 markitdown 伪表格残留，避免污染正文与检索：
        1. 裸分隔线行（含 --- 但不以 | 开头，如 '------- | ---- |'）→ 删除
           （真表格分隔线一定以 | 开头，会在表格块内被保留）
        2. 非表格行中夹的 | → 替换为空格（markitdown 把正文拼进伪表格的痕迹，
           如 '……重大信息。| 3.公司已于 |……'）"""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if self._is_table_line(stripped):
                cleaned.append(line)  # 真表格行（以 | 开头）不动
                continue
            if self._is_table_separator(stripped):
                continue  # 裸分隔线：markitdown 伪表格残留，删除
            if "|" in line:
                line = line.replace("|", " ")
            cleaned.append(line)
        return "\n".join(cleaned)

    # 目录行：可选列表前缀(-/*/•) + "第X章/节/编/篇" 开头 + 任意标题 + 结尾连续点号(>=2) + 页码
    _TOC_LINE_RE = re.compile(
        r"^\s*[-*\u2022]?\s*第[一二三四五六七八九十百零\d]+[章节编篇部分].*?\.{2,}\s*\d{1,5}\s*$"
    )

    def _strip_toc_lines(self, text: str) -> str:
        """剔除目录(TOC)条目行。目录是导航元数据（章节名+页码），对 RAG 检索无语义价值，
        且 '....' 点线是向量化噪声；章节标题在正文中本就有 heading 块，剔除无信息损失。
        只删目录条目行，保留 '目 录' 标题行与正文段落。"""
        lines = text.split("\n")
        kept, removed = [], 0
        for line in lines:
            stripped = line.strip()
            if stripped and self._TOC_LINE_RE.match(stripped):
                removed += 1
                continue
            kept.append(line)
        if removed:
            out = "\n".join(kept)
            out = re.sub(r"\n{3,}", "\n\n", out)  # 清理删行留下的连续空行
            return out
        return text

    def _is_fake_table(self, lines: List[str]) -> bool:
        """增强版伪表格判定。markitdown 常把"空格对齐的正文文本"误判为表格，
        其特征与真表格有明显差异，命中任一条即判定为伪表格：
          1. 行数 < 3 → 伪（表格至少 表头+分隔线+1 数据行）
          2. 列数不一致（差异 > 1）→ 伪
          3. 非空单元格比例 < 60% → 伪（markitdown 伪表格大量空列填充对齐）
          4. 分隔线密集：两条分隔线之间数据行 < 3 → 伪（真表格分隔线只在表头下一行）
          5. 表头（第一条分隔线前一行）过长或含句末标点 → 伪（表头应是短词标签）
        """
        if len(lines) < 3:
            return True  # 太少，不认为是表格
        table_lines = [l for l in lines if "|" in l]
        if len(table_lines) < 3:
            return True
        # 2. 列数不一致则是伪表格
        pipe_counts = [self._count_pipes(l) for l in table_lines]
        if max(pipe_counts) - min(pipe_counts) > 1:
            return True
        # 3. 非空单元格比例（排除分隔线行；伪表格靠大量空单元格对齐列宽）
        data_lines = [l for l in table_lines if not self._is_table_separator(l)]
        if len(data_lines) < 2:
            return True
        total_cells = 0
        nonempty_cells = 0
        for line in data_lines:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            total_cells += len(cells)
            nonempty_cells += sum(1 for c in cells if c)
        if total_cells and nonempty_cells / total_cells < 0.6:
            return True
        # 4. 分隔线密集：相邻两条分隔线之间数据行 < 3 → 伪
        sep_idxs = [i for i, l in enumerate(lines) if self._is_table_separator(l)]
        if len(sep_idxs) >= 2:
            for a, b in zip(sep_idxs, sep_idxs[1:]):
                data_rows = sum(1 for i in range(a + 1, b)
                                if not self._is_table_separator(lines[i]) and lines[i].strip())
                if data_rows < 3:
                    return True
        # 5. 表头异常：第一条分隔线前一行（表头）的单元格应是短名词标签；
        #    长句表头（>14 字符且含逗号）或带句末标点 → 伪（markitdown 会把句子碎片当表头）；
        #    纯长度长不算伪（如日期区间"2026年3月31日/2026年1-3月"是合法表头标签）
        for idx, line in enumerate(lines):
            if self._is_table_separator(line):
                if idx > 0:
                    header_cells = [c.strip() for c in lines[idx - 1].strip().strip("|").split("|") if c.strip()]
                    for cell in header_cells:
                        if re.search(r'[。！？；]$', cell):
                            return True
                        if len(cell) > 14 and re.search(r'[，,]', cell):
                            return True
                break
        return False

    def _table_block_to_text(self, lines: List[str]) -> str:
        """伪表格降级：把 | 分隔的碎片还原为连续纯文本。
        过滤分隔线行（---- 之类），去掉 | 与冗余空格，按行拼接。"""
        out_lines = []
        for line in lines:
            if self._is_table_separator(line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # 丢弃纯分隔线碎片的单元格（如 ----）
            cells = [c for c in cells if c and not re.fullmatch(r'[-:\s]+', c)]
            if not cells:
                continue
            out_lines.append(" ".join(cells))
        return "\n".join(out_lines)

    def _merge_table_fragments(self, content: str) -> List[Tuple[str, bool]]:
        """把表格碎片合并成完整表格块。
        返回 [(block_text, is_table), ...]"""
        lines = content.split("\n")
        blocks = []
        current_block = []
        current_is_table = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 检测 HTML 表格
            if self._is_html_table(line):
                # HTML 表格：整体作为一个块
                if current_is_table:
                    blocks.append(("\n".join(current_block), True))
                    current_block = []
                    current_is_table = False
                # 找 HTML 表格结尾
                html_lines = [line]
                if "</table>" not in line:
                    i += 1
                    while i < len(lines):
                        html_lines.append(lines[i])
                        if "</table>" in lines[i]:
                            break
                        i += 1
                blocks.append(("\n".join(html_lines), True))
                i += 1
                continue
            
            # 检测是否是 Markdown 表格行
            is_table_line = self._is_table_line(line)
            
            if is_table_line:
                # 开始或继续表格块
                if not current_is_table:
                    # 先把之前的非表格块 flush 掉
                    if current_block:
                        blocks.append(("\n".join(current_block), False))
                        current_block = []
                    current_is_table = True
                current_block.append(line)
            else:
                # 非表格行
                if current_is_table:
                    # 检查这行是不是表格分隔线（可能没 |）
                    if self._is_table_separator(line):
                        current_block.append(line)
                    else:
                        # 表格块结束
                        # 验证是不是真表格（列数一致、非空率高、分隔线不密集）
                        if self._is_fake_table(current_block):
                            # 伪表格，降级为纯文本（去掉 | 还原句子）
                            text = self._table_block_to_text(current_block)
                            if text:
                                blocks.append((text, False))
                        else:
                            blocks.append(("\n".join(current_block), True))
                        current_block = []
                        current_is_table = False
                        current_block.append(line)
                else:
                    current_block.append(line)
            i += 1
        
        # 最后一块
        if current_block:
            if current_is_table and self._is_fake_table(current_block):
                text = self._table_block_to_text(current_block)
                if text:
                    blocks.append((text, False))
            else:
                blocks.append(("\n".join(current_block), current_is_table))
        
        return blocks

    def _count_pipes(self, line: str) -> int:
        """统计一行中的 | 数量"""
        return line.count("|")

    def _validate_and_fix_table(self, table_text: str) -> str:
        """验证并修复表格：
        1. 列数一致性检查
        2. 表头分隔线补全
        3. 去掉空行
        """
        lines = [l.strip() for l in table_text.split("\n") if l.strip()]
        if not lines:
            return table_text

        # 双保险：若整体仍是伪表格（如 markitdown 误判的正文），降级为纯文本
        if self._is_fake_table(lines):
            return self._table_block_to_text(lines)
        
        # 找表头和分隔线
        header_idx = -1
        separator_idx = -1
        for idx, line in enumerate(lines[:5]):
            if self._is_table_separator(line):
                separator_idx = idx
                header_idx = idx - 1
                break
        
        # 没有分隔线则补全
        if separator_idx < 0 and header_idx < 0:
            # 第一行作为表头
            header_idx = 0
            # 统计表头列数
            header_pipes = self._count_pipes(lines[header_idx])
            # 生成分隔线
            separator = "| " + " | ".join(["---"] * (header_pipes - 1)) + " |"
            lines.insert(1, separator)
            separator_idx = 1
        
        # 列数一致性验证
        if header_idx >= 0:
            header_pipes = self._count_pipes(lines[header_idx])
            # 检查每一行的列数
            fixed_lines = []
            for line in lines:
                line_pipes = self._count_pipes(line)
                if line_pipes == 0:
                    # 非表格行，跳过
                    continue
                if line_pipes != header_pipes:
                    # 列数不一致，尝试补全
                    if line_pipes < header_pipes:
                        # 少列，补 |
                        missing = header_pipes - line_pipes
                        if line.endswith("|"):
                            line = line + " |" * missing
                        else:
                            line = line + " |" * missing
                fixed_lines.append(line)
            lines = fixed_lines
        
        return "\n".join(lines)

    def _split_table(self, table: str) -> List[str]:
        """拆分超长表格：按行拆，保留表头"""
        lines = table.split("\n")
        if len(lines) <= 3:
            return [table]
        # 找表头和分隔线
        header_idx = -1
        separator_idx = -1
        for idx, line in enumerate(lines[:5]):  # 前5行内找
            if self._is_table_separator(line):
                separator_idx = idx
                header_idx = idx - 1 if idx > 0 else 0
                break
        
        header = lines[header_idx] if header_idx >= 0 else ""
        separator = lines[separator_idx] if separator_idx >= 0 else ""
        data_start = separator_idx + 1 if separator_idx >= 0 else 0
        data_lines = lines[data_start:]
        
        # 每 20 行一组
        chunks = []
        current = [header, separator] if header and separator else []
        for line in data_lines:
            current.append(line)
            if len(current) >= 22:  # 表头+分隔+20行数据
                chunks.append("\n".join(current))
                current = [header, separator] if header and separator else []
        if current and (len(current) > 2 or not header):
            chunks.append("\n".join(current))
        return chunks if chunks else [table]

    def _split_into_children(self, content: str, parent_id: str, parent_text: str,
                             heading: str, heading_level: int, heading_path: str) -> List[Document]:
        """切子块：先还原表格占位符，再合并表格碎片，最后切分。"""
        # 第一步：还原表格占位符（父块切分阶段表格被保护为 [[TBLn]]）
        content = self._restore_tables(content)
        parent_text = self._restore_tables(parent_text)
        # 第二步：合并表格碎片
        blocks = self._merge_table_fragments(content)
        
        children = []
        chunk = ""
        child_idx = 0
        limit = self.max_child - self.overlap

        def flush():
            nonlocal chunk, child_idx
            if chunk:
                children.append(self._make_child(chunk, parent_id, child_idx, parent_text,
                                                 heading, heading_level, heading_path))
                child_idx += 1
                chunk = ""

        for block_text, is_table in blocks:
            block_text = block_text.strip()
            if not block_text:
                continue
            
            if is_table:
                # 表格块：先 flush 掉之前的文本
                flush()
                # 验证并修复表格
                block_text = self._validate_and_fix_table(block_text)
                if len(block_text) > 2000:
                    # 超长表格按行拆
                    for piece in self._split_table(block_text):
                        children.append(self._make_child(piece, parent_id, child_idx, parent_text,
                                                         heading, heading_level, heading_path, is_table=True))
                        child_idx += 1
                else:
                    # 短表格整体保留
                    children.append(self._make_child(block_text, parent_id, child_idx, parent_text,
                                                     heading, heading_level, heading_path, is_table=True))
                    child_idx += 1
            else:
                # 非表格块：按段落切分
                paragraphs = [p.strip() for p in block_text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    if len(para) > limit:
                        # 超长段：按句子拆
                        sentences = re.split(r'(?<=[。！？；])', para)
                        for sent in sentences:
                            if not sent.strip():
                                continue
                            if len(sent) > self.max_child:
                                flush()
                                for piece in self._hard_split(sent, self.max_child, self.overlap):
                                    chunk = piece
                                    flush()
                                continue
                            if len(chunk) + len(sent) > limit and chunk:
                                flush()
                                chunk = self._tail_at_boundary(chunk, self.overlap) + sent
                            else:
                                chunk = chunk + sent if chunk else sent
                    else:
                        if len(chunk) + len(para) > limit and chunk:
                            flush()
                            chunk = self._tail_at_boundary(chunk, self.overlap) + "\n\n" + para
                        else:
                            chunk = chunk + "\n\n" + para if chunk else para
        flush()
        return children

    # 句界标点（用于 overlap 回退与硬切断点选择）
    _SENT_BOUNDARY = "。！？；\n"
    _SOFT_PUNCT = "，、；。！？\n"

    def _tail_at_boundary(self, text: str, keep: int) -> str:
        """取 text 尾部约 keep 字作为新块开头，但回退到最近句界，
        避免新块从词/列举中间（如'、子女'）开始。"""
        if len(text) <= keep:
            return text
        anchor = len(text) - keep
        search_start = max(0, anchor - 40)
        best = -1
        for m in re.finditer(r"[。！？；\n]", text[search_start:anchor]):
            pos = search_start + m.end()
            if pos > best:
                best = pos
        # 句界不能回退过头（最多再退 keep 字）
        if best >= 0 and best >= anchor - keep:
            return text[best:]
        return text[anchor:]

    def _hard_split(self, text: str, size: int, overlap: int):
        """超长文本切片；优先在标点处断，避免从词中间切断；片间保留 overlap 重叠。"""
        if len(text) <= size:
            return [text]
        pieces = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # 在 [end-40, end] 找最近标点作断点
                window_start = max(start, end - 40)
                best = -1
                for punct in self._SOFT_PUNCT:
                    p = text.rfind(punct, window_start, end)
                    if p > best:
                        best = p
                if best > start + 20:  # 至少保留 20 字内容才回退
                    end = best + 1
            pieces.append(text[start:end])
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return pieces

    def _longest_overlap(self, tail: str, head: str) -> int:
        """tail 后缀与 head 前缀的最长公共长度（>=20 才返回）。"""
        max_n = min(len(tail), len(head), 100)
        for n in range(max_n, 19, -1):
            if tail[-n:] == head[:n]:
                return n
        return 0

    def _dedup_prefix_overlap(self, chunks):
        """相邻子块前缀去重：当前块开头若与上一块尾部大段重复（>=20字公共串），
        截掉当前块的重复前缀。表格块不动。"""
        if len(chunks) < 2:
            return chunks
        out = [chunks[0]]
        for cur in chunks[1:]:
            if cur.metadata.get("is_table"):
                out.append(cur); continue
            prev = out[-1].page_content
            ct = cur.page_content
            ov = self._longest_overlap(prev[-120:], ct[:150])
            if ov >= 20 and ov / max(1, min(80, len(ct))) > 0.5:
                ct = ct[ov:].lstrip("\n ")
                if len(ct) <= 4:  # 截后成残片则整块丢弃
                    continue
                cur.page_content = ct
            out.append(cur)
        return out

    def _make_child(self, text: str, parent_id: str, child_idx: int, parent_text: str,
                    heading: str = "", heading_level: int = 0, heading_path: str = "",
                    is_table: bool = False) -> Document:
        return Document(
            page_content=text.strip(),
            metadata={
                "parent_id": parent_id,
                "parent_content": parent_text,
                "heading": heading,
                "heading_level": heading_level,
                "heading_path": heading_path,
                "is_table": 1 if is_table else 0,
                "id": f"{parent_id}_child_{child_idx}"
            }
        )
    def _dedup_table_chunks(self, chunks: List[Document]) -> List[Document]:
        """文件内表格去重：同一张表的 HTML/Markdown 多格式只保留一份。

        判定：提取表格全部单元格文本构成集合，两表单元格 Jaccard 相似度 > 0.9
        视为同一张表（不同表格内容差异远大于此阈值，不会误伤）。
        保留策略：同表多份时优先保留 Markdown（向量检索可读性好），
        仅 HTML 时保留 HTML（复杂合并表信息完整）。
        """
        tables = [(i, c) for i, c in enumerate(chunks) if c.metadata.get("is_table")]
        if len(tables) < 2:
            return chunks

        def cells_of(content: str) -> set:
            if "<table" in content:
                tds = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", content, re.S)
                cleaned = [re.sub(r"<[^>]+>", "", t).strip() for t in tds]
                return set(c for c in cleaned if c)
            cells = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("|") and "---" not in line:
                    cells.extend(c.strip() for c in line.strip("|").split("|") if c.strip())
            return set(cells)

        # 按指纹聚类：相似的表归一组
        groups = []
        for idx, c in tables:
            cell_set = cells_of(c.page_content)
            placed = False
            for g in groups:
                if cell_set and g["ref"]:
                    union = cell_set | g["ref"]
                    if len(cell_set & g["ref"]) / len(union) > 0.9:
                        g["members"].append((idx, c))
                        placed = True
                        break
            if not placed:
                groups.append({"ref": cell_set, "members": [(idx, c)]})

        keep_idx = set()
        for g in groups:
            members = g["members"]
            if len(members) == 1:
                keep_idx.add(members[0][0])
            else:
                md_members = [m for m in members if "<table" not in m[1].page_content]
                chosen = md_members[0] if md_members else members[0]
                keep_idx.add(chosen[0])
        drop_idx = {i for i, _ in tables} - keep_idx
        if not drop_idx:
            return chunks
        logger.info(f"[表格去重] {len(tables)} 个表格子块 → 保留 {len(keep_idx)}（删除 {len(drop_idx)} 个重复格式）")
        return [c for i, c in enumerate(chunks) if i not in drop_idx]


# ==================== 主流程 ====================
MINERU_READY = False  # 全局标记：MinerU 模型是否已就绪
MINERU_PARSER = None  # 全局单例的 MinerUParser 对象


def locate_page(text: str, page_signatures) -> int:
    """按子块内容反查页码（1 基）；匹配不到返回 None
    先按子块前 80 字符整段匹配，失败降级按 15 字以上片段匹配。"""
    if not page_signatures:
        return None
    key = re.sub(r"\s+", "", text[:80])
    if key:
        for i, sig in enumerate(page_signatures):
            if key in sig:
                return i + 1
    pieces = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{15,}", re.sub(r"\s+", "", text))
    for piece in pieces:
        for i, sig in enumerate(page_signatures):
            if piece in sig:
                return i + 1
    return None


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
        logger.info("MinerU 模型预热完成")
        return True
    except Exception as e:
        MINERU_READY = False
        MINERU_PARSER = None
        raise RuntimeError(f"MinerU 预热失败，终止流程：{e}") from e


def _detect_cover_text(file_path: str, threshold: int = 300) -> str:
    """封面页检测：PDF 第一页去空白字符数 < threshold 视为封面页（标题+图片为主）。
    返回第一页完整文本（含换行）；非封面页返回空串。"""
    try:
        doc = fitz.open(file_path)
        if doc.page_count == 0:
            doc.close()
            return ""
        page0 = doc[0]
        t = page0.get_text()
        doc.close()
        if t and len(re.sub(r"\s", "", t)) < threshold:
            return t
    except Exception as e:
        logger.warning(f"[封面检测] 第一页读取失败: {e}")
    return ""


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
    if score >= 2:
        warmup_mineru()
    md_text = parser.parse()
    extracted_images = parser.extracted_images
    logger.info(f"提取图片: {len(extracted_images)} 张")

    # 3. 分块
    splitter = StructureAwareSplitter()
    metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "parse_method": parser.actual_method,  # 用实际的解析方式
        "complexity_score": score,
        "processed_at": datetime.now().isoformat()
    }
    cover_text = _detect_cover_text(file_path)
    if cover_text:
        cover_len = len(re.sub(r"\s", "", cover_text))
        logger.info(f"[封面检测] 第一页去空白 {cover_len} 字符 < 300，作为封面页整页保护")
    chunks = splitter.split(md_text, metadata, cover_text=cover_text)
    # 子块→页码反查（解析器已构建 page_signatures；纯文本 markitdown 走 PDF 文本层）
    located = 0
    for c in chunks:
        c.metadata["page_no"] = locate_page(c.page_content, parser.page_signatures)
        if c.metadata.get("page_no"):
            located += 1
    logger.info(f"生成子块: {len(chunks)} 个（定位页码 {located} 个）")
    return chunks, extracted_images


if __name__ == "__main__":
    test_pdf = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\pdf_data\920030_AN202609111829278225.pdf"
    chunks = process_pdf(test_pdf)
    print(f"\n共 {len(chunks)} 个子块")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- 子块 {i} ---")
        print(c.page_content[:200])
        print(f"parent_id: {c.metadata.get('parent_id')}")
