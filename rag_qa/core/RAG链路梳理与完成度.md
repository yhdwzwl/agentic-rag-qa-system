# 金融RAG系统链路梳理与完成度

## 一、整体流程图（从query流向）

```
用户提问
  ↓
[1] 第一层缓存：MySQL BM25精确匹配
  ├─ 命中 → 直接返回答案（最快，毫秒级）
  ↓ 未命中
[2] 查询分类（BERT二分类）
  ↓
[3] Query改写（结合对话历史）
  ↓
[4] 第二层缓存：Redis向量缓存（改写后query语义相似匹配）
  ├─ 命中 → 复用缓存上下文，仍走 LLM 生成（省检索不省生成）
  ↓ 未命中
[5] 检索策略选择
  ↓
[6] 混合检索（Milvus稠密+稀疏）
  ↓
[7] Rerank精排
  ↓
[8] 上下文组装（父块优先）
  ↓
[9] 答案生成（LLM流式输出）
  ↓
[10] 在线质量自动评估
  ├─ 达标 → 写入缓存
  ↓ 不达标
[11] React自愈Agent（反思+重试）
  ↓
[12] 返回给用户
```

---

## 二、每个节点：当前完成度 vs 最终目标

### [1] 第一层缓存：MySQL BM25精确匹配
| 维度 | 内容 |
|------|------|
| 功能 | 库内标准问答对 BM25 原始分匹配，命中直接返回标准答案，最快（毫秒级），未命中进 RAG |
| **当前完成度** | ✅ **100%**（2026-10-01 落地） |
| 现有实现 | 1. **数据源**：`standard_qa` 表（人工校对 + LLM 动态生成，当前 102 条 active，含 evidence 依据原文列）+ `standard_qa_meta` 溯源表<br>2. **匹配**：`mysql_qa/retrieval/bm25_search.py`，BM25 **原始分**与阈值 12 直接比较（不用 softmax——实测无关 query softmax 高达 0.60 不可用；raw 分分界干净：完全相同 19~35、相似/无关 ≤9.5）<br>3. **接线**：`new_main.py`/`app.py` 查询入口先进 BM25，返回状态码 `bm25_hit`（命中直返答案）/`fallback`/`error`/`invalid`（放行 RAG）；`app.py` 修复了 `if need_rag:` truthy 误判（bm25_hit 不再被当成"需要流式RAG"）<br>4. **QA 生成器**：`rag_qa/core/generate_test_queries.py` 锚点驱动五段式管线（Stage0 分层抽样剔除封面/备查文件 → Stage1 LLM 抽事实锚点 → Stage2 类型骨架动态提问 → Stage3 双重校验[规则硬校验+LLM软校验，软校验默认关闭] → Stage4 去重+配额+落库），LLMRouter 多模型轮换（glm-5.3 额度耗尽自动切 qwen 系列） |
| 最终目标 | 1. ✅ MySQL 表存标准 QA 对（standard_qa + standard_qa_meta）<br>2. ✅ query 先进 BM25 匹配，达标直接返回标准答案，不走后续链路<br>3. ⚠️ 人工校对环节尚未做（当前靠规则硬校验 + 去重兜底，软校验关闭期间弱 QA 会入库） |
| 遗留风险 | ① QA 库扩容（数千条）后 raw 分布会变，阈值 12 需重标定（代码注释已标明）<br>② 软校验关闭期间入库的弱 QA 待治理（可用 `llm_check='na'` 标记回扫）<br>③ 数据源是 LLM 生成+自动校验，非纯人工校对，命中质量依赖生成质量 |

---

### [2] 查询分类（BERT二分类）
| 维度 | 内容 |
|------|------|
| 功能 | 判断用户query是通用闲聊，还是股票财经相关 |
| **当前完成度** | ✅ **100%** |
| 现有实现 | 本地BERT模型，准确率99.6%，区分"通用无关"和"股票财经" |
| 最终目标 | 已完成，无需改动 |

---

### [3] Query改写（结合对话历史）
| 维度 | 内容 |
|------|------|
| 功能 | 把用户口语化问题，结合对话历史，改写成更适合检索的完整问题 |
| **当前完成度** | ✅ **100%**（2026-10-01 落地） |
| 现有实现 | 1. **统一改写入口**：`new_rag_system.py` 新增 `_rewrite_query()`，所有进入 RAG 的专业咨询 query 先改写，改写结果用于策略选择与检索；通用知识不走检索不改写<br>2. **全量改写**：无论有无历史都调 LLM 改写（`prompts.py` 的 `rewrite_prompt` 模板增强：显式区分有/无历史分支）——有历史：消解"它/这个"指代、补全公司实体；无历史（首轮）：跳过指代补全，直接精炼、明确化、补金融专业关键词；**多跳复杂问题无论有无历史都展开成可独立检索的完整表述，不丢子问题**<br>3. **兜底**：LLM 异常/返回空自动回退原 query，不阻断主流程<br>4. **评估字段**：改写耗时写入 `last_query_result["rewrite_time"]`、结果写入 `rewritten_query`，经 `_save_result_to_mongo` 落 MongoDB `rag_eval_results`（字段此前已预留，本次补写）<br>5. **附带修复**：原"回溯问题检索"内部改写存在 format 缺 history 变量导致策略 100% 失败返回空的隐性 bug，现改为直接用已改写的 query 检索，不再二次改写 |
| 最终目标 | 1. ✅ 所有查询都先做 query 改写，不是只有回溯策略才改<br>2. ✅ 结合多轮对话历史，解决指代问题（"它的股价呢" → "宁德时代股价现在是多少"）<br>3. ✅ 改写过程记录耗时，存到评估字段里 |
| 遗留风险 | ① 改写依赖 LLM 质量，改错时仅影响检索召回（答案生成仍用原 query），由"回退原 query"兜底<br>② 所有专业咨询 query 多一次 LLM 调用（约百 ms 级），首轮/多跳问题同样产生该开销 |

---

### [4] 第二层缓存：Redis向量缓存
| 维度 | 内容 |
|------|------|
| 功能 | 语义相似的query，直接复用之前的上下文（**不缓存答案**），省检索+重排；命中后仍走LLM生成 |
| **当前完成度** | ✅ **100%**（2026-10-01 落地） |
| 现有实现 | 1. **存储模型**：`rag_qa/core/redis_vector_cache.py` 重构为 v2 —— 缓存键 = md5(改写后query + source_filter) 文本 id（确定性、幂等覆盖、防并发撞 key）；HASH 内与向量共存（query_vector 1024维 + query_text/rewritten_query/chunks_json/parent_ids_json/source_metas_json/source_filter Tag/hit_count/created_time/last_hit_time/is_formal），一次 FT KNN 搜索取回全量<br>2. **两级缓存**：写入为临时（temporary_ttl=86400，命中可滑动续期），每次真实命中 hit_count+1，≥ hot_threshold(3) 自动升级正式（formal_ttl=604800，不再滑动）；淘汰以 TTL 为主<br>3. **接线**：`new_rag_system.py::generate_answer` 在 query 改写后、策略选择前查缓存（用**改写后 query** 算向量 + source_filter TAG 隔离）；命中跳过 策略选择+混合检索+重排，复用缓存上下文填 prompt 发 LLM（不提前 return，保证流式输出与会话历史正常）；未命中走原链路，生成成功且上下文非空时回填 save_cache（不存答案）<br>4. **命中率统计**：Redis 运行时 hits/misses/upgraded（搜索异常不计 miss，缓存绝不阻断主流程）+ MySQL 日表 `redis_cache_stats`（`sync_daily_stats` 每日增量 upsert，记录当日生效 threshold/hot_threshold/avg_cache_time_ms，保证调参后历史可比）<br>5. **配置**：`config.ini [redis_vector_cache]` similarity_threshold=0.90（原0.95过严）、cache_enabled、sliding_ttl_enabled；`base/config.py` 增 CACHE_ENABLED/SLIDING_TTL_ENABLED（getboolean 带 fallback）<br>6. **校验**：`rag_qa/data/verify_redis_vector_cache.py`（39项）、`verify_redis_cache_flow.py`（29项）、`verify_redis_cache_integration.py`（31项）全 PASS（真实 redis-stack + 独立测试索引）；回归 `verify_query_rewrite.py` 34 项通过 |
| 最终目标 | 1. ✅ 用改写后的query算向量（命中率更高）<br>2. ✅ 临时/正式两级缓存，自动升级（hot_threshold=3）<br>3. ✅ 有TTL自动过期（临时1天/正式7天，临时滑动续期）<br>4. ✅ 缓存命中率统计（Redis 运行时 + MySQL 日表 redis_cache_stats） |
| 遗留风险 | ① threshold=0.90 为起步值，需靠日表命中率数据标定（过低会误命中串题，过高命中率低）<br>② hot_threshold=3 为固定超参，后续可按命中分布标定<br>③ 换 embedding 模型（维度变化）需 DROP 旧索引重建（模块已注释说明）<br>④ 一级 BM25 命中直返答案，二级缓存仅省检索不省生成，高峰收益主要在检索耗时（Milvus 混合检索+重排）<br>⑤ 缓存上下文可能过期（如年报发布后），TTL 兜底；正式缓存 7 天仍需关注时效 |

---

### [5] 检索策略选择
| 维度 | 内容 |
|------|------|
| 功能 | 根据query类型，选最合适的检索方式（直接检索 / HyDE / 子查询 / 回溯问题） |
| **当前完成度** | ✅ **100%**（2026-10-01 落地） |
| 现有实现 | 1. **策略判定前置（先判策略，再决定是否改写）**：`new_rag_system.py` 新增 `_is_direct_query()` 规则判定——否定信号（抽象/比较/指代/汇总跨度）+ 多实体 + 强事实词三条件；**直接检索类不调 LLM、不改写，用原 query 直接检索**；非直接类（抽象/比较/指代/多实体/汇总）才走 LLM 选策略 + 改写<br>2. **先 LLM 选策略再改写**：非直接类用**原 query** 选策略（保留指代/多跳/抽象等原始信号，避免改写后信号丢失），再统一改写（改写结果用于缓存与检索）<br>3. **策略名归一化**：`strategy_selector.py` 新增 `STRATEGY_ALIASES`（36 个别名）+ `normalize()`——LLM 输出变体（如"假设问题检索（HyDE）"/"hyDE"/"回溯"）归一为四个规范名，无法识别自动回退"直接检索"；prompt 收紧只输出规范名<br>4. **直接检索 vs 回溯差异化**：直接 = 原 query 检索；回溯 = 改写后 query 检索（不再等价，改写即回溯的实质）<br>5. **source_filter 透传**：HyDE / 子查询 / 回溯三个策略方法补齐 source_filter 参数并透传 Milvus（原仅直接检索传，修了过滤失效 bug）<br>6. **评估字段**：`retrieval_strategy` + 新增 `strategy_select_time`（策略选择耗时）写入 MongoDB `rag_eval_results`<br>7. **校验**：`rag_qa/data/verify_strategy_selector.py`（71 项）全 PASS；回归 `verify_query_rewrite.py` 34 项通过 |
| 最终目标 | 1. ✅ 直接检索类规则判定，不调 LLM、不改写（写清"什么样的 query 属于直接检索"）<br>2. ✅ 非直接类先 LLM 选策略再改写<br>3. ✅ 策略名归一化 + 策略选择耗时记录到评估字段<br>4. ⚠️ 数据统计自动优化选择逻辑（依赖 [10] 在线质量评估的忠实度分，未上线，留待后续） |
| 遗留风险 | ① 直接检索判定词表为起步值（宁可少判直接，不可误判直接），需靠 `rag_eval_results` 策略分布数据标定<br>② 策略效果统计依赖 [10] 在线质量评估（0%），当前仅能记录策略使用分布<br>③ 非直接类缓存命中时仍多一次选策略 LLM 调用（可接受，命中率由 threshold 控制） |

---

### [6] 混合检索（Milvus）
| 维度 | 内容 |
|------|------|
| 功能 | 稠密向量+稀疏向量混合检索，召回相关文档 |
| **当前完成度** | ⚙️ **90%**（2026-10-01 更新：核对代码实际后刷新，此前 80% 描述滞后于代码） |
| 现有实现 | 1. **混合检索**：BGEM3 稠密+稀疏双路（WeightedRanker 1.0/0.7），topk=RETRIEVAL_K(100)；Milvus 存子块向量 + 元数据，MySQL 存父块完整内容<br>2. **父块内容回取**：`vector_store.py::hybrid_search_with_rerank` 检索到子块后，按 parent_id 去重，自动从 MySQL `parent_docs.parent_content` 批量取**完整父块内容**（不是子块）<br>3. **Rerank**：本地 BGE-Rerank（CrossEncoder `models/bge-reranker-large`）对父块重排，取 CANDIDATE_M(5)（详见 [7]）<br>4. **表格增强双轨**：Milvus 表格子块 text = `[表格] ` 前缀 + 语义化描述（复杂表走 LLM 生成、简单表走 L1 规则转写，`table_enhancer.py`）；MySQL content 保持原始 Markdown/HTML 表格，数字保真<br>5. **表格摘要分流**：检索结果中 `[表格]` 前缀摘要按 parent_id 聚合挂到父块 metadata（table_summaries），概览型 query 用摘要、精确数字型 query 用父块原文<br>6. **✅（本次）Milvus 补存标量字段**：`add_documents` 写入 is_table/has_number（动态字段）；存量 21,074 条已由回填脚本补齐（20,976 条/24.3s）；chunk_type 与 is_table 同义（纯冗余）已按用户要求全量删除（15.7s），新写入不再带<br>7. **✅（本次）表格类 query 优先召回**：`new_rag_system.py::_is_table_query` 词表判定（明细/列表/清单/前十大股东等），检索时 expr 叠加 `is_table == 1`，空结果自动去掉该条件回退兜底<br>8. **✅（本次）检索耗时拆分**：Milvus 检索 / MySQL 父块 / rerank 三段计时（`vector_store.last_timings`），`rerank_time` 真实落库；前端字段名不变<br>9. **✅（本次）缺向量盘点与重灌闭环**：诊断出 file_id=439（融资公告，mineru 解析）1,265 父块 MySQL 全在但 Milvus 零向量——根因是 `add_documents` 先把 doc_status 标 `indexed` 再 upsert，当时 Milvus 容器不稳导致 upsert 抛异常但状态已固化（状态与数据脱节且无标记可查）。已修时序（`pending_doc_ids`：upsert 成功后才统一标 indexed，失败标 failed+error_msg 并 re-raise），并由 `rag_qa/data/refill_milvus_from_mysql.py` 从 MySQL 重建向量重灌 2,683 子块（708 表格块走增强，732.6s，LLM 超时/额度耗尽自动切模型+降级兜底），**Milvus 与 MySQL 差异清零（全库 23,757 = child_chunks 23,757）**<br>10. **✅（本次）表格过滤回退落库标记**：`vector_store.last_timings` 增 `table_filter_fallback`（空结果回退置 True）/`table_hit_count`（rerank 后携带表格摘要的父块数）；`new_rag_system` 把 `table_only/table_fallback/table_hit_count` 真实落库 rag_eval_results（缓存命中保持默认不误标），供标定 TABLE_QUERY_WORDS<br>11. **✅（本次）表格增强 A/B 评估**：`rag_qa/data/eval_table_enhance_ab.py`——60 表格 + 60 文本块建临时 collection（ab_enh/ab_raw，schema/索引同生产），30 条表格类 query 纯召回（无 rerank）对比：增强 vs 原文 @5/@10 命中率均 100%，MRR 0.814 vs 0.789（增强略优、首个表格块位次更靠前）→ 保留 table_enhance 合理；临时 collection 已清理<br>12. **✅（本次）Milvus 异常降级 MySQL 检索**：`hybrid_search_with_rerank` 检索段包 try/except，Milvus 异常（容器不稳/超时/网络断）自动降级 `_fallback_search_mysql`——jieba 分词滤停用词取前 5 词 → child_chunks.content LIKE 匹配（OR 命中任一词，按命中词数降序，max_rows=300）→ 复用父块回取 + rerank，返回结构完全一致；table_only/source_filter 语义与正常路径同（table_only 空结果同样去过滤重试）；降级标记 `fallback_to_mysql` 落 rag_eval_results（可观测 Milvus 故障频率）；降级自身失败返回空不崩<br>13. **✅（本次）rerank 稳定排序修复**：`sorted(zip(scores, parent_docs))` 在两条父块得分相同时会回退比较 Document 对象而崩溃（实测必现），改为只按 score 排序（同分保持 MySQL 返回顺序） |
| 最终目标 | 1. ✅ Milvus 补存标量字段（is_table、has_number）——新写入已带，存量已回填（20,976 条）；chunk_type 冗余已删<br>2. ✅ 表格类 query 优先召回 is_table=1 的子块（空结果自动回退）<br>3. ✅ 检索到子块后自动去 MySQL 拿完整父块内容<br>4. ✅ 检索耗时单独记录（含 rerank 分段） |
| 遗留风险 | ① ✅ 存量回填已完成（2026-10-01：20,976 条/24.3s，跳过 1,301 父块 = 1,265 缺向量 + 36 文本匹配不上）；缺向量已由重灌闭环（见现有实现 ⑨）<br>② 表格类 query 判定词表为起步值，已开始靠 `rag_eval_results` 的 `table_only/table_fallback/table_hit_count` 真实数据标定（本日落库标记已接）<br>③ ✅ 差异已清零（2026-10-01：file_id=439 重灌 2,683 子块后，Milvus 全库 23,757 = MySQL child_chunks 23,757） |

---

### [7] Rerank精排
| 维度 | 内容 |
|------|------|
| 功能 | 对召回的 top100 个文档重新打分排序，只留最相关的（rerank_k=3） |
| **当前完成度** | ✅ **100%**（2026-10-01 本次落地：开关/阈值/分数落库/rerank_k 接线，verify_rerank.py 26 项 + 回归 16 项全 PASS） |
| 现有实现 | 1. **本地 BGE-Rerank**：`vector_store.py::__init__` 加载 `models/bge-reranker-large`（CrossEncoder，cuda/cpu 自动）<br>2. **重排时机**：Milvus 召回 100 子块 → parent_id 去重 → MySQL 批量取父块 → `reranker.predict(pairs)` 全量打分 → 降序取 `RERANK_K`(3)<br>3. **精排耗时单独记录**：rerank 段计时写 `last_query_result["rerank_time"]` 落 MongoDB（前端不读，无影响）<br>4. **✅（本次）分数落库闭环**：每个返回父块 metadata 带 `rerank_score`；`new_rag_system` 提取填 `retrieved_scores`（原空字段，本次接线）+ MongoDB `retrieved_scores` 落库；`last_timings` 增 `rerank_score_min/max/mean/median`、`rerank_candidate_count`（评估/阈值标定输入）<br>5. **✅（本次）rerank 开关**：`rerank_enabled=false` 时跳过 CrossEncoder（rerank_time≈0、`rerank_disabled=true` 落库、无分数回填），供 A/B 与故障排查<br>6. **✅（本次）低置信度保护**：`rerank_threshold`（默认 0=关闭）；>0 且 top1 分数低于阈值时整批返回空 + `rerank_low_confidence` 标记（留给 [10]评估/[11]自愈触发入口），不在内部砍第 N 名<br>7. **✅（本次）rerank_k 接线**：config.ini 死配置 `rerank_k` 正式接线为"精排后保留数"，用户要求默认值 5→3（config.py fallback=3） |
| 最终目标 | 1. ✅ 本地 BGE-Rerank 对 top100 重排序取 top 3（RERANK_K=3，用户拍板默认值 5→3）<br>2. ✅ 精排耗时单独记录<br>3. ✅ 精排后 topk 改为 3（原 5） |
| 遗留风险 | ① ⚠️ **"父块数远小于100"已被实测推翻**（2026-10-01 探测：8 条 query 去重后父块均值 87.1、子块/父块比值 1.2）→ 父块级重排候选充分，无需子块级精排<br>② ⚠️ **单一绝对阈值不可行**（实测：第5名与第6名分数几乎无断档[最大差 0.021]，但不同 query 分数基准差异极大[隆基绿能 top1=0.58 vs 毛利率 top5 最低=0.966]）→ 阈值语义定为"整批低置信度"而非"内部砍名次"，默认关闭，靠 `retrieved_scores` 真实分布标定<br>③ rerank 是耗时大头（单次批量≈87 父块 2.6~4.0s），依赖 [4] Redis 缓存省检索 |

---

### [8] 上下文组装（父块优先）
| 维度 | 内容 |
|------|------|
| 功能 | 把检索到的文档，组装成LLM能读的上下文 |
| **当前完成度** | ✅ **100%**（2026-10-02 落地：分层截断 + 来源编号引用 + is_table 判定迁移） |
| 现有实现 | 1. **父块完整内容**：上下文取自 `_get_parent_contents` 返回的 `parent_content`（MySQL 父块完整内容），非零碎子块<br>2. **表格类分流**：`[表格]` 语义摘要（table_summaries）用于概览型 query；精确数字型（`_is_precise_query`：多少/几/分别/明细等，原 query OR 改写 query 任一命中）用父块原文（带 `\|` 完整结构），保证数字可查<br>3. **来源信息**：file_name、code、name、source 拼进 source_metas<br>4. **✅（本次）来源时间补全**：`_get_parent_contents` 返回 metadata 补 `timestamp=pdf_time`（公告/文档时间，非入库时间），`source_metas["time"]` 不再为空，前端"公告日期"与缓存路径一致<br>5. **✅（2026-10-02）表格判定迁移 is_table==1**：表格摘要分流判定从 Milvus text 的 `[表格]` 前缀改为动态字段 `is_table==1`（用户拍板）；`hybrid_search_with_rerank` 两处检索 output_fields 补 `"is_table"`，`_fallback_search_mysql` 返回 entity 补 `"is_table"`（原 SQL 已查但返回结构漏带），降级路径 table_only 过滤同语义<br>6. **✅（2026-10-02）来源信息进 prompt**：新增 `_format_context_blocks`——上下文按 `[N] 块内容` 编号引用组织，尾部附 `来源列表：\n[N] 标题（时间）`；source_metas 与块同序，为 [9] 逐句来源标注铺路<br>7. **✅（2026-10-02）分层截断**：新增 `_truncate_layered`——question 完整保护 > history 近轮优先（`reversed` 从最新保留）> context 按块追加（编号 5 + 分隔 2 恒定计费），预算不足整块丢弃保前部、单块超预算硬截、question 异常超长时其余全丢不崩；原整串硬截仅保留作 question 超长兜底；`truncated/context_blocks_before/after/history_kept` 经 `truncated_info` 落库<br>8. **✅（2026-10-02）组装逻辑独立可测**：内联组装抽为 `_build_context_blocks`（表格分流 + source_metas 同序）/`_format_context_blocks`/`_truncate_layered` 三方法；缓存命中/未命中/通用知识三分支统一走分层截断 + 编号组装；缓存命中按序对齐加来源头（chunks[i]↔source_metas[i]，save_cache 存截断后块，老缓存缺字段防御）；`context_empty`（无检索块置 True）落库 |
| 最终目标 | 1. ✅ 优先用父块完整内容，不用零碎的子块<br>2. ✅ 表格类子块自动从 MySQL 拿完整表格（带\|的完整结构）<br>3. ✅ 上下文太长自动截断（分层保护：question > history 近轮 > context 按块）<br>4. ✅ 来源信息进 prompt（编号引用 [N] + 尾部来源列表）<br>5. ✅ 表格块判定用结构化字段 is_table==1 |
| 遗留风险 | ① ✅ 硬截断已解决（2026-10-02 分层截断落地）；剩余：预算由单一 `max_prompt_length`(16384) 控制，后续可按场景（表格/财报类）细分预算<br>② ✅ is_table 判定已迁移（2026-10-02 起表格摘要分流与过滤均用结构化字段）；剩余仅 TABLE_QUERY_WORDS 词表标定（依赖 rag_eval_results 的 table_only/table_fallback/table_hit_count 分布数据） |

---

### [9] 答案生成（LLM流式）
| 维度 | 内容 |
|------|------|
| 功能 | 把上下文+问题拼成prompt，调用LLM流式生成答案 |
| **当前完成度** | ✅ **100%**（2026-10-02 落地：来源标注 + 拒答检测，90%→100%） |
| 现有实现 | 1. **流式输出**：兼容 langchain stream()（AIMessageChunk）与生成器函数（yield str/元组）两类 LLM 接口；生成耗时 `generation_time`、总耗时 `total_time` 单独记录<br>2. **上下文超长自动截断**：分层截断（question > history 近轮 > context 按块）+ 兜底硬截（见 [8]），保证 prompt 不超 `max_prompt_length`<br>3. **✅（本次）来源标注**：`prompts.py::rag_prompt` 增强——要求模型在引用上下文内容（数字/结论）的**句子末尾标注来源编号**（如"2025年营收100亿元[1]"），尾部信息来源改为按实际引用编号 `[1][2]` 列出；生成完成后 `new_rag_system.py::_extract_citations` 解析答案中的 `[N]` → 结构化 `citations` 列表（index/title/time/source/code/name，与 source_metas 同序映射，越界编号忽略、重复去重、无编号/空输入返回空不崩），落 `last_query_result["citations"]` + MongoDB `rag_eval_results.citations`，供前端后续渲染可点击溯源（数据层铺垫，前端接线为下一步）<br>4. **✅（本次）拒答检测**：`new_rag_system.py::_detect_refusal` 双信号——① 专业咨询上下文为空（`context_empty=True`）直接判拒答场景（无可依据资料）；② 答案命中 `REFUSAL_PHRASES` 保守词表（无法回答/未找到/检索不到/资料未提及等 30+ 短语，起步值待标定）；通用知识分支不判拒答（不依赖检索，空上下文是常态）；结果 `refusal_detected/refusal_reason` 落 `last_query_result` + MongoDB `rag_eval_results`，日志打 warning，作为 [11] 自愈 Agent 的触发入口（[11] 未做，本次只做检测与标记）<br>5. **✅（本次）字段闭环**：`last_query_result` 默认字典与每轮复位块同步新增 citations/refusal_detected/refusal_reason 三字段（LLM 失败/异常保持默认，不污染下轮）；校验 `rag_qa/data/verify_answer_generation.py` 31 项全 PASS，回归 `verify_context_assembly.py` 32 项全 PASS |
| 最终目标 | 1. ✅ 来源标注：答案里每句话对应哪个公告，后面可以点（[N] 内联编号 + 结构化 citations 落库，前端接线为下一步）<br>2. ✅ 拒答检测：模型说"找不到"的时候，触发自愈（检测+标记+落库已完成，实际自愈循环属 [11]） |
| 遗留风险 | ① ⚠️ **来源标注可靠性依赖模型遵循编号指令**：模型可能漏标/错标编号；越界编号已被解析层忽略，漏标则 citations 为空——需靠 rag_eval_results.citations 覆盖率数据评估，必要时在 prompt 加"必须引用编号"强约束<br>② ⚠️ **拒答词表为起步值**：REFUSAL_PHRASES 保守收录，靠 rag_eval_results.refusal_detected 分布标定增删（误判会多耗一次自愈 LLM 调用，漏判则不触发自愈）<br>③ **前端"可点击溯源"接线未做**：citations 已落库（MongoDB + last_query_result），前端渲染/点击跳转属前端模块工作（static/ 未跟踪）<br>④ ✅ 通用知识分支不判拒答已处理（is_general 跳过 context_empty 信号），避免通用问答全部误判 |

---

### [10] 在线质量自动评估
| 维度 | 内容 |
|------|------|
| 功能 | 每次回答完，自动算这次回答的质量分 |
| **当前完成度** | ❌ **0%** |
| 现有实现 | ❌ 只有离线RAGAS评估代码，在线的还没写 |
| 最终目标 | 1. 每次回答完，自动算忠实度分数（答案里的话能不能在上下文里找到依据）<br>2. 分数低于阈值，自动触发自愈<br>3. 拒答检测：模型说"不知道"的时候，触发自愈<br>4. 评估耗时也单独记 |

---

### [11] React自愈Agent（运行时闭环）
| 维度 | 内容 |
|------|------|
| 功能 | 质量不达标了，Agent自己反思哪里错了，自动优化检索链路 |
| **当前完成度** | ❌ **0%** |
| 现有实现 | ❌ 还没开始做 |
| 最终目标 | 1. Agent自动反思：是query没改写好？还是召回的文档不对？还是策略选错了？<br>2. 自动执行优化：改写query、换检索策略、扩大topk、重新检索生成<br>3. 自愈过程记录日志，后面分析哪些问题经常出问题<br>4. 形成闭环：用户问 → 答 → 评估 → 不行就自愈 → 再答 |

---

### [12] 三层记忆模块
| 维度 | 内容                                                                             |
|------|--------------------------------------------------------------------------------|
| 功能 | 记住用户的对话历史和长期偏好                                                                 |
| **当前完成度** | ❌ **0%**                                                                       |
| 现有实现 | ❌ 现在只是临时传history参数，没有持久化                                                       |
| 最终目标 | 1. 窗口记忆：最近2轮对话，用完就扔<br>2. 工作记忆：当前会话的短期记忆，存Redis<br>3. 长期记忆：用户长期偏好、常问的公司，存MySQL |

---

## 三、整体完成度统计

| 模块 | 完成度 |
|------|--------|
| 查询分类 | 100% |
| Query改写 | 100% |
| 两级缓存 | 100%（一级 BM25 100% + 二级 Redis 100%） |
| 检索策略选择 | 100% |
| 混合检索 | 90% |
| Rerank精排 | 100% |
| 上下文组装 | 100% |
| 答案生成 | 100% |
| 在线质量评估 | 0% |
| React自愈Agent | 0% |
| 三层记忆 | 0% |
| **整体完成度** | **约73%**（2026-10-02 答案生成 90%→100% 后刷新；11 项合计 800/1100） |

---

## 四、后续优先级排序

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | ✅ 基础RAG + 离线RAGAS评估 | 已完成（混合检索/父块/Rerank 已接线） |
| P1 | ✅ 上下文拿父块 + 表格类从 MySQL 拿 | 已完成（2026-10-01 核对确认） |
| P1.5 | ✅ 存量 Milvus 回填标量字段（21,074 条） | 已完成（2026-10-01）：`rag_qa/data/backfill_milvus_scalar_fields.py` 全量执行 24.3s，回填 20,976/21,074 条（99.5%），is_table=1 共 2,947 条可被表格过滤命中；跳过 1,301 父块（1,265 缺向量 + 36 文本匹配不上，保守跳过）。同日按用户要求删除纯冗余 chunk_type 动态字段（`delete_milvus_chunk_type_field.py` 15.7s/20,772 条），新写入不再带 |
| P1.6 | ✅ 缺向量重灌 + 入库时序修复 + 回退落库标记 + 表格增强 A/B | 已完成（2026-10-01）：file_id=439 重灌 2,683 子块（`refill_milvus_from_mysql.py`，732.6s），Milvus=MySQL 差异清零（23,757）；`add_documents` 时序修复（upsert 成功才标 indexed）；`table_only/table_fallback/table_hit_count` 落库；A/B 结论增强不劣于原文（MRR 0.814 vs 0.789） |
| P1.7 | ✅ Milvus 异常降级 MySQL 检索 | 已完成（2026-10-01）：Milvus 挂时 jieba 关键词降级检索 + 父块回取 + rerank 复用，`fallback_to_mysql` 落库；顺带修复 rerank 同分崩溃 |
| P1.8 | ✅ 上下文组装补完 | 已完成（2026-10-02）：分层截断（question > history 近轮 > context 按块）+ 来源编号引用 [N] + 尾部来源列表 + 表格判定迁移 is_table==1 + 组装抽独立函数 + context_empty/truncated_info 落库；校验 verify_context_assembly.py 32 项 + verify_query_rewrite.py 34 项 + verify_fallback_mysql.py 15 项 + verify_milvus_hybrid.py 16 项全 PASS |
| P2 | ✅ Rerank精排 | 已完成（2026-10-01 补完：开关/阈值/分数落库/rerank_k=3，verify_rerank.py 26 项 + verify_milvus_hybrid.py 16 项全 PASS） |
| P3 | QA库扩容 + 弱QA治理（人工校对/回扫） | 一级缓存数据源质量：软校验关闭期间入库的弱QA需治理；QA库扩容后重标定 BM25 阈值 12 |
| P4 | 在线质量评估 + 自愈Agent | 运行时自动优化 |
| P5 | 三层记忆模块 | 多轮对话体验 |

> 已完成：✅ 第一层缓存 MySQL BM25（2026-10-01）——原 P3 任务"加MySQL BM25缓存层"已落地。
> 已完成：✅ Query改写（2026-10-01）——统一改写入口 + 对话历史消解指代 + 耗时写入评估字段，并修复回溯策略隐性 bug。
> 已完成：✅ 第二层缓存 Redis向量缓存（2026-10-01）——改写后query算向量 + 临时/正式两级自动升级 + TTL淘汰 + 命中率统计（Redis运行时 + MySQL日表），三套校验 99 项全 PASS，回归 34 项通过。
> 已完成：✅ 混合检索补齐（2026-10-01）——Milvus 补存 is_table/has_number 标量字段（新写入 + 存量回填 20,976 条/24.3s；chunk_type 纯冗余已删除）、表格类 query 优先召回（空结果回退）、检索耗时拆分（Milvus/MySQL/rerank 三段，rerank_time 落库）、来源时间补全（pdf_time）。
> 已完成：✅ 混合检索补完（2026-10-01 晚）——file_id=439 缺向量重灌闭环（2,683 子块，Milvus=MySQL 差异清零 23,757）、入库时序修复（upsert 成功才标 indexed）、表格回退落库标记（table_only/table_fallback/table_hit_count）、表格增强 A/B（增强不劣于原文，MRR 0.814 vs 0.789）。
> 已完成：✅ 答案生成收尾（2026-10-02）——来源标注（prompt 内联编号引用 [N] + `_extract_citations` 结构化 citations 落库，前端溯源数据层就绪）+ 拒答检测（`_detect_refusal`：context_empty/拒答短语双信号，refusal_detected/refusal_reason 落库，[11] 自愈触发入口）；校验 verify_answer_generation.py 31 项 + 回归 verify_context_assembly.py 32 项全 PASS。
