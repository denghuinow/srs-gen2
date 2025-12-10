你是一个高级 Python 工程师，需要实现一套“多版本 SRS 生成器”。
本项目的核心逻辑已经用伪代码给出，你需要用 Python + OpenAI 大语言模型 和 OpenAI 向量嵌入，把它实现为可运行代码。

### 一、整体目标

实现一个主函数，例如：

```python
def run_srs_iteration(
    d_orig: str,
    r_base: str,
    d_base: str,
    output_dir: str,
    rho: float,
    max_outer_iter: int,
    max_inner_iter: int,
) -> None:
    ...
```

功能：

* 按照给定的算法逻辑，反复进行“需求探索 → 澄清 → 更新需求池 → 生成 SRS”。
* 每一轮外循环生成一版 SRS 文档，并写入磁盘：
  `output_dir/srs_iter_1.md`、`output_dir/srs_iter_2.md`、…
* 函数**不需要返回值**，输出完全体现在磁盘文件上。

### 二、技术要求

1. 使用 **Python 3**。
2. 使用 **OpenAI 官方 Python SDK** 调用：

   * 大语言模型（用于：split_to_semantic_units、requirement_explorer、requirement_clarifier、srs_generator）
   * 向量嵌入模型（用于：filter_low_similarity）
3. 所有对 LLM 的调用必须通过**模板化提示词**完成：

   * 每个功能一个模板文件：

     * `prompts/split_to_semantic_units.md`
     * `prompts/requirement_explorer.md`
     * `prompts/requirement_clarifier.md`
     * `prompts/srs_generator.md`
   * 模板文件是普通文本，内部用占位符（例如 `{{D_BASE}}`、`{{UNITS_JSON}}`）进行填充。
   * 运行时从磁盘加载模板，进行字符串替换后，作为 prompt 发送给 LLM。
4. filter_low_similarity 使用 OpenAI 向量嵌入 + 余弦相似度，过滤掉与已有需求过于相似的语义单元。

### 三、项目结构建议

参考如下结构（可以在此基础上扩展）：

```text
project_root/
  main.py
  srs_pipeline.py
  models.py
  openai_utils.py
  prompts/
    split_to_semantic_units.md
    requirement_explorer.md
    requirement_clarifier.md
    srs_generator.md
```

### 四、数据结构

定义一个语义单元类：

```python
# models.py
from typing import Optional

class SemanticUnit:
    def __init__(self, text: str, grade: Optional[float] = None, meta: Optional[dict] = None):
        self.text = text
        self.grade = grade
        self.meta = meta or {}

    def to_dict(self) -> dict:
        return {"text": self.text, "grade": self.grade, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "SemanticUnit":
        return SemanticUnit(
            text=d["text"],
            grade=d.get("grade"),
            meta=d.get("meta") or {},
        )
```

### 五、OpenAI 封装

实现一个工具模块负责：

* 加载模板
* 调用 chat LLM
* 调用 embedding 模型

例如：

```python
# openai_utils.py
import os
from typing import List
from openai import OpenAI

client = OpenAI()  # 使用环境变量 OPENAI_API_KEY

def load_prompt_template(name: str) -> str:
    # name 例如 "split_to_semantic_units"
    base_dir = os.path.join(os.path.dirname(__file__), "prompts")
    path = os.path.join(base_dir, f"{name}.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def call_llm(prompt: str) -> str:
    # 简化：单轮 user prompt，模型、温度可做成参数或常量
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # 或者你认为合适的模型名
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",  # 或其他合适模型
        input=text,
    )
    return resp.data[0].embedding
```

### 六、四个基于 LLM 的函数实现要求

#### 1. split_to_semantic_units

签名：

```python
from typing import List
from models import SemanticUnit
from openai_utils import load_prompt_template, call_llm

def split_to_semantic_units(d_base: str) -> List[SemanticUnit]:
    ...
```

实现要求：

* 读取模板：`split_to_semantic_units.md`。

* 模板中应说明：

  * 输入是一份 SRS 文本（d_base）。
  * 需要将其拆分为一组“语义单元”，每个单元代表一个尽可能原子化的需求。
  * 要求 LLM 以 **JSON 格式** 返回，例如：

    ```json
    {
      "units": [
        {"text": "..."},
        {"text": "..."}
      ]
    }
    ```

* 代码中：

  * 使用 `template.format(D_BASE=d_base)` 或其他方式注入文本。
  * 调用 LLM，解析 JSON，构造 `SemanticUnit` 列表（grade 初始 None）。

模板文件大致形态（由你编码实现时具体写清楚）：

```text
# prompts/split_to_semantic_units.md（示意）

你是一个帮助拆分软件需求文档的助手。
给定一份 SRS 文本，请将其拆分为一系列“语义单元”，每个单元代表一个独立的需求或约束。

要求：
- 不要遗漏原文中的需求信息。
- 一个语义单元的 text 字段应该是一段完整、可理解的话或条目。
- 输出必须是合法 JSON，结构如下：
{
  "units": [
    {"text": "..."},
    {"text": "..."}
  ]
}

下面是 SRS 文本：

{{D_BASE}}
```

> 注意：这里只是示意，实际内容由你在工程中写入模板文件。

#### 2. requirement_explorer

签名：

```python
def requirement_explorer(
    r_base: str,
    negative_pool: list[SemanticUnit],
    required_num: int,
) -> list[SemanticUnit]:
    ...
```

实现要求：

* 读取模板：`requirement_explorer.md`。

* 需要向 LLM 传入：

  * requirement_base 文本 `r_base`
  * 已有需求池（负样本）文本列表，建议只传 text 和简要上下文，避免 prompt 过长
  * 本轮希望“新挖掘”的 target 数量 `required_num`

* 让 LLM 基于 r_base，在尽量避免与 negative_pool 重复的前提下，生成最多 `required_num` 条**候选新需求**。

* 同样要求 LLM 以 JSON 返回，例如：

  ```json
  {
    "units": [
      {"text": "..."},
      {"text": "..."}
    ]
  }
  ```

* 代码中将 JSON 转为 SemanticUnit 列表（grade 可以先设为 None）。

模板示意（实际写入文件时你来细化）：

```text
# prompts/requirement_explorer.md（示意）

你是一个需求探索助手。
给你：
1. requirement base 文本；
2. 已有需求列表（不希望重复的内容）；
3. 本轮希望新增的需求条数 K。

请在 requirement base 的语义范围内，提出不超过 K 条“新的潜在需求”，尽量避免与已有需求重复或高度相似。

输出格式（必须是合法 JSON）：
{
  "units": [
    {"text": "..."},
    {"text": "..."}
  ]
}

K = {{REQUIRED_NUM}}

[Requirement base]
{{R_BASE}}

[Existing requirements]
{{NEGATIVE_UNITS_TEXT}}
```

#### 3. requirement_clarifier

签名：

```python
def requirement_clarifier(
    units: list[SemanticUnit],
    d_orig: str,
) -> list[SemanticUnit]:
    ...
```

实现要求：

* 读取模板：`requirement_clarifier.md`。

* 需要向 LLM 提供：

  * 原始 SRS 文档 `d_orig`
  * 待评分的语义单元列表（建议用 JSON 形式提供给模型），**每个单元需要包含唯一标识符 `id`，使用自增序号（从 1 开始）**
  
  * 代码实现中，在调用 LLM 前为每个 `SemanticUnit` 分配自增序号作为 `id`（例如：1, 2, 3, ...），可以通过 `meta` 字段存储，或在构造 JSON 时动态生成。
  
  输入 JSON 格式示例：
  ```json
  {
    "units": [
      {"id": 1, "text": "..."},
      {"id": 2, "text": "..."}
    ]
  }
  ```

* 要求 LLM 输出：同样数量的评分结果，每个包含 `id` 和 `grade` 字段，**不需要输出原需求文本**。

* 输出 JSON 示例：

  ```json
  {
    "units": [
      {"id": 1, "grade": 2},
      {"id": 2, "grade": -1}
    ]
  }
  ```

* 代码中通过 `id`（数字ID）匹配评分结果，更新对应 `SemanticUnit.grade`，**不修改 `text`**。

模板示意：

```text
# prompts/requirement_clarifier.md（示意）

你是一个需求吻合度评估助手。
给定：
1. 原始 SRS 文档；
2. 一组候选需求语句。

任务：
- 为每条需求语句进行需求吻合度评分。
- 评分采用5分制：
  * +2：强采纳（该需求与原始 SRS 高度吻合，强烈建议采纳）
  * +1：采纳（该需求与原始 SRS 吻合，建议采纳）
  * 0：中性意见（该需求与原始 SRS 的吻合度不明确）
  * -1：不采纳（该需求与原始 SRS 不太吻合，不建议采纳）
  * -2：强不采纳（该需求与原始 SRS 明显不吻合，强烈不建议采纳）

输出必须是 JSON，**需要包含 id 和 grade 字段，不需要输出原需求文本**。通过 id（数字ID）匹配对应的需求：
{
  "units": [
    {"id": 1, "grade": 2},
    {"id": 2, "grade": -1}
  ]
}

[Original SRS]
{{D_ORIG}}

[Candidate units]
{{UNITS_JSON}}
```

#### 4. srs_generator

签名：

```python
def srs_generator(grade_units: list[SemanticUnit]) -> str:
    ...
```

实现要求：

* 读取模板：`srs_generator.md`。
* 输入是已经过滤过的 `grade_units`（保证 `grade > 0`，即只包含采纳和强采纳的需求）。
* 让 LLM 基于这些语义单元，生成一份**结构化的 SRS 文本**（例如 Markdown 格式：章节、条目编号等）。
* 输出是纯文本字符串（SRS 文本），由调用方直接写入 `.md` 文件，不再 JSON。

模板示意：

```text
# prompts/srs_generator.md（示意）

你是一个资深系统分析师。
给你一组已经澄清并打分的需求语句，请根据它们生成一份结构化的软件需求规格说明书（SRS），使用 Markdown 格式。

要求：
- 合理分章节（例如：简介、总体描述、功能需求、非功能需求等）。
- 将输入的每条需求归类并编号。
- 不要遗漏任何一条输入需求。

下面是输入的需求列表（JSON 格式，每条包含 text 和 grade）：

{{UNITS_JSON}}

请输出最终的 SRS 文本（Markdown），不要再包裹在 JSON 里。
```

### 七、filter_low_similarity：使用 OpenAI 向量 + 余弦相似度

签名：

```python
from typing import List
from models import SemanticUnit
from openai_utils import get_embedding
import math

def filter_low_similarity(
    explored_units: List[SemanticUnit],
    existing_pool: List[SemanticUnit],
    threshold: float = 0.8,
) -> List[SemanticUnit]:
    ...
```

实现要求：

1. 对 `existing_pool` 中的每个 `SemanticUnit.text` 计算 embedding（可以做缓存，避免重复算）。
2. 对 `explored_units` 中每个候选：

   * 计算该候选文本的 embedding。
   * 依次与 existing_pool 的 embedding 计算余弦相似度：
     [
     \cos(\theta) = \frac{u \cdot v}{|u||v|}
     ]
   * 如果与任何一个 existing 向量的相似度 ≥ threshold，则认为“太像了”，丢弃。
   * 否则保留。
3. 返回保留下来的 `SemanticUnit` 列表。

可实现一个小工具函数：

```python
def cosine_similarity(a: List[float], b: List[float]) -> float:
    ...
```

### 八、主流程实现（严格按照这段伪代码）

请将下面伪代码翻译为 Python 实现（注意用上面定义的函数）：

```text
input:
    D_orig      # 原始 SRS 文档
    R_base      # requirement_base
    D_base      # 基线方法生成的 SRS 文档 (baseline SRS)
    OUTPUT_DIR  # 输出目录路径

# -------------------------
# 0. 工具函数：从 pool 生成一版 SRS
# -------------------------
function generate_srs_from_pool(pool):
    grade_units  = { u in pool | grade(u) > 0 }
    return srs_generator(grade_units)

# 文件写入函数（抽象）
function write_srs_to_disk(curr_srs, iter_idx, OUTPUT_DIR):
    filename = OUTPUT_DIR + "/srs_iter_" + to_string(iter_idx) + ".md"
    write_file(filename, curr_srs)

# -------------------------
# 1. 基线 SRS -> 语义单元 -> 初始化需求池
# -------------------------
baseline_units = split_to_semantic_units(D_base)

pool = ∅
clarified_baseline_units = requirement_clarifier(baseline_units, D_orig)
pool = pool ∪ clarified_baseline_units

rho = ...
N   = ceil(rho * size(baseline_units))

MAX_OUTER_ITER = ...
MAX_INNER_ITER = ...

outer_iter = 1

# -------------------------
# 2. 外层循环：每一轮 = 探索 + 澄清 + 更新池 + 生成一版 SRS
# -------------------------
while outer_iter <= MAX_OUTER_ITER:

    buffer_new_units = ∅
    inner_iter = 0

    # 内层循环：多次 explorer，尝试凑够 N 条 brand-new requirements
    while (size(buffer_new_units) < N) and (inner_iter < MAX_INNER_ITER):

        required_num = N - size(buffer_new_units)

        explored_units = requirement_explorer(
            R_base,
            pool ∪ buffer_new_units,
            required_num
        )

        brand_new_units = filter_low_similarity(
            explored_units,
            pool ∪ buffer_new_units
        )

        buffer_new_units = buffer_new_units ∪ brand_new_units
        inner_iter = inner_iter + 1

    # ---- 内层结束，检查本轮有没有有效增量 ----

    if size(buffer_new_units) == 0:
        break

    if size(buffer_new_units) < N:
        break

    # 走到这里说明 >= N 条 brand-new requirements，触发一次 clarifier
    clarified_new_units = requirement_clarifier(buffer_new_units, D_orig)
    pool = pool ∪ clarified_new_units

    # 每次外循环结束后：基于当前 pool 生成一个 SRS 版本
    curr_srs = generate_srs_from_pool(pool)

    # ---- 写入磁盘文件，文件名用 outer_iter 做编号 ----
    write_srs_to_disk(curr_srs, outer_iter, OUTPUT_DIR)

    outer_iter = outer_iter + 1

# -------------------------
# 3. 过程结束
# -------------------------
# 无显式返回值，结果是磁盘上的一串：
# OUTPUT_DIR/srs_iter_1.md
# OUTPUT_DIR/srs_iter_2.md
# ...
# OUTPUT_DIR/srs_iter_K.md
```

### 九、其他要求

* 为关键步骤添加日志输出（例如使用 logging 模块），方便查看每轮生成了多少新需求、生成了多少版本。
* 所有 TODO / 简化实现请用注释标明，方便后续替换为更精细的逻辑。
* 请最终输出完整的 Python 源码文件内容（不需要真的去调用 OpenAI API 运行，只需代码在合理配置下可以直接运行）。

---

**以上是完整需求，请根据该说明实现代码。**
