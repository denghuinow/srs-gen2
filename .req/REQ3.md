你是一个高级 Python 工程师，需要实现一套“多版本 SRS 生成器”。
本项目的核心逻辑已经用伪代码给出，你需要用 **Python + OpenAI 大语言模型 + OpenAI 向量嵌入** 把它实现为可运行代码。

---

## 一、整体目标

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

* 按照给定的算法逻辑，反复进行“**需求探索 → 澄清评分 → 更新需求池 → 生成 SRS**”。
* 每一轮外循环生成一版 SRS 文档，并写入磁盘：
  `output_dir/srs_iter_1.md`、`output_dir/srs_iter_2.md`、…
* 函数**不需要返回值**，输出完全体现在磁盘文件上。

---

## 二、技术要求

1. 使用 **Python 3**。
2. 使用 **OpenAI 官方 Python SDK** 调用：

   * Chat LLM（用于：`split_to_semantic_units`、`requirement_explorer`、`requirement_clarifier`、`srs_generator`）
   * 向量嵌入模型（用于：`filter_low_similarity`）
3. 所有对 LLM 的调用必须通过**模板化提示词**完成：

   * 每个功能一个模板文件（后缀为 `.md`）：

     * `prompts/split_to_semantic_units.md`
     * `prompts/requirement_explorer.md`
     * `prompts/requirement_clarifier.md`
     * `prompts/srs_generator.md`
   * 模板文件是普通文本，内部用占位符（例如 `{{D_BASE}}`、`{{UNITS_JSON}}`）进行填充。
   * 运行时从磁盘加载模板，进行字符串替换后，作为 prompt 发送给 LLM。
4. `filter_low_similarity` 使用 OpenAI 向量嵌入 + 余弦相似度，过滤掉与已有需求过于相似的语义单元。

---

## 三、项目结构建议

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

---

## 四、数据结构

**重要变更：去掉 `meta`，增加向量字段 `vector`。**

```python
# models.py
from typing import Optional, List

class SemanticUnit:
    def __init__(
        self,
        text: str,
        grade: Optional[float] = None,
        vector: Optional[List[float]] = None,
    ):
        self.text = text
        self.grade = grade
        self.vector = vector  # 用于缓存 OpenAI embedding

    def to_dict(self) -> dict:
        return {"text": self.text, "grade": self.grade, "vector": self.vector}

    @staticmethod
    def from_dict(d: dict) -> "SemanticUnit":
        return SemanticUnit(
            text=d["text"],
            grade=d.get("grade"),
            vector=d.get("vector"),
        )
```

说明：

* `vector` 字段用于缓存文本的向量表示，避免重复调用 embedding API。
* 其它地方如果需要将 `SemanticUnit` 序列化为 JSON，可以通过 `to_dict()`。

---

## 五、OpenAI 封装

实现一个工具模块负责：

* 加载模板
* 调用 chat LLM
* 调用 embedding 模型

示例：

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

---

## 六、四个基于 LLM 的函数实现要求

### 1. `split_to_semantic_units`

签名：

```python
from typing import List
from models import SemanticUnit
from openai_utils import load_prompt_template, call_llm

def split_to_semantic_units(d_base: str) -> List[SemanticUnit]:
    ...
```

要求：

* 模板文件：`prompts/split_to_semantic_units.md`
* 模板中说明：

  * 输入是一份 SRS 文本（`d_base`）。
  * 需要拆分为一组“语义单元”，每个单元代表一个独立需求。
  * 输出 JSON 格式，例如：

    ```json
    {
      "units": [
        {"text": "..."},
        {"text": "..."}
      ]
    }
    ```
* 代码中：

  * 用 `template.replace("{{D_BASE}}", d_base)` 注入文本。
  * 调用 LLM，解析 JSON，构造 `SemanticUnit` 列表（`grade=None`, `vector=None`）。

模板示意略（与前一版一致，这里不重复写）。

---

### 2. `requirement_explorer`

签名：

```python
from typing import List
from models import SemanticUnit

def requirement_explorer(
    r_base: str,
    negative_pool: List[SemanticUnit],
    required_num: int,
) -> List[SemanticUnit]:
    ...
```

要求：

* 模板文件：`prompts/requirement_explorer.md`
* 提供给 LLM：

  * requirement_base 文本 `r_base`
  * 已有需求列表（负样本），只需传 text
  * 目标新增条数 `required_num`
* LLM 在 `r_base` 的语义范围内生成最多 `required_num` 条新需求，尽量避免与已有内容重复。
* 返回 JSON：

  ```json
  {
    "units": [
      {"text": "..."},
      {"text": "..."}
    ]
  }
  ```
* 解析后转为 `SemanticUnit` 列表（`grade=None`, `vector=None`）。

---

### 3. `requirement_clarifier`

签名：

```python
from typing import List
from models import SemanticUnit

def requirement_clarifier(
    units: List[SemanticUnit],
    d_orig: str,
) -> List[SemanticUnit]:
    ...
```

要求：

* 模板文件：`prompts/requirement_clarifier.md`

* 提供给 LLM：

  * 原始 SRS 文档 `d_orig`
  * 待评分语义单元列表，带自增 `id`（从 1 开始），格式大致为：

    ```json
    {
      "units": [
        {"id": 1, "text": "..."},
        {"id": 2, "text": "..."}
      ]
    }
    ```

* 评分规则（写在模板里）：

  5 分制与原始 SRS 的吻合度：

  * +2：强采纳（高度吻合，强烈建议采纳）
  * +1：采纳（吻合，建议采纳）
  * 0：中性（吻合度不明确）
  * -1：不采纳（不太吻合）
  * -2：强不采纳（明显不吻合）

* 要求 LLM 输出仅包含 `id` 与 `grade` 的 JSON，例如：

  ```json
  {
    "units": [
      {"id": 1, "grade": 2},
      {"id": 2, "grade": -1}
    ]
  }
  ```

* Python 端通过 `id` 匹配，更新 `SemanticUnit.grade`，**不修改 `text`，不修改 `vector`**。

---

### 4. `srs_generator`

签名：

```python
from typing import List
from models import SemanticUnit

def srs_generator(grade_units: List[SemanticUnit]) -> str:
    ...
```

**重要变更：模型在生成文档时需要对评分为 1 的语义单元进行细化扩展。**

要求：

* 模板文件：`prompts/srs_generator.md`
* 输入是已过滤的 `grade_units`，满足 `grade > 0`（即只包含 +1 和 +2 的语义单元）。
* 模板中要明确要求：

  * `grade = 2` 的语义单元：视为高质量、完整需求，主要负责原样整合、合理编排和分类。
  * `grade = 1` 的语义单元：视为“值得采纳但相对粗略”的需求，**在生成 SRS 时需要对这些需求进行细化扩展**，例如：

    * 拆分为更具体的子需求条目；
    * 明确输入输出、边界条件、异常处理等；
    * 强化可测试性（可选）。
* 生成结果是 Markdown 格式 SRS 文本，不再用 JSON 包裹。

模板示意（要在文件里写清楚“对 grade=1 进行扩展”）：

```text
# prompts/srs_generator.md（示意）

你是一个资深系统分析师。
给你一组已经打分且被采纳的需求语句，请根据它们生成一份结构化的软件需求规格说明书（SRS），使用 Markdown 格式。

输入中每条需求包含 text 和 grade：
- grade = 2：高质量、完整的需求描述。
- grade = 1：需要采纳，但描述相对粗略，需要你在生成 SRS 时对其进行细化和扩展。

要求：
- 对 grade = 2 的需求，保持其核心含义，合理编排到合适的章节和条目中。
- 对 grade = 1 的需求，在保持原意的前提下，进行细化扩展，例如：
  - 拆解为更具体的子需求条目；
  - 明确输入、输出、前置条件、后置条件或异常情况；
  - 提高可测试性和可操作性。
- 合理分章节（例如：简介、总体描述、功能需求、非功能需求等）。
- 将所有输入需求（包括扩展后的）归类并编号。
- 不要遗漏任何一条输入需求。

下面是输入的需求列表（JSON 格式，每条包含 text 和 grade）：

{{UNITS_JSON}}

请输出最终的 SRS 文本（Markdown），不要再包裹在 JSON 里。
```

---

## 七、`filter_low_similarity`：OpenAI 向量 + 余弦相似度

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

1. 为 `existing_pool` 中的每个 `SemanticUnit` 准备向量：

   * 如果 `unit.vector` 为 `None`，调用 `get_embedding(unit.text)` 计算，并写回 `unit.vector`。
   * 否则复用已有向量。
2. 对 `explored_units` 中每个候选：

   * 若 `unit.vector` 为 `None`，同样调用 `get_embedding(unit.text)` 填充。
   * 依次与 `existing_pool` 的向量计算余弦相似度：

     ```python
     cos_sim = dot(a, b) / (norm(a) * norm(b))
     ```
   * 若与任何一个 existing 向量的相似度 ≥ `threshold`，则认为“太像”，丢弃。
   * 否则保留。
3. 返回保留下来的 `SemanticUnit` 列表。

可以实现一个小工具函数：

```python
def cosine_similarity(a: List[float], b: List[float]) -> float:
    ...
```

---

## 八、主流程实现（核心伪代码）

请将下面伪代码翻译为 Python 实现（注意用上面定义的函数和 `SemanticUnit` 结构）：

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

在 Python 实现时，将集合操作 `pool ∪ buffer_new_units` 实现为合适的数据结构（比如 list + 去重，或 dict keyed by text）。

---

## 九、其他要求

* 为关键步骤添加日志输出（例如使用 `logging` 模块），方便查看：

  * 每轮 outer_iter
  * 每轮探索到多少新需求
  * 生成了多少版 SRS
* 所有 TODO / 简化实现请用注释标明，方便后续替换为更精细的逻辑。
* 最终输出完整的 Python 源码文件内容（可以分模块），不需要实际调用 OpenAI 运行，只需逻辑正确、结构清晰，配置好 API Key 后可以直接运行。

---

以上就是完整需求。请严格按照这些说明完成实现，尤其注意：

* `SemanticUnit` 使用 `text + grade + vector`
* `srs_generator` 模板必须要求：**对 grade=1 的语义单元进行细化扩展**。
