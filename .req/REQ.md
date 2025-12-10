你现在是一个资深后端工程师，需要根据下面的算法说明实现一套可运行的代码（推荐用 Python 实现，如果我没特别指出语言，就默认用 Python）。

### 目标

实现一个“多版本 SRS 生成器”，根据给定的 3 个输入文档，反复执行：

* 需求探索（requirement_explorer）
* 需求澄清（requirement_clarifier）
* 需求池更新（pool）
* SRS 生成（srs_generator）

每一轮外层迭代生成一版新的 SRS 文档，并写入磁盘文件：

```text
OUTPUT_DIR/srs_iter_1.md
OUTPUT_DIR/srs_iter_2.md
...
OUTPUT_DIR/srs_iter_K.md
```

---

### 接口与总体结构要求

请实现一个主函数，例如：

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

其中：

* `d_orig`：原始 SRS 文档内容（字符串，假设是 markdown / 纯文本）
* `r_base`：requirement_base 文本内容
* `d_base`：基线方法生成的 SRS 文本内容
* `output_dir`：用于存放 `srs_iter_k.md` 的目录路径（若不存在需要自动创建）
* `rho`：扩展倍率
* `max_outer_iter`：外层最大迭代次数
* `max_inner_iter`：每轮外层迭代中 requirement_explorer 的最大调用次数
* 函数无返回值，只在磁盘中产生一批 `.md` 文件

### 必须实现/预留的内部函数

请按以下签名实现或预留这些函数（如果无法真正实现内容，就写清楚 TODO 并保证整体结构可运行）：

```python
from typing import List, Any

class SemanticUnit:
    """
    表示一个语义单元/需求单元。
    至少包含：
    - text: 原始文本
    - grade: 澄清后的打分（float 或 int，初始可为 None）
    - 其他你认为必要的字段
    """
    def __init__(self, text: str, grade: float | None = None):
        self.text = text
        self.grade = grade


def split_to_semantic_units(d_base: str) -> List[SemanticUnit]:
    """
    将基线 SRS 文本拆分为语义单元列表。
    要求：
    - 使用 OpenAI 大语言模型进行语义拆分
    - 提示词需要模板化，每个提示词对应一个模板文件
    """
    ...


def requirement_clarifier(units: List[SemanticUnit], d_orig: str) -> List[SemanticUnit]:
    """
    在原始 SRS 文档 d_orig 的上下文下，对输入的语义单元进行澄清和打分。
    要求：
    - 使用 OpenAI 大语言模型进行需求澄清和打分
    - 提示词需要模板化，每个提示词对应一个模板文件
    - 不改变 units 的数量
    - 为每个 unit 赋值 grade 属性，范围大致在 (0, 2] 或者你自定义的逻辑
    """
    ...


def requirement_explorer(
    r_base: str,
    negative_pool: List[SemanticUnit],
    required_num: int,
) -> List[SemanticUnit]:
    """
    基于 requirement_base 和 negative_pool 探索最多 required_num 条新的语义单元。
    要求：
    - 使用 OpenAI 大语言模型进行需求探索
    - 提示词需要模板化，每个提示词对应一个模板文件
    - 输出数量 <= required_num
    - 输出的文本与 negative_pool 中已有内容尽量不同（通过 filter_low_similarity 进行相似度过滤）
    """
    ...


def filter_low_similarity(
    explored_units: List[SemanticUnit],
    existing_pool: List[SemanticUnit],
) -> List[SemanticUnit]:
    """
    过滤掉与 existing_pool 文本相似度过高的语义单元。
    要求：
    - 使用 OpenAI 进行文本向量化，比较余弦相似度
    - 过滤掉相似度超过阈值的语义单元
    """
    ...


def srs_generator(grade_units: List[SemanticUnit]) -> str:
    """
    根据输入的语义单元列表生成一份 SRS 文本（字符串）。
    要求：
    - 使用 OpenAI 大语言模型生成 SRS 文档
    - 提示词需要模板化，每个提示词对应一个模板文件
    """
    ...
```

文件写入函数：

```python
import os

def write_srs_to_disk(curr_srs: str, iter_idx: int, output_dir: str) -> None:
    """
    将当前版本 SRS 写入 output_dir/srs_iter_{iter_idx}.md
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"srs_iter_{iter_idx}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(curr_srs)
```

---

### 迭代控制逻辑（核心伪代码需实现）

请严格按照以下伪代码实现主逻辑（可以适当翻译为 Python 风格）：

```text
input:
    D_orig      # 原始 SRS 文档
    R_base      # requirement_base
    D_base      # 基线方法生成的 SRS 文档 (baseline SRS)
    OUTPUT_DIR  # 输出目录路径

# 0. 从 pool 生成一版 SRS
function generate_srs_from_pool(pool):
    grade_units  = { u in pool | 0 < grade(u) <= 2 }
    return srs_generator(grade_units)

# 1. 基线初始化
baseline_units = split_to_semantic_units(D_base)

pool = ∅
clarified_baseline_units = requirement_clarifier(baseline_units, D_orig)
pool = pool ∪ clarified_baseline_units

rho = ...  # 由调用方传入
N   = ceil(rho * size(baseline_units))

MAX_OUTER_ITER = ...
MAX_INNER_ITER = ...

outer_iter = 1

# 2. 外层循环
while outer_iter <= MAX_OUTER_ITER:

    buffer_new_units = ∅
    inner_iter = 0

    # 2.1 内层循环：多次 explorer，尝试凑够 N 条 brand-new requirements
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

    # 2.2 内层结束，检查是否有增量
    if size(buffer_new_units) == 0:
        break

    if size(buffer_new_units) < N:
        break

    # 2.3 澄清 + 更新池
    clarified_new_units = requirement_clarifier(buffer_new_units, D_orig)
    pool = pool ∪ clarified_new_units

    # 2.4 从 pool 生成一版 SRS 并写盘
    curr_srs = generate_srs_from_pool(pool)
    write_srs_to_disk(curr_srs, outer_iter, OUTPUT_DIR)

    outer_iter = outer_iter + 1

# 3. 无返回值，结果是磁盘上的 srs_iter_k.md 文件
```

要求：

* 保留上述控制逻辑不变，只需按所用语言习惯改写语法；
* 要有合理的日志打印 / 注释，方便调试（例如打印每轮 outer_iter、buffer_new_units 数量等）；
* 代码应结构清晰，可作为库调用，也可以通过一个简单的 `if __name__ == "__main__":` 示例演示如何传入文本和参数运行。

---

请根据上述说明，输出一份完整、可运行的代码实现（包含必要的类定义、函数、主入口），并在必要的地方用 TODO 标记未来可替换为真实 NLP/LLM 能力的占位实现。
