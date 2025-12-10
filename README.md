# 多版本 SRS 生成器

基于 OpenAI 大语言模型和向量嵌入的多版本软件需求规格说明书（SRS）生成系统。

## 功能说明

本系统通过迭代的方式，反复进行"需求探索 → 澄清评分 → 更新需求池 → 生成 SRS"，每一轮外循环生成一版 SRS 文档。

## 项目结构

```
srs-gen2/
├── main.py                    # 主函数入口
├── srs_pipeline.py            # 核心业务逻辑函数
├── models.py                  # 数据模型（SemanticUnit）
├── openai_utils.py            # OpenAI API 封装
├── config.py                  # 配置加载模块
├── config.yaml                # 配置文件（YAML 格式）
├── pyproject.toml             # 项目配置（uv 使用）
├── prompts/                   # 提示词模板目录
│   ├── split_to_semantic_units.md
│   ├── requirement_explorer.md
│   ├── requirement_clarifier.md
│   └── srs_generator.md
└── README.md
```

## 安装

使用 [uv](https://github.com/astral-sh/uv) 管理项目环境：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync
```

## 配置

### 配置文件（推荐）

项目使用 YAML 配置文件 `config.yaml` 管理所有参数，避免硬编码。

1. **编辑配置文件** `config.yaml`：

```yaml
# OpenAI API 配置
openai:
  api_key: "your-api-key-here"  # 或留空，使用环境变量
  base_url: null  # 可选，用于代理或兼容服务
  
  # Chat 模型默认配置
  chat:
    model: "gpt-4o-mini"
    temperature: 0.2
  
  # Embedding 模型配置
  embedding:
    model: "text-embedding-3-small"
  
  # 各组件独立的模型和温度配置（可选）
  # 如果未指定，则使用 chat 的默认值
  components:
    split_to_semantic_units:
      model: null  # 如果为 null，则使用 chat.model
      temperature: null  # 如果为 null，则使用 chat.temperature
    requirement_explorer:
      model: null
      temperature: null
    requirement_clarifier:
      model: null
      temperature: null
    srs_generator:
      model: null
      temperature: null

# 迭代参数
iteration:
  rho: 0.5
  max_outer_iter: 5
  max_inner_iter: 3

# 相似度过滤
similarity:
  threshold: 0.8

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


```

2. **或使用环境变量**（优先级更高）：

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
```

### 配置说明

#### OpenAI 配置

- **openai.api_key**: OpenAI API 密钥（可通过环境变量 `OPENAI_API_KEY` 设置）
- **openai.base_url**: API 基础 URL（可选，可通过环境变量 `OPENAI_BASE_URL` 设置）
- **openai.chat.model**: Chat 模型默认名称，默认 `gpt-4o-mini`
- **openai.chat.temperature**: Chat 温度默认参数，默认 `0.2`
- **openai.embedding.model**: Embedding 模型名称，默认 `text-embedding-3-small`

#### 组件独立配置

系统支持为不同的组件配置独立的模型和温度参数，以便针对不同任务优化性能：

- **openai.components.split_to_semantic_units**: 语义单元拆分组件的配置
- **openai.components.requirement_explorer**: 需求探索组件的配置
- **openai.components.requirement_clarifier**: 需求澄清评分组件的配置
- **openai.components.srs_generator**: SRS 生成组件的配置

每个组件可配置：
- `model`: 模型名称，如果为 `null` 则使用 `openai.chat.model` 的默认值
- `temperature`: 温度参数，如果为 `null` 则使用 `openai.chat.temperature` 的默认值

**配置示例**：

```yaml
openai:
  components:
    split_to_semantic_units:
      model: "gpt-4o"  # 使用更强大的模型进行拆分
      temperature: 0.1  # 使用较低温度保证稳定性
    requirement_explorer:
      model: "gpt-4o-mini"  # 使用更便宜的模型进行探索
      temperature: 0.7  # 使用较高温度增加创造性
    requirement_clarifier:
      model: null  # 使用默认模型
      temperature: null  # 使用默认温度
    srs_generator:
      model: "gpt-4o"
      temperature: 0.3
```

#### 其他配置

- **iteration.rho**: 每轮探索新需求的比例（相对于基线单元数量），默认 `0.5`
- **iteration.max_outer_iter**: 最大外循环次数，默认 `5`
- **iteration.max_inner_iter**: 最大内循环次数，默认 `3`
- **similarity.threshold**: 相似度阈值（0-1），默认 `0.8`
- **logging.level**: 日志级别，默认 `INFO`
- **logging.format**: 日志格式

#### 容错重试配置

为保证长流程的稳定性，所有依赖 LLM 的组件都支持统一的重试配置：

```yaml
robustness:
  split_to_semantic_units:
    max_attempts: 3        # 任何异常时的最大重试次数
    delay_seconds: 2.0     # 每次重试前的等待时间
  requirement_explorer:
    max_attempts: 3
    delay_seconds: 2.0
  requirement_clarifier:
    max_attempts: 3
    delay_seconds: 2.0
  srs_generator:
    max_attempts: 3
    delay_seconds: 2.0
```

若未配置，则默认最多重试 3 次，每次间隔 2 秒。遇到 JSON 解析失败、响应缺少字段、网络抖动等任何异常都会自动重试，直至成功或达到上限。

## 使用方法

### 基本用法

使用 `uv run` 运行程序：

```python
from main import run_srs_iteration

# 准备输入数据
d_orig = "..."  # 原始 SRS 文档内容（字符串）
r_base = "..."  # 需求基础文档内容（字符串）
d_base = "..."  # 基线方法生成的 SRS 文档内容（字符串）

# 运行迭代（参数可从配置文件读取，也可手动指定）
run_srs_iteration(
    d_orig=d_orig,
    r_base=r_base,
    d_base=d_base,
    output_dir="./output",      # 输出目录
    # 以下参数可选，如果不指定则使用 config.yaml 中的值
    # rho=0.5,
    # max_outer_iter=5,
    # max_inner_iter=3,
)
```

### 使用 uv 运行

```bash
# 运行 Python 脚本
uv run python main.py

# 或直接运行模块
uv run -m main
```

## 输出结果

系统会在 `output_dir` 目录下生成多版 SRS 文档：

- `srs_iter_1.md` - 第 1 版 SRS
- `srs_iter_2.md` - 第 2 版 SRS
- ...

## 核心功能

### 1. split_to_semantic_units
将基线 SRS 文档拆分为独立的语义单元。

**可配置项**：可通过 `openai.components.split_to_semantic_units` 配置独立的模型和温度。

### 2. requirement_explorer
基于需求基础文档探索新需求，避免与已有需求重复。

**可配置项**：可通过 `openai.components.requirement_explorer` 配置独立的模型和温度。

### 3. requirement_clarifier
对语义单元进行评分（-2 到 +2），评估与原始 SRS 的吻合度。

**可配置项**：可通过 `openai.components.requirement_clarifier` 配置独立的模型和温度。

### 4. srs_generator
根据已评分的语义单元生成 SRS 文档。**重要**：对 grade=1 的需求会进行细化扩展。

**可配置项**：可通过 `openai.components.srs_generator` 配置独立的模型和温度。

### 5. filter_low_similarity
使用 OpenAI 向量嵌入和余弦相似度，过滤掉与已有需求过于相似的语义单元。

## 注意事项

1. 所有对 LLM 的调用都通过模板化提示词完成，模板文件位于 `prompts/` 目录。
2. `SemanticUnit` 使用 `text + grade + vector` 结构，`vector` 字段用于缓存 embedding，避免重复调用 API。
3. `srs_generator` 会对 grade=1 的语义单元进行细化扩展，包括拆解为子需求、明确输入输出等。
4. 系统使用日志记录关键步骤，方便调试和监控。
5. **组件独立配置**：可以为不同的组件（`split_to_semantic_units`、`requirement_explorer`、`requirement_clarifier`、`srs_generator`）配置不同的模型和温度参数，以便针对不同任务优化性能和成本。如果组件配置为 `null`，则自动使用 `openai.chat` 的默认值。

## 配置优先级

配置加载优先级（从高到低）：
1. 函数参数（如果调用时指定）
2. 环境变量（`OPENAI_API_KEY`, `OPENAI_BASE_URL`）
3. `config.yaml` 配置文件
4. 代码中的默认值

## 自定义配置

可以通过以下方式自定义配置：

1. **修改 `config.yaml`**（推荐）
2. **设置环境变量**（用于敏感信息如 API Key）
3. **通过函数参数**（用于临时调整）

例如，使用自定义配置文件：

```bash
export SRS_GEN2_CONFIG="/path/to/custom-config.yaml"
```

### 组件配置最佳实践

根据不同任务的特点，建议为不同组件选择合适的模型和温度：

- **split_to_semantic_units**：需要准确拆分，建议使用较低温度（0.1-0.3），可使用较强的模型
- **requirement_explorer**：需要创造性探索，建议使用较高温度（0.6-0.8），可使用成本较低的模型
- **requirement_clarifier**：需要准确评分，建议使用较低温度（0.1-0.3）
- **srs_generator**：需要生成高质量文档，建议使用中等温度（0.2-0.4），可使用较强的模型

## 参数说明

- `rho`: 每轮探索新需求的数量 = `ceil(rho * 基线单元数量)`
- `max_outer_iter`: 最大外循环次数，每轮生成一版 SRS
- `max_inner_iter`: 最大内循环次数，每轮内尝试探索新需求的次数
- `threshold`: 相似度阈值（默认 0.8），用于过滤相似需求
