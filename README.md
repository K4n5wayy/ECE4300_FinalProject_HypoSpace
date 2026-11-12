# ECE4300 期末项目

本项目探索提高大语言模型在 HypoSpace 基准测试上的表现。

## 官方代码来源

`HypoSpace/_HypoSpace-main/` 为下载解压的官方代码，来源：
https://github.com/CTT-Pavilion/_HypoSpace

该目录保留作者的说明和署名。项目目前处于环境准备阶段，尚未生成实验成绩。

## 项目记录

后续在本仓库记录环境配置、改进方法、实验设置和可复现结果。
请勿提交模型接口密钥；本地密钥应通过环境变量或未纳入版本记录的配置保存。

## 调用火山方舟 Coding Plan

在本文件所在目录运行，要求 Python 3.10 或更新版本，模型调用模块无需安装额外依赖：

```powershell
python -m llm "请用 Python 编写一个计算列表平均值的函数"
```

先将 `config/llm.json.example` 复制为 `config/llm.json`，在本地文件的
`ark_coding_plan.api_key` 中填写密钥，可同时设置 `model` 和 `timeout`（秒）。
本地文件已加入 `.gitignore`，只有不含密钥的示例文件可以上传。
客户端自动定位项目根目录下的配置，不依赖当前工作目录；也支持 `--config 文件路径`。
优先级：显式参数 > 本地配置 > 环境变量 > 默认值。空密钥会回退读取 `ARK_API_KEY`。
不会自动读取 `.env` 文件。自定义配置路径必须自行加入忽略规则。
可通过 `--model 模型名称` 覆盖配置，模型默认 `ark-code-latest`。
该别名会随控制台设置变化，正式实验应固定具体模型并记录返回的模型名称。
可用 `--max-tokens 2048` 调整输出上限；返回的 `finish_reason=length` 表示可能被截断。

### 接口设计

- `LLMClient` 是抽象父类，业务代码只依赖其 `generate()` 方法。
- `OpenAICompatibleClient` 实现兼容协议的 HTTP 通信。
- `ArkCodingPlanClient` 固定方舟 Coding Plan 地址并读取配置。
- API key 是实例配置，不是每个密钥都创建一个子类；不同协议可新增父类的实现。
- 返回值包含文本、实际模型名、用量和停止原因；请求失败抛出异常，不作为模型答案计分。
- 不自动重试，避免重复消耗额度；目前支持同步文本调用，未接入官方评测程序。

```python
from llm import LLMClient, ArkCodingPlanClient

def solve(client: LLMClient, question: str) -> str:
    return client.generate(question).text

# 先在 config/llm.json 中填写密钥
client = ArkCodingPlanClient()
```

离线测试（不调用真实模型、不消耗额度）：

```powershell
python -m unittest discover -s tests -v
```

官方参考：
- https://docs.volcengine.com/docs/ark/coding-plan-personal-get-started
- https://docs.volcengine.com/docs/ark/coding-plan-personal-ai-opencode

Coding Plan 专用地址为 `https://ark.cn-beijing.volces.com/api/coding/v3`。
不要替换为普通方舟 `/api/v3` 地址，官方说明该地址可能产生额外费用。
使用范围和额度以你的套餐说明为准；本模块的接入完成不代表已验证批量评测可用。

## Agent 基线与单场景真实测试

`agents/base.py` 定义统一的 `BaseAgent.solve()` 接口。
`agents/direct.py` 的 `DirectAgent` 接收 `LLMClient`，直接调用一次模型，不使用记忆、工具或自我修正。
不同平台通过客户端替换，后续策略通过新增 Agent 子类扩展。

官方测试包括因果图、三维体素重建、布尔逻辑。本次只测试一个人工选定的因果场景：
节点 A、B、C，干预 A 后 B、C 受影响。官方枚举程序给出 5 个合法图。
复用官方提示词生成器、解析器和验证器；标准答案仅供评分，不交给模型。
5 次独立生成采用相同提示，不附带历史答案，因此不同于官方带历史的基线流程。
有效率 = 有效次数 / 调用次数；独特性 = 不同可解析图数量 / 调用次数；
覆盖率 = 命中的不同标准答案数量 / 5。接口失败也计入调用预算。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m experiments.test_direct_agent --queries 5
```

这是实际联网实验，会消耗模型额度，不属于离线单元测试。
`results/direct_causal_*.json` 保存提示词、原始回答、用量、实际模型名、评分和官方脚本指纹。
仅一道简单题和一次运行，不能代表整个 HypoSpace 的成绩。
