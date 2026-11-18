# ECE4300 Final Project 交接说明

## 项目目标

用 HypoSpace 衡量答题 Agent 能否为同一组观测找到尽可能多的不同解释。每次提交的是一个候选解，评分器用标准解集合计算覆盖率。这里的“继续训练”指恢复评测进程、历史答案和搜索树，实验没有更新模型权重。

- `agents/` 实现直接回答、历史记忆和推理树等策略。普通 MCTS 由模型选择分支。两步反思先调用模型总结最近最多 8 条候选，再把总结附在下一次搜索提示词后。
- `experiments/direct_evaluation.py` 负责选择题目、执行实验、评分、保存断点和恢复状态。
- `experiments/boolean_scoring.py` 负责布尔表达式解析和结构判分。表达式需要满足观测和结构约束。同一真值表可以对应多个不同的逻辑结构。参考解只供评分，复盘使用历史输出。
- `HypoSpace/_HypoSpace-main/` 是上游基准代码。项目使用的固定题库在 `datasets/`。续跑时要使用相同的题库文件，加载时会校验文件哈希。

## 环境与模型

所有示例从 `Final Project` 根目录执行。Windows 可以用现有的 `.venv/Scripts/python.exe`，或先创建虚拟环境并用它安装依赖。

```powershell
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m unittest discover -s tests -v
```

首次配置时，把 `config/llm.json.example` 复制为 `config/llm.json`，从本机配置读取方舟 Coding Plan 密钥，或设置 `ARK_API_KEY`。现有配置应保留密钥，别用示例文件覆盖。长跑沿用 `timeout: 600` 秒。`config/llm.json` 已被 Git 忽略。

答题评测入口固定使用 **`deepseek-v4.1-flash-economy`**。请求走方舟 Coding Plan 的 `https://ark.cn-beijing.volces.com/api/coding/v3`。评测结果记录请求模型和每次响应返回的模型名。通用的 `python -m llm` 可以单独检查接口。上游套件与历史答题记录中的模型字段用于溯源。

[方舟 Coding Plan 支持模型与接入说明](https://docs.volcengine.com/docs/ark/coding-plan-personal-ai-other-tools?lang=zh)

## 选择实验入口

- 正式题库用 `experiments.test_hard_balanced`，共 **136 题、3,600 个标准解**。因果图 8 题 1,203 解，3D 结构 31 题 1,197 解，布尔表达式 97 题 1,200 解（含扩展集 17 题）。首题有 638 个标准解，启动后可能长时间显示 0 道已处理。
- 验证题库用 `experiments.test_hard_boolean`，固定一道 40 解布尔题。旧的固定三道高难题用 `experiments.test_hard_quick`。
- 历史实验入口包括 `test_full_baseline`（105 道基础题）、`test_medium`（30 道题）和 `test_hard_full`（原高难 90 题）。

```powershell
$py = ".\.venv\Scripts\python.exe"
$runner = @("-u", "-m", "experiments.test_hard_balanced", "--provider-token-limit")
$tree = @("--method", "mcts-tree", "--tree-policy", "llm", "--duplicate-reward", "-1", "--exploration-list", "off")
$two = @("--two-step-reflection", "--unbounded-tree")
& $py @runner @tree @two --plan
& $py -m experiments.test_hard_boolean @tree @two --provider-token-limit --search-cap 120 --plan
```

`--plan` 只在本地装载题库和显示预算。移除它会使用 API 并生成独立的时间戳结果文件。要做同一模型的正式比较，可以分别运行 `@runner --method direct`、`@runner --method memory`、`@runner @tree --child-limit 7` 和 `@runner @tree @two`。每行通过 `& $py` 执行，并行时分别放在不同终端。每个进程使用唯一的结果文件，同一个断点只能有一个写入进程。建议同时为每组指定各自的 `--pause-file`，便于单独暂停。

直接法和记忆法每题最多提交“标准解数 + 5”次。普通 MCTS 的内部搜索预算是其两倍，且沿用每节点最多 7 个子节点的历史配置。当前两步反思组也用该内部搜索预算，树深度、子节点数和提交数均无独立上限。一次内部搜索最多使用两次 API 请求，复盘失败时这一轮可能没有进入搜索。有效的重复解仍提交。找全标准解会提前结束。正式组遇到连续 3 次请求失败会记录已有结果并跳到下一题，汇总时应注明这些题。

## 现有断点与恢复

2026-11-17 18:06 保存的正式进度，以下旧运行请求模型均为 `ark-code-latest`。以本地 JSON 的最新内容为准。

- 直接法 `results/direct_hard_balanced_checkpoint.json` 已处理 136 题，其中 25 题因请求失败跳过。按题等权覆盖率（含跳过题已有结果）15.56%。
- 记忆法 `results/memory_hard_balanced_checkpoint.json` 已处理 49 题，其中 9 题因请求失败跳过。当前布尔题 `n2_inputs_00_01_outputs_10` 已找到 9/12 解。
- 普通 MCTS `results/mcts-tree_hard_balanced_checkpoint.json` 已处理 40 题，其中 13 题因请求失败跳过。当前布尔题 `n1_inputs_00_outputs_1` 已找到 21/25 解。
- 两步反思 `results/mcts-tree-list-off-branch-examples-two-step-unbounded_hard_balanced_20261117T052744652867Z.json` 正在做第 1 题。146 次内部搜索找到 144/638 解。文件名没有 `unbounded` 的两步反思正式断点是更早的配置。

三组旧进程已设置暂停。暂停文件为 `results/PAUSE_HARD_BALANCED`（记忆法、普通 MCTS 共用）和 `results/PAUSE-hard-balanced-two-step`（两步反思）。进程在当前请求结束后暂停，旧进程已经载入原模型。切换到固定模型时保留这两个暂停文件，以免误唤醒旧进程。

如果要用新模型接着已有进度跑，使用 `--continue-from`。以下三条分别运行，可并行。每条会创建独立的新结果文件并记录 `model_transition`，保留旧断点、已处理题和当前题状态。当前题可能包含两种模型生成的答案。之前因失败跳过的题也会保留，需要补测时单独安排。要求整个结果都出自固定模型时，应从新结果文件跑完整题库。

```powershell
& $py @runner --method memory --pause-file results/PAUSE_DEEPSEEK_MEMORY --continue-from results/memory_hard_balanced_checkpoint.json
& $py @runner @tree --child-limit 7 --pause-file results/PAUSE_DEEPSEEK_MCTS --continue-from results/mcts-tree_hard_balanced_checkpoint.json
& $py @runner @tree @two --pause-file results/PAUSE_DEEPSEEK_TWO_STEP --continue-from results/mcts-tree-list-off-branch-examples-two-step-unbounded_hard_balanced_20261117T052744652867Z.json
```

新进程的输出会显示 `Saving results to ...`。用同一套入口和参数，在对应的 `--pause-file` 路径创建空文件以暂停，删除该空文件可恢复仍存活的进程。若进程已退出，改用 `--resume results/<新断点文件名>.json`，并先确认原进程已退出。恢复会跳过已处理题并重放当前题的历史以重建搜索树。断点模型、题库文件和方法配置必须匹配。历史结果如果位于另一台机器，先把 JSON 断点复制到 `results/`，并复用相同版本的题库和评测代码。

## 查看结果

- `cases` 是已处理的题，`interrupted_case` 是正在做的题。用 `stop_reason=request_errors` 识别因请求失败跳过的题，`completed=true` 表示已遍历题库。
- 单题覆盖率是找到的不同标准解数除以该题标准解数，对题取等权平均得到归一化覆盖率（`summary.all.recovery`）。汇总只对正常结束的题（`stop_reason != request_errors`）取平均，因请求失败跳过的题不计入覆盖率、单独列出待补测。`summary.*.normal_cases` 记录正常结束题数，`python -m experiments.rescore_summary <结果文件...>` 用该口径重算并导出 `skipped_cases_to_rerun` 清单。
- 比较方法时优先对齐同一批题、同一个模型及相同停止规则。分别报告内部搜索轮数、答案提交数、API 请求数与 token。两步反思的 `reflection_response` 记录额外 API 调用，整轮结束时汇总到 `api_calls`。
