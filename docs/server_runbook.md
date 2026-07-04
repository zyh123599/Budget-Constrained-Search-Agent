# 服务器操作手册(2×A800)

> 面向场景:服务器上没有 Claude Code,所有代码已在本仓库写好,
> 你只需按本手册顺序执行命令,并把指定产物带回来分析。
> 每一步都有"预期输出"和"出错怎么办",照着核对即可。

---

## 0. 前置条件核对

| 项 | 要求 | 检查命令 |
|---|---|---|
| GPU | 2×A800(80G) | `nvidia-smi` |
| 磁盘 | ≥ 200 GB 空闲(建议 NVMe;wiki-18 语料+索引约 60–70 GB,模型+检查点+数据集再占几十 GB) | `df -h .` |
| conda | 任意较新版本 | `conda --version` |
| 网络 | 能访问 HuggingFace 与 GitHub;不通则用镜像:`export HF_ENDPOINT=https://hf-mirror.com` | `curl -sI https://huggingface.co \| head -1` |
| CUDA 驱动 | 支持 cu121 | `nvidia-smi` 右上角 CUDA Version ≥ 12.1 |

建议全程在 `tmux` 里操作(下载和训练都以小时计,防 ssh 断连):

```bash
tmux new -s budget   # 断连后 tmux attach -t budget 恢复
```

---

## 1. 拉代码 + 零 GPU 自检(5 分钟)

```bash
git clone -b claude/research-startup-neh7n4 \
    https://github.com/zyh123599/Budget-Constrained-Search-Agent.git
cd Budget-Constrained-Search-Agent
pip install -e ".[dev]"
pytest                      # 预期:92 passed
python scripts/smoke_test.py   # 预期:前两项 PASS,后两项 SKIP(训练栈还没装)
```

**任何一步不符合预期 → 停下,把完整报错带回来。**

## 2. 搭训练/检索双环境(30–60 分钟,主要是下载编译)

```bash
bash scripts/setup_env.sh
```

- 创建 `searchr1`(训练:torch/vllm/verl/flash-attn)与 `retriever`(faiss-gpu)两个 conda 环境,并克隆 Search-R1 到 `third_party/`。
- **flash-attn 编译最慢**(可 20+ 分钟),卡在 `Building wheel` 是正常的。
- faiss-gpu 的 conda solve 失败时,改用:`conda run -n retriever pip install faiss-gpu-cu12`。

## 3. 下载语料索引 + 常驻检索服务(下载数小时,视带宽)

**单独开一个 tmux 窗口**(服务要一直挂着):

```bash
tmux new-window -t budget -n retriever
bash scripts/setup_retrieval.sh
```

- 下载 wiki-18 语料 + e5 Flat 索引(约 60–70 GB)后启动服务,监听 `:8000`。
- 服务用的是本仓库自带的 `scripts/retrieval_server.py`(API 与上游 Search-R1 完全兼容),
  61GB Flat 索引以 **fp16 分片到两张卡,每卡只占 ~15GB**,给训练留足显存。
- 启动日志会打印 `faiss sees N GPU(s)` 和每卡预估占用——**N 必须是 2**;
  是 1 的话检查该窗口的 `CUDA_VISIBLE_DEVICES` 是否被设置过。
- 看到 `service ready` / `Uvicorn running on ...:8000` 即就绪,**这个窗口不要关**。
- GPU 上放不下(或 faiss GPU 支持有问题)时的兜底:
  `RETRIEVAL_GPU_MODE=cpu bash scripts/setup_retrieval.sh`(索引留内存,需 ~64GB 空闲 RAM)。

**显存预算(两卡各 80GB,索引分片共卡时):** faiss 索引 ~15GB/卡 + e5 编码器 ~1GB,
剩 ~63GB/卡给 vLLM 与训练。`rollout_eval.py` 默认 `--gpu-memory-utilization 0.6`
已按此留了余量;若你把索引放 CPU,可提到 `0.85` 提吞吐。

回到主窗口验证:

```bash
bash scripts/run_baseline.sh env-check
# 预期:打印一段 JSON(检索结果)+ "OK:检索服务可用"
python scripts/smoke_test.py --retrieval-url http://127.0.0.1:8000/retrieve
# 预期:4 项全 PASS
```

## 4. Gate 1:baseline 复现(半天)

### 4a. 官方 checkpoint 推理评估(~1 小时)

```bash
conda run -n searchr1 --no-capture-output python scripts/prepare_data.py \
    --datasets nq --split test --budget-mode grid --grid train \
    --max-per-dataset 500 --out data/eval_nq_test_grid.parquet
bash scripts/run_baseline.sh eval
```

**判据(版本敏感):** 默认检查点(无版本后缀)是 **v0.1 preliminary**(少量训练步数);
论文的 NQ EM=0.480 是 **v0.2** 口径,两者不可直接对照。实测判读:

- `EM ≥ 0.40` 且下面 probe 的 `hit rate ≥ 0.65` → 环境对齐,冻结为本地 baseline;
- 不满足 → 先跑 probe(下一步)把问题定位到检索侧或生成侧,再把输出带回来。
- 想对齐论文数字:`BASELINE_CKPT=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3 bash scripts/run_baseline.sh eval`

**检索侧隔离测试(分钟级,EM 无论高低都建议跑一次留档):**

```bash
bash scripts/run_baseline.sh probe
```

只跑检索不跑生成,统计 gold answer 命中 top-3 段落的比例。
`≥0.65` 检索侧健康;`<0.60` 检索侧有病(脚本会打印排查顺序)。

**baseline 的 Pareto 对比曲线(论文需要,可放到 Gate 1 之后再跑):**

```bash
bash scripts/run_baseline.sh pareto    # 500 题 × 9 预算档,数小时
```

baseline 不感知预算,单点积不出 Pareto 面积;此命令用外部硬截断协议
(token 耗尽/检索配额用完 → 强制作答)扫出它在各预算档下的真实成功率–成本曲线,
这是审稿人必问的"同等约束下 baseline 表现"的答案。

### 4b. 训练闭环冒烟(1–2 小时,跑通即可手动 Ctrl-C)

```bash
bash scripts/run_baseline.sh train-smoke
```

**判据:能看到 verl 的 step 日志、reward/loss 在动、无 OOM。** 跑 20–50 步即可停。
报错时:上游脚本字段名随 verl 版本演进,把 `runs/baseline_train_smoke.log` 带回来,我来改配置。

> **4a + 4b 都过 = Gate 1 基建侧通过**,填进 `docs/experiment_log.md`。

## 5. 方法管线首跑(方法 v1,Gate 2 之前的主线)

### 5a. 生成预算增广数据(10 分钟)

```bash
conda run -n searchr1 --no-capture-output python scripts/prepare_data.py \
    --datasets nq hotpotqa --split train --budget-mode sample --out data/train.parquet
conda run -n searchr1 --no-capture-output python scripts/prepare_data.py \
    --datasets nq --split test --budget-mode grid --grid interp \
    --max-per-dataset 500 --out data/eval_interp.parquet
conda run -n searchr1 --no-capture-output python scripts/prepare_data.py \
    --datasets nq --split test --budget-mode grid --grid extrap \
    --max-per-dataset 500 --out data/eval_extrap.parquet
```

### 5b. SFT 热启动(2–4 小时,含采样)

```bash
bash scripts/run_sft.sh
```

一条龙:基座 Qwen3-4B rollout 采 800 条示范 → hindsight 重标注 → SFT。
产物 `checkpoints/qwen3_4b_sft`。中途看 `kept N, skipped M`:**kept < 100 时说明基座格式遵循率太低**,把 `runs/sft_demos_raw.jsonl` 前几行带回来,我调 prompt 或改用 API 蒸馏。

验证 SFT 效果(格式遵循率应大幅提高):

```bash
conda run -n searchr1 --no-capture-output python scripts/rollout_eval.py \
    --model checkpoints/qwen3_4b_sft --data data/eval_interp.parquet \
    --limit 100 --tensor-parallel 2 --out runs/sft_check.jsonl
conda run -n searchr1 --no-capture-output python scripts/eval_budget_sweep.py runs/sft_check.jsonl
```

### 5c. GRPO 主训练(数天)

把 `configs/grpo_qwen3_4b.yaml` 里 `actor_rollout_ref.model.path` 改成 `checkpoints/qwen3_4b_sft`,然后:

```bash
bash scripts/train_grpo.sh
```

**首跑必看:** verl/Search-R1 版本演进可能导致配置字段名不匹配,报
`ConfigAttributeError`/`MissingMandatoryValue` 时把完整报错带回来,我来对齐字段。
多轮检索 rollout 与 `<budget>` 块逐轮重注入依赖 Search-R1 的 generation 循环,
若其接口对不上,同样把报错带回来——这属于预期内的接线工作,不是设计问题。

### 5d. 评估(每个 checkpoint ~1 小时)

```bash
conda run -n searchr1 --no-capture-output python scripts/rollout_eval.py \
    --model <checkpoint路径> --data data/eval_interp.parquet \
    --tensor-parallel 2 --out runs/grpo_interp.jsonl
conda run -n searchr1 --no-capture-output python scripts/eval_budget_sweep.py \
    runs/grpo_interp.jsonl        # 四大生死指标一站式
```

外推档换 `data/eval_extrap.parquet`。干预实验(§5.4,论文最硬的牌,主训练收敛后再跑):

```bash
conda run -n searchr1 --no-capture-output python scripts/rollout_eval.py \
    --model <checkpoint> --data data/eval_interp.parquet \
    --intervene noise --sigma 0.5 --tensor-parallel 2 --out runs/intervene_s05.jsonl
# σ 扫 {0(=oracle), 0.25, 0.5, 1.0, 2.0},加上 .control.jsonl 自动保存的对照组
```

---

## 6. 带回来给我的东西(按优先级)

1. **每一步的终端输出**(尤其是 EM/violation 汇总行和任何报错栈);
2. `runs/` 下所有 `.jsonl`(轨迹文件,我可以离线重放出全部指标和图);
3. `runs/baseline_train_smoke.log`、wandb 的 run 链接或曲线截图;
4. 修改过的配置文件(如果你为了跑通改了什么)。

拿到这些我就能:判 Gate 1、诊断训练问题、画首张 Pareto 曲线、迭代 reward 权重。

## 7. 常见故障速查

| 症状 | 原因 | 处置 |
|---|---|---|
| HF 下载超时/403 | 网络不通 | `export HF_ENDPOINT=https://hf-mirror.com` 后重跑(建议写进 `~/.bashrc`,全程需要) |
| HF 报 "not a valid model identifier" | 用户名写错:HF 上是 `PeterJinGo`,GitHub 才是 `PeterGriffinJin` | 所有 Search-R1 模型/数据 repo 前缀用 `PeterJinGo/`;全部公开,无需 token |
| faiss 导入失败 / `GLIBCXX_x.x.xx not found` | 系统 libstdc++ 过旧 | `setup_retrieval.sh` 已自动 `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`;手动跑 retriever 环境的命令时同样加上 |
| flash-attn 装不上 / `import flash_attn` 崩(GLIBC 2.31 等旧系统) | 预编译 wheel 依赖较新 GLIBC,但 verl 里有模块级硬 import | `setup_env.sh` 已自动落到 SDPA 垫片;手动装:`conda run -n searchr1 python scripts/install_flashattn_shim.py`(纯 PyTorch,能跑通;数值一致但无 kernel 加速)。`--check` 验证,`--uninstall` 移除 |
| 后台 shell 里 `conda: command not found` | 非交互 shell 没 source conda 初始化 | 直接用绝对路径 python:`/opt/conda/envs/searchr1/bin/python scripts/xxx.py`(绕过 `conda run`);或先 `source /opt/conda/etc/profile.d/conda.sh` |
| 写 JSONL 时 `UnicodeEncodeError` | 语料含非 ASCII,系统 locale 非 UTF-8 | 已在代码里全部 `encoding="utf-8"`(拉最新即可);临时法:`export PYTHONIOENCODING=utf-8 LC_ALL=C.UTF-8` |
| faiss-gpu solve 失败 | conda 源问题 | `pip install faiss-gpu-cu12` |
| faiss cudaMalloc OOM(卡明明空闲) | 索引未分片,fp16 整份(~32GB)复制到单卡失败,或 faiss 构建的多卡支持有问题 | 用本仓库 `retrieval_server.py`(setup_retrieval.sh 已默认):`RETRIEVAL_GPU_MODE=shard`;启动日志确认 `faiss sees 2 GPU(s)`;还不行 → `RETRIEVAL_GPU_MODE=cpu` |
| vLLM 启动 OOM | 检索索引分片占了每卡 ~15GB | 降 `--gpu-memory-utilization`(默认 0.6 已留余量,再降到 0.5);或索引改 CPU 模式后提回 0.85 |
| 检索服务想独占一张卡 | 默认 shard 用两张卡 | `CUDA_VISIBLE_DEVICES=1 RETRIEVAL_GPU_MODE=single bash scripts/setup_retrieval.sh`(单卡 fp16 整份 ~31GB);训练侧则只见 0 号卡 |
| verl 配置字段报错 | 版本演进 | 对照 `third_party/Search-R1/train_grpo.sh` 的写法改 `configs/grpo_qwen3_4b.yaml`,或把报错带回来 |
| rollout EM 异常低 | prompt/模板不匹配 | baseline 检查点务必 `--no-chat`;instruct 模型务必默认 `--chat` |
