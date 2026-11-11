# openTPU 仿真架构与运行方法

本文说明 openTPU 从 kernel/模型到 ISA、RTL 和板级模型的仿真链路，给出各入口的适用场景和常用命令。

## 1. 仿真层次

openTPU 不是只有一个 simulator，而是由多个层次组成：

```text
模型适配器 / kernel
        |
        v
compiler: layout + loop + instruction selection
        |
        v
8 x 32-bit openTPU ISA program + DRAM image
        |
        +--------------------+
        |                    |
        v                    v
ISA simulator           Verilator RTL
opentpu/isasim.py       opentpu/rtlsim.py
        |                    |
        |                    v
        |             top / board model
        |             AXI, memory, control
        |                    |
        +---------+----------+
                  |
                  v
        token/logit/cycle/trace comparison
```

三种主要后端的定位：

| 后端 | 入口 | 主要验证内容 | 速度 |
|---|---|---|---|
| ISA       | `opentpu.isasim`、`backend=isa`       | ISA 功能、DRAM/TMEM 结果、模型正确性                | 快 |
| RTL       | `opentpu.rtlsim`、`backend=rtl`       | SystemVerilog 微架构、握手、周期、RTL 与 ISA 一致性  | 慢 |
| Board-sim | `SimTransport`、  `backend=board-sim` | 板级控制寄存器、AXI/native memory、host driver 协议 | 最慢 |
| FPGA      | `XdmaTransport`、 `backend=board`     | 由 `opentpu/host/board.py` 通过 XDMA 访问真实设备  | 最快 |

RTL 和板级仿真使用 Verilator 编译 SystemVerilog。

## 2. 共同输入和执行流程

无论使用哪种后端，模型运行都遵循相同的高层流程：

1. 模型适配器读取 Hugging Face 配置和 safetensors 权重。
2. runtime 为输入、权重、输出和 KV cache 分配 DRAM 地址。
3. 权重按 `int8`、`int4` 或 `fp4` 格式量化并生成 DRAM image。
4. `@ol.jit` kernel 经过 compiler，生成每个 slice 的 ISA program。
5. 后端接收 `programs`、DRAM image 和运行参数 `R8..R15`。
6. 程序执行 DMA、TMEM、ACT RAM、MXU、VPU、quantizer 和 collective 操作。
7. 后端返回 DRAM、TMEM 和统计信息；模型层据此读取 logits 并继续 decode。

runtime 的核心实现是 [`opentpu/runtime.py`](opentpu/runtime.py)，其中：

- `place()` 负责把 host 参数放进 DRAM image；
- `Weight` 负责权重格式和分片；
- `KVCache` 负责 KV cache 的布局；
- `launch()`/`run()` 负责选择 ISA 或 RTL 执行路径。

模型层的 backend 选择以 [`opentpu/llm/qwen3.py`](opentpu/llm/qwen3.py) 的 `Engine` 为例；
LFM2、Qwen3.5、Gemma 4 等模型在各自的 `opentpu/llm/*.py` 中复用同一套协议。

`Engine` 依据当前位置执行 prefill/decode，把 kernel 编译成 ISA program，
然后通过所选 backend 的统一 `run()` 或 `start()/wait()` 接口提交执行。
以一条 VOP（例如逐元素 ADD）为例：

```text
Engine.step() / Engine._run_rows() （初始化时根据 backend == "isa"，创建 IsaBackend）
    -> compile_step() / compile_rows()
        -> compiler 调用 isa.vop(V_ADD, ...) 生成 Instr(op=VOP)
    -> backend.run(programs)
       ├── backend=isa
       │     -> IsaBackend.run()
       │     -> Machine.load(programs).run()
       │     -> Machine 调度 Slice.step() （循环调度 slice）
       │     -> Slice.execute(ins) （ISA interpreter 的单条指令分派器）
       │     -> Slice._vop(ins) -> F.add(A, B) -> 通过 tput() 写回 TMEM
       └── backend=board
             -> BoardBackend.run()/start()
             -> Board.load_program() / 写入参数 / CTRL.RUN
             -> FPGA otpu_seq 解码并发射 VOP
             -> otpu_slice.u_vpu 执行向量运算
             -> STATUS.HALTED 后 host 读取 logits/结果
```

`Slice.execute()` 和 `_vop()` 是 ISA simulator 的 Python 指令解释器，只会在 `backend=isa` 分支调用。
`backend=board` 时，Python 负责上传 ISA program 和控制运行；
VOP 的硬件执行发生在 FPGA 的 `otpu_seq -> otpu_vpu` 路径中，不会调用 Python 的 `Slice.execute()`。
RTL 仿真也类似，由 SystemVerilog sequencer 和 VPU 执行同一 ISA。

## 3. ISA 功能仿真 (backend=isa)

### 3.1 实现和入口

ISA simulator 位于 [`opentpu/isasim.py`](opentpu/isasim.py)：

- `Config` 描述 slice、MXU、TMEM、ACT RAM、IMEM 和 DRAM 容量；
- `board_config()` 返回 YPCB-00338 的配置；
- `Machine` 执行 ISA program；
- 每个 `Slice` 保存寄存器、PC、TMEM、ACT RAM 和 DRAM；
- `step()` 执行一条指令，检查地址、循环、`WAITW` 和 barrier 等行为。

这是 openTPU 的功能 golden model。它不模拟 SystemVerilog 的握手和精确时序，
但要求程序结束后数据结果与 RTL 一致。

### 3.2 直接运行 kernel

runtime API 的入口是：

```python
from opentpu import runtime

result = runtime.launch(kernel, cfg, backend="isa", **args)
```

其中 `cfg` 是 [`opentpu.isasim.Config`](opentpu/isasim.py)，
`args` 使用 `runtime.Input`、`runtime.Weight`、`runtime.Output` 和 `runtime.KVCache` 等对象。
kernel 的编译和参数约束由 `opentpu.compiler`、`opentpu.language` 以及现有 `opentpu/kernels/` 负责；
新 kernel 建议参考现有 kernel/test，而不是绕过 compiler 直接构造 program。

### 3.3 运行模型和验证，运行正常

```sh
python3 tools/validate.py --model qwen3
python3 tools/validate.py --model lfm2 --wformat fp4 --head-format int8
```

`validate.py` 默认使用 ISA simulator，并将设备输出与 CPU 上的量化 golden 比较。
`--against` 可以把另一个 ISA/RTL/board 运行作为第二个设备进行比较。

ISA 仿真适合首先确认：

- kernel 和模型适配器是否正确；
- 权重布局和量化是否正确；
- KV cache、prefill、decode 是否正确；
- 输出 token/logit 是否符合 golden。

## 4. Verilator RTL 仿真 (backend=rtl)

### 4.1 实现和构建入口

RTL 仿真的 Python 封装位于 [`opentpu/rtlsim.py`](opentpu/rtlsim.py)：

- `RTL_SOURCES` 列出 openTPU 核心 RTL；
- `build()` 调用 Verilator，按 RTL 内容、参数和 Verilator 版本缓存可执行文件；
- `run_sim()` 启动并管理 Verilator 子进程；
- `run()` 将 ISA program 和 DRAM image 交给 RTL testbench；
- `BOARD_UARCH` 提供板卡微架构参数。

核心 RTL 源文件位于 [`rtl/`](rtl/)，通用 testbench 位于 [`sim/verilator/`](sim/verilator/)。

RTL 仿真至少有两种常用范围：

1. `tb_top.sv`：验证核心 slice、ISA 执行和内存模型。
2. `tb_board.sv`：验证板级控制寄存器、程序加载、两个内存通道和 host script 协议；
   它不模拟真实 PCIe、XDMA PHY、FPGA 时钟树或 DDR PHY 校准。

### 4.2 直接运行 RTL 后端，运行正常

```sh
python3 tools/validate.py \
  --model lfm2 \
  --backend rtl \
  --against isa \
  --tokens 3 \
  "The capital of France is"
```

RTL 后端由 [`opentpu/llm/rtl_backend.py`](opentpu/llm/rtl_backend.py) 实现，
它保留 DRAM image，在每次 run 时调用 `rtlsim.run()`。
真实模型在 RTL 上很慢，通常使用短 prompt 和少量 token。

### 4.3 直接调用 `rtlsim.run()`

性能测试、协同仿真和专项工具通常直接调用：

```python
from opentpu import rtlsim

drams, tmems, stats = rtlsim.run(
    cfg,
    programs,
    images,
    uarch=rtlsim.BOARD_UARCH,
    axi=True,
    boot=True,
)
```

常用参数：

- `uarch`：选择窗口、读写端口、FIFO 等微架构参数；
- `axi=True`：启用 AXI 相关路径；
- `boot=True`：启用板级启动/加载流程；
- `trace=True`：收集硬件 trace；
- `stall`、`dram_lat`：控制内存延迟和随机阻塞；
- `args`：传入运行时参数；
- `max_cycles`：防止死循环。

### 4.4 RTL 与 ISA 的正确性比较

```sh
python3 tools/validate.py \
  --model qwen3 \
  --backend rtl \
  --against isa \
  --tokens 3 \
  "The capital of France is"
```

通过条件通常包括相同 token、相同 logits，以及必要时的 bit-exact 检查。
如果只需要测试核心而不运行完整模型，可以运行 pytest：

```sh
python3 -m pytest -q tests/test_rtlsim_build.py
```

完整 RTL 测试还需要 Verilator 5 和项目依赖。

### 4.5 `validate.py` 与 `pytest` 的关系

命令：

```sh
python3 tools/validate.py \
        --model lfm2 \
        --backend rtl \
        --against isa \
        --tokens 3 \
        "The capital of France is"
```

各参数的含义是：

| 参数 | 含义 |
|---|---|
| `tools/validate.py` | 模型级验证入口，不是 pytest 测试文件 |
| `--model lfm2`      | 加载 LFM2 的 tokenizer、配置和权重 |
| `--backend rtl`     | 把主设备设为 RTL 后端；首次遇到新的 RTL/参数组合时由 Verilator 编译 testbench |
| `--against isa`     | 再用相同输入、权重和配置运行 ISA simulator，并比较两个设备结果 |
| `--tokens 3`        | 最多生成 3 个 token；这里采用 greedy decode |
| 最后的字符串          | 输入 prompt，先执行 prefill，再执行 decode |

实际执行顺序如下：

1. 加载 LFM2 权重，构造量化后的 DRAM image 和模型运行配置。
2. 创建 `Engine(..., backend=RtlBackend)`；`RtlBackend` 将 kernel 编译出的
         ISA programs、DRAM image 和参数交给 [`opentpu/rtlsim.py`](opentpu/rtlsim.py)。
3. `rtlsim.build()` 调用 Verilator 生成或复用 `tb_top` 可执行文件，随后运行
         RTL testbench，完成 prompt 的 prefill 和最多 3 步 decode。
4. 用 ISA simulator 对同一个 prompt 再执行一次相同流程。
5. 比较生成 token 和逐步 logits；成功时通常会看到 `same tokens`、
         `logits bit-exact` 和 `PASS`。
6. 除非指定 `--no-golden`，脚本还会把设备结果与 CPU/Hugging Face 的量化
         golden 做额外比较。

因此，这条命令是一次**模型级端到端对比**：它重点回答“当前 LFM2 prompt 在 RTL 上的结果是否与 ISA simulator 一致”。
它会实际运行模型，速度通常比单元测试慢很多；`--tokens 3` 只是缩短 decode 阶段，不会跳过模型加载、权重布局、kernel
编译或 RTL 编译。

`python3 -m pytest -q` 则是另一条入口：

```sh
python3 -m pytest -q
```

- pytest 根据 [`pyproject.toml`](pyproject.toml) 的 `testpaths = ["tests"]`
        自动发现整个 `tests/` 目录；`-q` 只控制输出详细程度。
- 测试范围比上面的 LFM2 命令更广，包含 ISA、compiler、kernel、FP、host、
        board 和 RTL 等测试，不等价于“运行一次 `validate.py`”。
- 部分 RTL 测试会通过 `opentpu.rtlsim` 间接调用 Verilator，因此可能复现 Verilator 编译错误；
        但许多 pytest 用例只测试 Python/ISA/compiler 逻辑，不会启动 RTL。
- `tests/test_validate.py` 可以测试 `validate.py` 的部分辅助逻辑，但默认全量
        pytest 不会自动执行上面的具体 LFM2 prompt，除非测试代码显式调用它。

两者的关系可以概括为：

```text
python3 -m pytest -q
                -> 多个独立测试，覆盖整个项目
                -> 其中部分测试 -> rtlsim / Verilator

python3 tools/validate.py ...
                -> 一次指定模型和 prompt 的端到端运行
                -> RTL run + ISA run + logits/token 对比 + CPU golden 对比
```

排查 RTL 编译问题时，先运行短的 `validate.py` 命令可以快速确认模型级 RTL 路径；
修复完成后再运行 `python3 -m pytest -q`，确认没有影响其他模块。
两条命令应使用同一个 Python 环境，并确保该环境安装了 `numpy`、PyTorch、Transformers
以及 Verilator 5 等项目依赖。

## 5. 板级 Verilator 仿真 (backend=board-sim)

### 5.1 结构

仿真的 Python 封装依然位于 [`opentpu/rtlsim.py`](opentpu/rtlsim.py)。
板级 testbench 是 [`sim/verilator/tb_board.sv`](sim/verilator/tb_board.sv)。
它实例化 `otpu_board`，并通过 host script 模拟 AXI-Lite 控制访问：

```text
SimTransport
    |
    | host.txt: W / R / P / C
    v
tb_board.sv
    |
    +-- AXI-Lite -> otpu_ctrl
    +-- program loader / run / status
    +-- otpu_board -> openTPU slice
    `-- two memory channels
            +-- otpu_chmem (MEM_NATIVE=2, default)
            `-- otpu_native_mem (MEM_NATIVE=1)
```

脚本命令含义：

| 命令 | 含义 |
|---|---|
| `W addr value`      | AXI-Lite 写寄存器 |
| `R addr`            | 读取寄存器并输出 `REG` |
| `P addr mask value` | 轮询寄存器直到 `(read & mask) == value` |
| `C cycles`          | 等待 core clock 周期 |

默认每次 flush 都是一次新的 RTL 仿真；channel DRAM image 会保留，但控制寄存器、IMEM 和 TMEM 从 reset 开始。
因此 `Board` 协议会把一次 program load、run 和结果读取放进同一次 flush。
这一层模拟的是板卡 host protocol，不是完整 PCIe 链路。真实硬件中的 PCIe/XDMA 驱动被 `SimTransport` 替代。

### 5.2 通过 host driver 运行

```sh
otpu-chat --model lfm2 --backend board-sim
```

也可以使用模块入口（即 otpu-chat 的实现位置）：

```sh
python3 -m opentpu.host.chat \
  --model lfm2 \
  --backend board-sim
```

`board-sim` 的调用链是：

```text
host.chat.make_backend("board-sim")
    -> BoardBackend
    -> Board
    -> SimTransport
    -> tb_board.sv
    -> otpu_board / RTL
```

因此它能复用真实卡使用的 `BoardBackend`、寄存器协议、程序加载和轮询逻辑。
代价是每个 flush 都要编译或运行 Verilator，完整模型会非常慢；
建议先使用 `--tokens 1` 或短 prompt。

### 5.3 self-test 和诊断

```sh
otpu-selftest --sim --model lfm2 --tokens 1
otpu-diag --sim
```

`--sim` 使用板级 Verilator 模型，适合验证从寄存器、loader、内存测试到模型运行的整条 host 流程。
它不能替代真实卡上的 PCIe link、XDMA 驱动、DDR PHY 校准和板级时钟验证。

## 6. FPGA 真实板卡 (backend=board)

### 6.1 准备、诊断和运行命令

真实板卡使用 `backend=board`，默认设备前缀是 `/dev/xdma0`。运行前需要满足：

- FPGA 已经加载 openTPU bitstream；
- PCIe 链路已经枚举，Xilinx XDMA 驱动已经加载；
- `/dev/xdma0_user`、`/dev/xdma0_h2c_0` 和 `/dev/xdma0_c2h_0` 存在；
- bitstream 的 DDR3 控制器已经完成校准。

#### 设备节点是如何产生的

这三个路径不是 Python 创建的，也不是 FPGA 逻辑直接创建的文件，而是 Linux
PCIe/XDMA 驱动注册后的字符设备节点：

```text
openTPU bitstream
                -> FPGA PCIe endpoint + XDMA BAR/AXI 接口
                -> BIOS / PCIe core 枚举 PCI function
                -> Linux 看到 Xilinx device 10ee:7028
                -> udev 将该 PCI function 的 driver_override 设为 xdma
                -> 内核加载并绑定项目版 Xilinx XDMA 驱动
                -> XDMA 驱动注册 /dev/xdmaN_* 字符设备
                -> udev 重新应用权限规则
```

设备编号 `N` 由系统枚举顺序决定，所以默认是 `/dev/xdma0_*`，但多张卡或枚举
顺序变化时可能变成 `/dev/xdma1_*`。三个节点的作用不同：

| 节点 | 产生的接口 | Python 中的用途 |
|---|---|---|
| `/dev/xdma0_user`  | XDMA 的 user BAR，映射 FPGA BAR0 | `mmap()` 后访问 AXI-Lite 控制寄存器 |
| `/dev/xdma0_h2c_0` | Host-to-Card DMA channel 0      | `os.pwrite()` 把 DRAM image、program 和参数写入卡 |
| `/dev/xdma0_c2h_0` | Card-to-Host DMA channel 0      | `os.pread()` 从卡读回 logits、状态、DRAM 和 trace |

其中 `user` 对应控制通路，`h2c_0`/`c2h_0` 对应数据通路。
FPGA 内部的 AXI-Lite 控制寄存器和 DDR3 AXI master 必须已经连接到 XDMA bridge；
仅安装驱动、没有正确 bitstream 时，驱动可能加载但不会得到可用的 openTPU 设备。

#### 产生这些节点需要的依赖

依赖分为硬件、Linux 内核和用户态三层：

1. **硬件/bitstream**
         - openTPU FPGA bitstream，包含 PCIe endpoint、XDMA bridge、BAR0 AXI-Lite
                 控制路径和 DDR3 AXI memory path；
         - PCIe 插槽、链路训练和供电正常；
         - JTAG 或 flash 工具用于加载 bitstream。JTAG 只负责配置 FPGA，不负责创建 `/dev/xdma*` 节点。
2. **Linux 驱动构建环境**
         - Linux 主机和当前运行内核的 headers，例如 `linux-headers-$(uname -r)`；
         - `make`、`gcc` 或内核使用的 `clang/llvm`、`patch`；
         - 项目固定提交的 Xilinx `dma_ip_drivers` XDMA 源码，以及
                 `opentpu/host/pcie/xdma-otpu.patch`；
         - 可选 DKMS，用于内核升级后自动重建模块。
3. **驱动加载和设备管理**
         - 可加载的 `xdma.ko`，并且必须是项目版而不是其他平台的同名 XDMA 模块；
         - `udev` 规则 `59-otpu-xdma.rules`：按 PCI vendor/device 或 subsystem ID 将设备绑定到 `xdma`；
         - `udev` 规则 `60-otpu.rules`：为 `*_user`、`*_h2c_*`、`*_c2h_*` 设置访问权限；
         - 用户需要有访问这些节点的权限。项目规则默认设置为非 root 可访问，生产环境
                 可按主机安全策略改为专用用户组。

仓库中的 [`opentpu/host/setup_pcie.sh`](opentpu/host/setup_pcie.sh) 会安装或检查上述驱动、udev 规则和 modprobe 选项。
它固定了 XDMA 源码提交和 `otpu` 模块标签，并检查驱动是否真的绑定到 openTPU 卡，而不是只检查 `modprobe xdma` 是否成功。

首次安装或内核升级后，在板卡主机上安装匹配的 XDMA 驱动并检查设备：

```sh
sudo otpu-setup
otpu-setup --check
```

JTAG 重新加载 bitstream 后，PCIe 设备可能需要重新扫描和绑定：

```sh
otpu-setup --rescan
otpu-setup --check
```

`otpu-setup` 使用项目固定版本的 Xilinx XDMA 驱动；
不要把系统自带的、面向其他 Alveo 卡的同名 `xdma` 模块误认为是 openTPU 驱动。

如果节点没有出现，按以下顺序排查：

```sh
otpu-setup --check
lspci -nnk | grep -A3 -i '10ee:'
ls -l /dev/xdma*
lsmod | grep xdma
dmesg | grep -i -E 'xdma|pcie|openTPU'
```

常见原因是 bitstream 在 PCIe 枚举之后重新加载但没有执行 `otpu-setup --rescan`、
驱动没有为当前内核编译、系统自带 `xdma` 抢先绑定，或 udev 规则没有安装。
节点存在但 Python 无法打开时，优先检查文件权限和设备锁；
不要手工创建 `/dev/xdma*`，因为手工节点没有对应的内核 file operation，不能访问 FPGA。

先做分阶段硬件自检。默认不带 `--sim` 时访问真实卡：

```sh
otpu-selftest
otpu-diag
otpu-smi
```

`otpu-selftest` 会依次检查设备 ID、bitstream 配置、DDR 校准、DRAM scrub、寄存器、
地址/数据模式以及更高层的运行路径；某一阶段失败时停止并给出排查提示。
`otpu-diag` 用于更详细的 PCIe、bitstream、寄存器和内存诊断，`otpu-smi` 用于查看运行状态和计数器。
需要指定其他 XDMA 设备时使用对应 CLI 的 `--dev /dev/xdmaN`。

使用真实卡运行模型：

```sh
otpu-lock -- python3 tools/validate.py \
  --model lfm2 \
  --backend board \
  --against isa \
  --tokens 3 \
  "The capital of France is"
```

`otpu-lock` 保证同一块卡不会被多个进程同时访问。交互式运行使用：

```sh
otpu-lock -- otpu-chat --model lfm2 --backend board
```

单次 prompt 或纯文本输出可以使用 `otpu-chat` 的 `--prompt`、`--plain`；
真实卡的设备前缀通过 `--dev /dev/xdma0` 指定。
`backend=board` 与 `backend=board-sim` 的区别在 transport 层：
前者使用 `XdmaTransport`，后者使用 `SimTransport`，
后者不会打开 PCIe 设备，也不会访问真实 FPGA。

### 6.2 Python 到 FPGA 逻辑电路的调用栈

真实卡上的 Python 调用栈如下：

```text
otpu-chat / tools/validate.py
        |
        v
Engine(..., backend="board")
        |
        v
opentpu.host.chat.make_backend("board")
        |
        +-- XdmaTransport("/dev/xdma0")
        |       |
        |       +-- /dev/xdma0_user  -> BAR0 mmap -> AXI-Lite control registers
        |       +-- /dev/xdma0_h2c_0 -> host-to-card DRAM DMA
        |       `-- /dev/xdma0_c2h_0 -> card-to-host DRAM DMA
        |
        v
BoardBackend -> Board
        |
        +-- DRAM image / program / state read-write
        +-- IMEM program load and R8..R15 argument registers
        +-- CTRL.RUN / CTRL.LOAD / STATUS polling
        `-- counters, trace and output readback
        |
        v
PCIe + XDMA bridge
        |
        +-- BAR0 AXI-Lite -> otpu_ctrl.sv -> sequencer / slice control
        +-- AXI memory path -> LiteDRAM or native DRAM adapter
        `-- DRAM channels -> MXU, VPU, DMA, TMEM and ACT RAM
```

#### 6.2.1 Python 与 `dma_ip_drivers` 的交互（驱动相关）

Python 不直接导入或调用 `dma_ip_drivers` 的 Python/C API。
`dma_ip_drivers` 是Linux 内核态 XDMA 驱动，Python 通过 Linux 设备文件、文件描述符和内存映射间接使用它：

```text
Python 标准库
        os.open() / os.pwrite() / os.pread() / mmap.mmap()
                |
                v
Linux 文件描述符和 mmap
                |
                v
项目版 XDMA 内核模块 xdma.ko
                |
                v
PCIe + FPGA XDMA bridge + AXI-Lite / AXI memory
```

运行时的入口是 [`XdmaTransport`](opentpu/host/board.py)：

| Python 层 | 设备节点/API | 作用 |
|---|---|---|
| `XdmaTransport.__init__()`                | `os.open("/dev/xdma0_user")` | 打开 BAR0 user 接口 |
| `mmap.mmap()`、`reg_read()`、`reg_write()` | BAR0 AXI-Lite | 访问 VERSION、CAPS、ARG、CTRL、STATUS 和计数器 |
| `mem_write()`                             | `/dev/xdma0_h2c_0` + `os.pwrite()` | Host-to-Card，写入 DRAM image、program 和参数 |
| `mem_read()`                              | `/dev/xdma0_c2h_0` + `os.pread()` 或 `os.readinto()` | Card-to-Host，读回 logits、状态、DRAM 和 trace |
| `poll()`                                  | BAR0 寄存器连续读取 | 等待 `STATUS.HALTED`、`STATUS.WR_IDLE` 或加载完成 |

这些底层传输由 `Board` 封装为逻辑操作：

```text
BoardBackend.run()
        -> Board.load_program()
        -> Board.write() / Board.read()
        -> XdmaTransport.mem_write() / mem_read()
        -> os.pwrite(h2c) / os.pread(c2h)
        -> XDMA DMA engine
        -> FPGA AXI memory path
        -> DDR3

Board.start() / Board.wait()
        -> XdmaTransport.reg_write()
        -> CTRL.LOAD / CTRL.RUN
        -> XdmaTransport.poll()
        -> STATUS.LOADING / HALTED / WR_IDLE
```

对于 DRAM 数据，`Board` 不把逻辑地址直接当成单一物理地址写入。
它会按 64-byte beat 和 bitstream 的 channel map，把逻辑 DRAM 分发到两个物理通道；
读回时再重新交织。
未对齐的边缘访问会先扩展到完整 beat，避免向 FPGA DDR/ECC 路径发出不支持的 sub-beat DMA。
[`opentpu/host/setup_pcie.sh`](opentpu/host/setup_pcie.sh) 负责准备内核驱动：

```text
固定版本 dma_ip_drivers 源码
        -> 应用 xdma-otpu.patch
        -> 针对当前 Linux kernel 编译 xdma.ko
        -> 安装模块、udev 规则和 modprobe 选项
        -> modprobe / 绑定 PCI device
        -> 驱动注册 /dev/xdmaN_user、h2c_N、c2h_N
```

`opentpu/host/pcie_setup.py` 只是 `setup_pcie.sh` 的入口包装。
安装完成后，`otpu-setup --check` 检查驱动是否是项目版、是否绑定到 openTPU PCI device、
设备节点是否存在以及访问权限是否正确；
Python 模型运行只使用已经创建好的 `/dev/xdmaN_*` 节点。

Python 不直接调用 MXU、VPU 或 sequencer 的 RTL 函数；
它通过 XDMA 暴露的两个接口与逻辑电路交互：

1. **控制通路**：`XdmaTransport` 将 `/dev/xdma0_user` 映射为 BAR0，
   `reg_read()`/`reg_write()` 对应 AXI-Lite 寄存器读写。寄存器最终进入 FPGA shell
   的 `otpu_ctrl`，控制程序加载、启动、停止、状态、DDR 校准和统计计数器。
2. **数据通路**：`mem_write()`/`mem_read()` 通过 H2C/C2H XDMA 设备访问卡上 DRAM。
   `Board` 将 openTPU 的逻辑 DRAM 地址拆成 64-byte beat，按 bitstream 的 channel
   map 分配到两个物理内存通道，并在读回时重新交织。
3. **执行通路**：`BoardBackend` 将 compiler 生成的 ISA program 写入 DRAM 的程序区，
   通过寄存器请求 FPGA loader 装入 IMEM，再写入参数并置 `CTRL.RUN`。逻辑电路执行
   DMA、MXU、VPU、quantizer 和 collective；host 轮询 `STATUS.HALTED`，确认
   `STATUS.WR_IDLE` 后才读取结果，避免 FPGA 仍有 DRAM 写事务。

一次普通运行的顺序是：

```text
读取 bitstream VERSION/CAPS
    -> 得到 D / MCOLS / LANES / S / channel map
    -> DDR calibration 与必要的 DRAM ECC scrub
    -> 写入权重、prompt、KV cache 和运行时状态
    -> 写入 ISA program 与 R8..R15
    -> CTRL.LOAD，等待 STATUS.LOADING 清除
    -> CTRL.RUN，轮询 STATUS.HALTED
    -> 等待 STATUS.WR_IDLE
    -> DMA 读回 logits、token、状态和 counters
```

decode 时，`Engine` 可以在卡执行当前 program 的同时编译下一个位置的 program；
`BoardBackend` 将 `start()` 和 `wait()` 分开，减少 host 编译和卡上运行之间的空隙。
因此卡上的计算由 FPGA 逻辑电路完成，Python 主要负责 tokenizer、编译调度、DMA、
寄存器控制、logits 读取和采样。

### 6.3 真实卡与 board-sim 的区别

两者复用 `BoardBackend`、`Board` 和寄存器协议，但 transport 层不同：

| 项目 | `backend=board` | `backend=board-sim` |
|---|---|---|
| transport | `XdmaTransport` | `SimTransport` |
| 控制接口 | `/dev/xdma0_user`、BAR0、AXI-Lite     | testbench 中的寄存器脚本 |
| DRAM    | XDMA DMA 到真实 DDR3                  | Verilator 内存模型 |
| 逻辑电路 | FPGA bitstream                       | `sim/verilator/tb_board.sv` 和 RTL |
| 覆盖范围 | PCIe、XDMA、DDR 校准、host driver、RTL | host driver 协议和 RTL 行为 |

因此 `board-sim` 通过不能证明真实 PCIe、XDMA 驱动、DDR3 校准或物理时序通过；
真实板卡的结论必须至少经过 `otpu-selftest`、`otpu-diag` 和一次 `backend=board` 模型运行。

## 7. 仿真配置和环境变量

仿真配置的默认值在 [`opentpu/isasim.py`](opentpu/isasim.py) 的
`board_config()` 中，主要环境变量如下：

| 环境变量 | 作用 |
|---|---|
| `OTPU_MCOLS`       | MXU columns；必须与目标 bitstream/board model 匹配 |
| `OTPU_LANES`       | VPU/TMEM lanes |
| `OTPU_ACT_ROWS`    | ACT RAM rows |
| `OTPU_PAIR`        | 4-bit MM column reuse |
| `OTPU_DSTEP`       | DeltaNet head step |
| `OTPU_STREAM`      | stream engine |
| `OTPU_NATIVE`      | RTL memory path；`ld` 使用 LiteDRAM channel model，`1` 使用 native memory model |
| `OTPU_UARCH=board` | 使用板卡微架构参数 |
| `OTPU_AXI=1`       | 工具中启用 AXI 路径 |
| `OTPU_BOOT=1`      | 工具中启用板级 boot/load 流程 |

例如使用板级 RTL 参数运行测试：

```sh
OTPU_UARCH=board OTPU_AXI=1 OTPU_BOOT=1 OTPU_NATIVE=ld \
python3 -m pytest -q tests/test_board.py
```

不要用环境变量覆盖真实卡能力；真实 `backend=board` 会从 bitstream 的
`VERSION/CAPS` 寄存器读取 `MCOLS`、`LANES` 和功能位。

## 8. Trace 和性能分析

### 8.1 Lens

Lens 可以记录 RTL 或板级模型的执行 trace：

```sh
python3 -m opentpu.lens list
python3 -m opentpu.lens record mlp attn -o run.otpuprof
otpu-lens record --sim --workload qwen-tiny --pos 8 -o qwen.otpuprof
otpu-lens summary qwen.otpuprof
otpu-lens html qwen.otpuprof -o qwen.html
```

`--sim` 使用板级模型；不带 `--sim` 时通常记录 RTL workload。
实现和输出格式见 [`docs/lens.md`](docs/lens.md)。

### 8.2 统计数据

`rtlsim.run()` 返回的统计信息可用于分析 cycles、stall、各单元 busy 和 DRAM访问。
模型/板级运行还可通过 `otpu-smi --sim` 读取模拟寄存器：

```sh
otpu-smi --sim
```

统计用途应与功能验证分开：ISA 结果正确不代表 RTL 时序、AXI backpressure 或
板级 memory path 没有问题。

## 9. 推荐使用顺序

遇到一个新 kernel 或新模型时，建议按以下顺序缩小问题范围：

1. **ISA**：先用 `backend=isa` 确认布局、量化、kernel 和模型结果。
2. **ISA 与 RTL 对比**：用 `tools/validate.py --backend rtl --against isa`，
   只跑少量 token。
3. **板级 host flow**：用 `otpu-selftest --sim` 或 `backend=board-sim`，
   验证寄存器、loader、内存通路和主机驱动协议。
4. **真实卡**：最后使用 `backend=board`，验证 PCIe/XDMA、DDR 校准和实际带宽。

如果只是看算法结果，用 ISA；如果定位 RTL bug，用 RTL；如果验证 host/board 集成，用 board-sim。
不要把 board-sim 的通过结果解释为真实 PCIe、XDMA 或 DDR PHY 已经通过硬件验证。

## 10. 常见问题

### Verilator 未安装

RTL 和 board-sim 需要 `verilator`。ISA backend 不需要 Verilator。
安装后重新运行相同命令即可，`rtlsim.build()` 会按版本和源码内容重新选择缓存。

### RTL 运行很慢

这是预期行为。减少 `--tokens`、缩短 prompt，或先使用 ISA 验证功能。
完整模型的 RTL 运行适合少量回归和 bit-exact 对比，不适合作为日常聊天后端。

### 仿真结果与真实卡不一致

依次比较：

1. bitstream 的 `VERSION/CAPS` 与 `board_config()`；
2. `MCOLS`、`LANES`、`ACT_ROWS`、`DSTEP`、`STREAM` 等参数；
3. 权重格式和 LM head 格式；
4. DRAM image、运行参数和模型 token；
5. RTL/board model 是否覆盖了真实硬件的 AXI、LiteDRAM 或 reset 路径。

board-sim 默认不覆盖 PCIe/XDMA 物理链路、DDR PHY 和校准过程，因此这些差异必须在真实卡上单独验证。
