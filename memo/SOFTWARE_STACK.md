# openTPU 软件栈分析

## 1. 总体结构

openTPU 的软件栈是一条从 Hugging Face 模型到 FPGA 指令的完整链路：

```text
Hugging Face 模型/权重
        |
        v
opentpu.llm          Spec + Engine: 模型 image、权重/KV cache、prefill/decode 编排        
        |
        v
opentpu.kernels      attention、MLP、RoPE、RMSNorm、DeltaNet 等算子
        |
        v
opentpu.language     Triton 风格 Python DSL：Tile、dot、load、store、range
        |
        v
opentpu.compiler     内存布局、Affine 地址、寄存器分配、循环和指令生成
        |
        v
8 x 32-bit openTPU ISA
        |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
opentpu.isasim       Verilator RTL   board.py + PCIe XDMA
Python 位精确模拟器    SV 硬件模型      FPGA BoardBackend
```

核心思想是：硬件不负责复杂的动态调度，软件显式生成数据搬运和计算指令。因此，程序的执行轨迹能够直接解释性能。

## 2. 模型层：`opentpu.llm`

主要实现位于 `opentpu/llm/`，包括 `qwen3.py`、`lfm2.py`、`qwen35.py`、`gemma4.py` 和 `generate.py`。

这一层负责：

- 从 Hugging Face 的 `config.json` 解析模型规格；
- 按需读取 safetensors 权重，避免多十亿参数模型全部展开到 fp32 内存；
- 构造模型的 DRAM image，包括各层权重、norm、embedding、LM head 和 KV Cache；
- 把 attention、MLP、RoPE、RMSNorm、卷积或 DeltaNet 等模型结构映射到 openTPU kernel；
- 选择 int8、int4、FP4 或混合权重格式。

一次 decoder layer 大致会被拆成：

```text
RMSNorm -> Q/K/V projection -> RoPE -> attention -> output projection
         -> residual -> RMSNorm -> gate/up projection -> SiLU
         -> down projection -> residual
```

KV Cache 常驻设备 DRAM。K 使用 token-major 布局，V 使用转置并按 tile 组织的布局，追加数据时通过 `QST` 写入量化后的值和 scale。

## 3. Kernel 层：`opentpu.kernels`

主要文件包括：

- `opentpu/kernels/attention.py`
- `opentpu/kernels/mlp.py`
- `opentpu/kernels/lib.py`
- `opentpu/kernels/deltanet.py`
- `opentpu/kernels/gather.py`

kernel 使用 `opentpu.language` 描述，而不是直接写 NumPy 或 PyTorch。典型操作如下：

| DSL 操作 | 硬件操作 |
|---|---|
| `ol.load()`                        | DRAM 到 TMEM |
| `ol.store()`                       | TMEM 到 DRAM |
| `ol.quantize()`                    | TMEM 到 ACT RAM 的 int8 |
| `ol.dot()`                         | MXU 矩阵乘法 |
| `ol.exp2()`、`ol.max()`、`ol.sum()` | VPU 运算 |
| `ol.all_gather()`                  | slice 间数据交换 |
| `ol.kv_append()`                   | 量化写入 KV Cache |
| `ol.range()`                       | 生成硬件循环 |

attention 使用分块和在线 softmax，接近 FlashAttention 的数据流。编译器还会做操作融合，例如：

```text
exp2(a - b)              -> EXP2SUB
quantize(x * scale)      -> QACT CSCALE
sum(x * x, axis=1)       -> RSSQ
max(dot(...), axis=1)    -> MM RMAX
acc * alpha + dot(...)   -> MM ACC + ASCALE
```

## 4. DSL 与编译器

核心文件是 `opentpu/language.py` 和 `opentpu/compiler.py`。

编译器追踪 `@ol.jit` kernel，并把它降为 ISA 程序。主要抽象包括：

| 抽象 | 实际位置 | 作用 |
|---|---|---|
| `Tensor`     | DRAM 中的 fp32          | 普通输入输出 |
| `QTensor`    | DRAM 中的量化权重和 scale | MXU streamed operand |
| `Tile`       | TMEM 中的 fp32          | VPU 和 MXU 输出 |
| `Stationary` | ACT RAM 中的 int8       | MXU stationary operand |
| `Bcast`      | Tile 的广播视图          | VPU 行列广播 |

在 openTPU 中，layout transition 就是一条硬件指令：

```text
DRAM -> TMEM                 LD
TMEM -> ACT RAM              QACT
ACT RAM + DRAM weight        MM
TMEM -> DRAM                 ST
TMEM -> DRAM quantized       QST
slice 间复制                  GATHER
```

地址使用 `Affine` 表达，可包含常量、硬件循环变量和运行时参数。编译器负责：

- 硬件循环和地址寄存器分配；
- token、position 等运行时参数的寄存器映射；
- TMEM 和 ACT RAM 生命周期管理；
- TMEM bank 冲突检查；
- 多 slice 分片和 gather；
- 对齐、shape、stride 和容量检查。

## 5. Runtime 与 DRAM image

`opentpu/runtime.py` 把 Python/Numpy 参数转换成设备内存映像。

权重的典型流程是：

```text
fp32 权重
  -> 按 D=128 分块
  -> int8/int4/FP4 量化
  -> 生成数据区和 scale 区
  -> 分配 DRAM 地址
  -> 创建 QTensor descriptor
```

权重可以按行、按列分片，也可以复制到所有 slice。MLP 通常在 slice 间分片 gate/up 或 down 权重，再用 `GATHER` 交换中间结果。

## 6. ISA：软件和硬件的契约

ISA 定义在 `opentpu/isa.py` 和 `docs/isa.md`。每条指令由 8 个 32-bit word 组成，包含 opcode、寄存器、flags、地址、shape 和 stride。

### 6.1 `isa.py` 与 `isasim.py` 的关系

这两个模块不是两个不同的 ISA，也不是两个并列的 simulator，而是同一套 ISA 的两个软件层次：

```text
opentpu/isa.py
        ISA 数据结构、opcode、指令构造、assemble/decode
                                |
                                v
opentpu/isasim.py
        Config、Machine、Slice：读取并执行 ISA program
                                |
                                v
        寄存器 / PC / IMEM / DRAM / TMEM / ACT RAM / DMA / scoreboard
```

`opentpu/isa.py` 主要回答“指令长什么样、如何生成和编码”，推测会根据这个文件将模型转成指令：

- 定义指令和操作数的数据结构、opcode、flags 以及地址/shape 字段；
- 提供 `Instr`、opcode 构造函数等指令抽象；编译器的 `Program` 容器定义在
        `opentpu/compiler.py`；
- 提供 `assemble()`，把编译器生成的指令转换成硬件使用的 32-bit word；
- 提供反汇编、解码和调试辅助，使同一条指令可以被检查和打印；
- 提供 `I.ld()`、`I.mm()`、`I.vop()`、`I.halt()` 等面向编译器和测试的构造函数。

`opentpu/isasim.py` 主要回答“指令如何改变设备状态”，推测用于模拟硬件指令：

- `Config` 描述 slice、MXU、TMEM、ACT RAM、IMEM 和 DRAM 容量；
- `Machine` 管理一个或多个 slice，执行程序并返回最终内存和统计结果；
- `Slice` 保存寄存器、PC、IMEM、DRAM、TMEM、ACT RAM 和执行单元状态；
- 取指后使用 `isa.py` 的指令格式和 opcode 解释指令，再模拟 DMA、MXU、VPU、
  quantizer、scoreboard、barrier、`WAITW` 和多 slice collective；
- 复现硬件的 fp32 rounding、flush-to-zero、固定累加顺序和特殊函数近似。

典型调用关系是：

```text
@ol.jit kernel
        -> compiler
        -> compiler.Program / isa.Instr
        -> isa.assemble()
        -> 8 x 32-bit ISA words
        -> isasim.Machine(...).run()
```

这里 `isa.py` 负责生成和编码程序，`isasim.py` 负责执行程序；前者不运行模型，后者通常也不重新编译 kernel。
RTL 和真实 FPGA 则复用由 `isa.py` 编码出的同一份 program 和相同的 DRAM image：
RTL 由 SystemVerilog sequencer 执行，FPGA 由真实 sequencer 执行，`isasim.py` 作为 Python 参考执行器执行。

因此三者的职责可以概括为：

| 层 | 使用模块 | 作用 |
|---|---|---|
| 指令定义/编码    | `opentpu/isa.py`     | 规定 ISA 程序的结构和二进制表示 |
| Python 执行模型 | `opentpu/isasim.py`  | 在 CPU 上执行 ISA，模拟设备状态和数值语义 |
| 硬件执行        | RTL / FPGA sequencer | 执行同一份 ISA word，验证硬件实现是否一致 |

主要指令类别：

```text
NOP / HALT / LI / ADDI / LOOP / BAR
LD / ST / WAITW / RLD
MM / QACT / QST
VOP
GATHER
```

其中：

- `MM` 负责 ACT RAM 与 DRAM 量化权重的矩阵乘法，支持 int8、int4、FP4、累加、row max 和 rescale；
- `VOP` 负责 fp32 向量操作、exp2、reciprocal、rsqrt、log2、行归约和 outer product；
- `QACT/QST` 负责 fp32 与量化数据之间的转换；
- `WAITW`、`RLD` 等指令支持主机协作、运行时地址和设备生成的地址。

## 7. ISA simulator

`opentpu/isasim.py` 和 `opentpu/fp32.py` 实现位精确 ISA simulator。它不仅做功能模拟，还模拟硬件语义：

- fp32 rounding 和 flush-to-zero；
- exp2、reciprocal、rsqrt、log2 的硬件近似；
- MXU 固定累加顺序；
- quantizer 行为；
- TMEM、ACT RAM、DRAM 和 DMA；
- scoreboard 依赖、多 slice collective 和 `WAITW`。

因此 ISA simulator 同时承担快速功能模型、设备侧 golden、RTL 对照基准和板卡验证基准的角色。

## 8. RTL 与 Verilator (推测是 backend=rtl 的描述)

RTL 顶层是 `rtl/top/otpu_top.sv`，下面连接多个 `otpu_slice` 和 `otpu_coll`。一个 slice 包含：

```text
sequencer + instruction memory
DMA
TMEM
ACT RAM
MXU
VPU
quantizer
DRAM adapter
```

`sim/verilator/tb_top.sv` 负责装载程序、DRAM image 和运行参数，等待所有 slice halt，并 dump 最终状态。

RTL 与 ISA simulator 使用同一份 ISA 程序、同一份 DRAM 初始镜像和同一份配置，然后比较 DRAM、TMEM、输出、logits、cycle 和错误状态。

RTL 仿真的内存路径可以切换：

```text
AXI=0：行为级固定延迟 DRAM
AXI=1：native memory adapter + 延迟模型
AXI=2：接近板卡 LiteDRAM channel 的模型
```

## 9. FPGA 板卡和 Host

板卡顶层为 `rtl/boards/ypcb-00338/otpu_board.sv`。

板卡连接关系：

```text
Host --PCIe--> Xilinx XDMA
                 |-- AXI-Lite -> 控制寄存器
                 |-- AXI master -> DDR3 channel 0/1
                                      -> openTPU RTL
```

主机运行一次程序时通常会：

1. 获取设备独占锁；
2. 读取 VERSION、CAPS 和 BUILD_ID；
3. 从 bitstream 推断 D、MCOLS、LANES 等配置；
4. 把程序和数据写入板卡 DRAM；
5. 写入 `ARG0..ARG7`；
6. 写 RUN 控制寄存器；
7. 等待 HALTED；
8. 读取输出、logits、计数器和 trace。

resident decode 模式下，同一个程序可以通过参数寄存器接收 token、position 和 KV Cache 偏移，减少每个 token 的程序上传开销。

## 10. 一次 decode 的端到端流程

```text
读取 tokenizer 和 safetensors
  -> 解析 Spec
  -> 构造模型 image 和 KV Cache
  -> 编译 decode kernel
  -> 上传权重、KV Cache 和 program
  -> 设置 token/position 参数
  -> 启动 ISA program
  -> attention、MLP、LM head 在设备上运行
  -> 读取或流式读取 logits
  -> 主机完成 sampling
  -> 追加 token 和 KV Cache
  -> 执行下一个 token
```

设备负责主要的权重访问、attention、MLP 和 KV Cache 更新；主机主要负责 tokenizer、chat template、sampling、UI 和运行控制。

## 11. 验证体系

项目的验证链分为三层：

### 算子和硬件单元

测试 fp32、量化、VPU、MXU、DMA、地址生成、loop、memory hazard 和 collective。

### Kernel 和 RTL

`tests/test_rtl.py` 把同一个 kernel 编译后分别交给 ISA simulator 和 RTL simulator，并逐字节比较 DRAM 与 TMEM。

### 模型和板卡

`tools/validate.py` 使用 Hugging Face CPU golden，并加入 openTPU 相同的权重量化和 activation quantization，再与 ISA simulator、RTL 或 FPGA board 比较 token 和 logits。

`tools/qual/qual.sh` 进一步覆盖 self-test、内存测试、带宽、decode、长上下文、WAITW、ECC、PCIe DMA 和稳定性测试。

## 12. RTL 和软件栈哪个更复杂？

### 简短结论

如果问的是**实现规模、模块数量和时序/接口细节**，RTL 更复杂；如果问的是**跨层抽象、模型适配和系统行为**，软件栈更复杂。
综合整个工程，软件栈的系统性复杂度略高，RTL 的局部实现难度和验证成本更高。

### RTL 的复杂性

RTL 的难点集中在硬件约束和精确时序：

- sequencer、DMA、MXU、VPU、TMEM、ACT RAM、quantizer 之间存在大量握手和依赖；
- scoreboard 必须正确处理 RAW、WAR、WAW 和不同执行单元的完成时间；
- DRAM、PCIe、LiteDRAM、AXI/native memory path 都有不同的协议和 backpressure；
- TMEM bank、BRAM、DSP、FIFO 和时钟频率直接影响布局布线；
- fp32、量化、累加顺序和异常边界必须与软件模型逐 bit 一致；
- 任何一个周期、握手或复位问题都可能在仿真、综合或板卡上表现不同。

因此 RTL 的复杂性更像是“时间、硬件资源和协议的三维约束”。它对局部正确性要求极高，修改一个数据通路可能影响时序、面积和整个系统的可运行性。

### 软件栈的复杂性

软件栈的难点在于跨越多个抽象层：

- 模型结构要映射到固定的 MXU/VPU/内存能力；
- 编译器要同时管理 layout、地址、循环、寄存器、生命周期和 bank conflict；
- 权重、activation、KV Cache 需要匹配硬件量化和存储布局；
- 不同模型族有不同的 attention、卷积、DeltaNet、MoE 和 embedding 规则；
- simulator、RTL 和板卡必须保持同一 ISA 与数值语义；
- host 还要处理 PCIe、DMA、设备锁、流式 logits、性能计数器和异常恢复。

软件栈的复杂性更像是“语义、布局和后端一致性的多层约束”。它不仅要算对，还要让模型适配硬件、让多个后端表现一致，并且能在真实板卡上高效运行。

### 对比表

| 维度 | RTL | 软件栈 |
|---|---|---|
| 主要难点   | 时序、握手、资源、协议、复位               | 抽象、模型映射、布局、编译和后端一致性 |
| 复杂性类型 | 局部深、周期敏感                         | 跨层广、语义耦合强 |
| 调试成本   | 很高，常需 Verilator、波形、综合和板卡验证 | 很高，常需 ISA/RTL/CPU golden/板卡交叉验证 |
| 变化影响   | 改动可能影响 fmax、面积和协议             | 改动可能影响多个模型、权重格式和后端 |
| 代码可读性 | 低层细节多，理解成本高                    | 抽象层多，但 Python/DSL 相对易读 |
| 运行时风险 | 周期、死锁、数据损坏、时序不收敛            | 编译失败、地址错误、数值偏差、性能退化 |

### 最终判断

这个工程中，**RTL 是最难做对的部分，软件栈是最难做全的部分**。

如果只看硬件核心 datapath，RTL 的局部复杂度更高；如果把模型支持、量化、编译器、模拟器、host、板卡后端和验证工具全部算上，软件栈的总体系统复杂度略高。
两者不是替代关系，而是通过 ISA 强耦合：ISA 越简单，编译器和 kernel 需要承担的布局工作越多；硬件越强，RTL 的时序和资源压力越大。

## 13. 推荐阅读顺序

1. `README.md`
2. `docs/isa.md`
3. `docs/compiler.md`
4. `opentpu/isa.py`
5. `opentpu/isasim.py`
6. `opentpu/runtime.py`
7. `opentpu/kernels/attention.py`
8. `opentpu/llm/qwen3.py`
9. `rtl/top/otpu_top.sv`
10. `opentpu/host/board.py`
11. `tools/validate.py`