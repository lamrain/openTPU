# openTPU RTL 层级

> **LOC 统计口径**：按 module 定义范围内的有效物理代码行计数，包含 `module` 和
> `endmodule` 行；不计空行、注释和预处理指令，并排除 `SYNTHESIS` 未定义时启用的
> 仿真代码。技术替代模块按 YPCB-00338 构建所选文件计数（Xilinx7 DSP、generic
> RAM）；Vivado 生成 IP 没有仓库内手写 RTL 源文件，标为“生成 IP，未统计”。

本文从 [`rtl/top/otpu_top.sv`](rtl/top/otpu_top.sv) 出发，梳理核心 RTL 的模块层级和对应源文件。
这里的“层级”指 SystemVerilog module 的实例化关系；同一个 `.sv` 文件可以定义多个可复用的 module。

## 1. 总体树

```text
otpu_top (157 LOC)                                        rtl/top/otpu_top.sv
├── g_slice[S]                                            每个 slice 一份
│   ├── DRAM backend                                      AXI 参数选择其一
│   │   ├── otpu_dram (67 LOC)                            rtl/mem/otpu_dram.sv             AXI=0
│   │   ├── otpu_native_dram (508 LOC)                    rtl/mem/otpu_native_dram.sv      AXI=1/2
│   │   ├── otpu_native_mem (296 LOC)                     sim/verilator/otpu_native_mem.sv AXI=1
│   │   └── otpu_ldc_mem (340 LOC)                        sim/verilator/otpu_ldc_mem.sv    AXI=2
│   └── otpu_slice (731 LOC)                              rtl/top/otpu_slice.sv
│       ├── otpu_seq (616 LOC)                            rtl/seq/otpu_seq.sv
│       ├── otpu_tmem (547 LOC)                           rtl/mem/otpu_tmem.sv
│       ├── otpu_actram (134 LOC)                         rtl/mem/otpu_actram.sv
│       ├── otpu_dma (728 LOC)                            rtl/dma/otpu_dma.sv
│       ├── otpu_mxu (1090 LOC)                           rtl/mxu/otpu_mxu.sv
│       │   ├── otpu_mx_lane (46 LOC)                     rtl/mxu/otpu_mxu.sv
│       │   ├── otpu_ram_sdp (13 LOC)                     rtl/mxu/otpu_mxu.sv
│       │   ├── otpu_skew (17 LOC)                        rtl/mxu/otpu_mxu.sv
│       │   ├── otpu_delay (23 LOC)                       rtl/vpu/otpu_fpipe.sv
│       │   └── otpu_pe (LOC: 36) / otpu_mxce (LOC: 61)   rtl/tech/xilinx7/otpu_tech_dsp.sv
|       | 以下包含了 vpu 文件夹所有文件
│       ├── otpu_quant (584 LOC)                          rtl/vpu/otpu_quant.sv
│       │   ├── otpu_qscale (61 LOC)                      rtl/vpu/otpu_quant.sv
│       │   ├── otpu_qdly (19 LOC)                        rtl/vpu/otpu_quant.sv
│       │   └── otpu_fmul (LOC: 17) / fmadd (LOC: 15)     rtl/vpu/otpu_fpipe.sv
│       └── otpu_vpu (1193 LOC)                           rtl/vpu/otpu_vpu.sv
│           ├── otpu_vtree (160 LOC)                      rtl/vpu/otpu_vtree.sv
│           │   └── otpu_fadd (LOC: 21) / fadd2 (LOC: 25) rtl/vpu/otpu_fpipe.sv
│           ├── otpu_se_comp (413 LOC)                    rtl/vpu/otpu_se_comp.sv
│           │   └── otpu_fmadd (15 LOC)                   rtl/vpu/otpu_fpipe.sv
│           ├── otpu_se_tail (263 LOC)                    rtl/vpu/otpu_se_tail.sv
│           │   ├── otpu_fmadd (15 LOC)                   rtl/vpu/otpu_fpipe.sv
│           │   ├── otpu_fmma (16 LOC)                    rtl/vpu/otpu_fpipe.sv
│           │   ├── otpu_fmul (17 LOC)                    rtl/vpu/otpu_fpipe.sv
│           │   └── otpu_fadd (21 LOC)                    rtl/vpu/otpu_fpipe.sv
│           ├── otpu_fmma (LOC: 16) / fmadd (LOC: 15)     rtl/vpu/otpu_fpipe.sv
│           └── otpu_fmul (LOC: 17) / fadd (LOC: 21)      rtl/vpu/otpu_fpipe.sv
└── collective backend                                    参数 COLL_GATHER 选择其一
    ├── otpu_coll (LOC: 159)                              rtl/top/otpu_coll.sv       COLL_GATHER=1
    └── otpu_coll_bar (LOC: 45)                           rtl/top/otpu_coll.sv       COLL_GATHER=0
```

`S` 个 slice 共享 top 级 collective 接口，但每个 slice 拥有自己的
sequencer、DMA、TMEM、ACT RAM、MXU 和 VPU。
`otpu_top` 只负责把这些单元连接起来，并选择 DRAM/collective 实现；
具体指令执行由 `otpu_seq` 和各执行单元完成。

## 2. 根模块：`otpu_top`（LOC: 157）

文件：[rtl/top/otpu_top.sv](rtl/top/otpu_top.sv)

`otpu_top` 是核心仿真的顶层，不是 PCIe 板级 shell。它的主要职责是：

- 生成 `S` 个 `otpu_slice`；
- 为每个 slice 选择行为级 DRAM 或 native/AXI memory backend；
- 连接 slice 的 collective 请求和返回数据；
- 汇总 `all_halted`、`any_error` 和每个 slice 的 `icount`；
- 向 testbench 提供程序加载、运行参数、DRAM 和状态接口。

### DRAM 条件分支（仅 otpu_top, 推测是仿真用）

| 参数 | 文件 | 用途 |
|---|---|---|
| `AXI=0` | `rtl/mem/otpu_dram.sv`                                            | 固定延迟的行为级 DRAM 模型 |
| `AXI=1` | `rtl/mem/otpu_native_dram.sv`、`sim/verilator/otpu_native_mem.sv` | native memory adapter 加延迟模型 |
| `AXI=2` | `rtl/mem/otpu_native_dram.sv`、`sim/verilator/otpu_ldc_mem.sv`    | 接近 LiteDRAM channel 的仿真路径 |

`otpu_native_dram`、`otpu_native_mem` 和 `otpu_ldc_mem` 都是 `otpu_top` 的同级实例：
前者是 openTPU memory request 与 native memory channel 之间的 adapter，
后两者按 `AXI` 参数二选一，提供仿真的 memory backend。
它们通过 native channel 信号连接；`otpu_native_dram` 本身不是 DDR controller。

### Collective 条件分支

- `COLL_GATHER=1`：实例化 `otpu_coll`，支持 slice 间 gather/read/write。
- `COLL_GATHER=0`：实例化 `otpu_coll_bar`，只保留 barrier 风格路径，通常用于单 slice 或板卡受限配置。

## 3. Slice 层：`otpu_slice`（LOC: 731）

文件：[rtl/top/otpu_slice.sv](rtl/top/otpu_slice.sv)

`otpu_slice` 是一个 openTPU 计算 slice 的集成模块，连接 sequencer、存储器和执行单元：

| 子模块 | 文件 | 责任 |
|---|---|---|
| `otpu_seq`    | `rtl/seq/otpu_seq.sv`    | 取指、解码、循环、scoreboard、unit issue、barrier 和等待逻辑 |
| `otpu_tmem`   | `rtl/mem/otpu_tmem.sv`   | TMEM bank、读写端口、bank conflict/replay 和多端口访问 |
| `otpu_actram` | `rtl/mem/otpu_actram.sv` | ACT RAM 中 stationary int8 数据和 scale 的存取 |
| `otpu_dma`    | `rtl/dma/otpu_dma.sv`    | DRAM 与 TMEM/ACT RAM 之间的 DMA、LD/ST/WAITW 相关操作 |
| `otpu_mxu`    | `rtl/mxu/otpu_mxu.sv`    | 矩阵乘、累加、ASCALE、RMAX 和权重流式读取 |
| `otpu_quant`  | `rtl/vpu/otpu_quant.sv`  | QACT/QST、scale、int8/int4/FP4 等量化路径 |
| `otpu_vpu`    | `rtl/vpu/otpu_vpu.sv`    | 向量运算、特殊函数、归约、attention 和逐元素操作 |

slice 对外连接三类接口：

```text
otpu_seq
   ├── issue -> DMA / QUANT / VPU / MXU
   ├── scoreboard <- 各单元 ready/done 和 memory footprint
   ├── TMEM / ACT RAM / DRAM 请求
   └── collective request -> otpu_top 的 collective unit
```

## 4. Sequencer、DMA 和存储层

### `otpu_seq`（LOC: 616）

文件：[rtl/seq/otpu_seq.sv](rtl/seq/otpu_seq.sv)

负责把 IMEM 中的 ISA program 转换为各执行单元的 command。它管理：

- PC、循环和程序结束；
- `LD`、`ST`、`MM`、`QACT`、`QST`、`VOP`、`GATHER` 等指令分发；
- 单元之间的依赖和 memory footprint scoreboard；
- `WAITW`、barrier、错误和 halt；
- 运行时寄存器参数。

Sequencer 使用的技术 RAM wrapper 来自技术映射文件，具体见第 8 节。

### `otpu_dma`（LOC: 728）

文件：[rtl/dma/otpu_dma.sv](rtl/dma/otpu_dma.sv)

处理 DRAM 与片上存储之间的数据搬运。它向 top 级 DRAM backend 发出 A/B/SW
请求，向 TMEM 或 ACT RAM 发出读写，并处理地址、长度、stride、对齐和等待条件。

### `otpu_tmem`（LOC: 547；helper `otpu_pump_bank`: LOC 50）

文件：[rtl/mem/otpu_tmem.sv](rtl/mem/otpu_tmem.sv)

TMEM 是 VPU/MXU 共享的 fp32 tile memory。
该文件还包含与具体实现相关的 helper module，例如：

- `otpu_pump_bank`：pumped/sub-bank 访问路径；
  双频 TMEM 时钟由 `clk_f` 和相关时序约束支持，但 `otpu_clk2x` 不是一个 RTL module；
- `otpu_ram_1r1w`、`otpu_ram_1rw1r`：RAM wrapper。

实际启用哪些 helper 由 `TMEM_IMPL`、`TMEM_TDP`、`TMEM_PMP`、`TMEM_RQ` 等参数决定。

### `otpu_actram`（LOC: 134）

文件：[rtl/mem/otpu_actram.sv](rtl/mem/otpu_actram.sv)

保存 MXU 的 stationary activation 和对应 scale。
不同 FPGA/ASIC 技术映射下，底层 RAM wrapper 可来自 generic、ASIC 或 Xilinx 技术文件。

## 5. MXU 层：`otpu_mxu`（LOC: 1090）

文件：[rtl/mxu/otpu_mxu.sv](rtl/mxu/otpu_mxu.sv)

`otpu_mxu` 实现矩阵乘和相关的流水线数据通路：

```text
DRAM weight stream + ACT RAM stationary data
        -> dot/product pipeline
        -> fp32 conversion and scale
        -> partial accumulation
        -> ACC / ASCALE / RMAX
        -> TMEM result
```

同一文件中定义的辅助 module 包括：

- `otpu_mx_lane`（LOC: 46）：MXU lane/column 的乘加数据路径；
- `otpu_ram_sdp`（LOC: 13）：simple-dual-port RAM wrapper；
- `otpu_skew`（LOC: 17）   ：lane/column 数据 skew 和延迟；
- `otpu_delay`（LOC: 23）  ：流水线延迟。

MXU 还复用 `rtl/vpu/otpu_fpipe.sv` 中的浮点流水线 module，以及技术 DSP 文件中的 `otpu_pe`、`otpu_mxce`。
`MXU_IMPL` 和 `CL` 参数决定使用 adder tree 还是 DSP cascade 等实现。

## 6. VPU 和量化层

### `otpu_vpu`（LOC: 1193）

文件：[rtl/vpu/otpu_vpu.sv](rtl/vpu/otpu_vpu.sv)

VPU 是 slice 的 fp32 向量执行单元，负责 VOP、向量 load/store、广播、归约和特殊函数流水线。它连接以下子模块：

- `otpu_vtree`（LOC: 160）  ：[rtl/vpu/otpu_vtree.sv](rtl/vpu/otpu_vtree.sv)，向量树形归约；
- `otpu_se_comp`（LOC: 413）：[rtl/vpu/otpu_se_comp.sv](rtl/vpu/otpu_se_comp.sv)，复合特殊函数的控制/计算路径；
- `otpu_se_tail`（LOC: 263）：[rtl/vpu/otpu_se_tail.sv](rtl/vpu/otpu_se_tail.sv)，特殊函数尾段、DeltaNet step 等路径；
- `rtl/vpu/otpu_fpipe.sv` 中的 `otpu_fadd`（LOC: 21）、`otpu_fadd2`（LOC: 25）、`otpu_fmul`（LOC: 17）、
  `otpu_fmadd`（LOC: 15）、`otpu_fmma`（LOC: 16）、`otpu_delay`（LOC: 23）；
- `rtl/vpu/otpu_fp.sv` 中的 fp32 package/function 定义。

`otpu_se_comp` 和 `otpu_se_tail` 是 `otpu_vpu` 的可选特殊函数路径，
是否启用由 `HAS_SE`、`HAS_SS`、模型/bitstream 参数决定。

### `otpu_quant`（LOC: 584）

文件：[rtl/vpu/otpu_quant.sv](rtl/vpu/otpu_quant.sv)

实现 fp32 到量化格式以及量化格式到 fp32 的转换，主要 helper 是：

- `otpu_qdly`（LOC: 19）  ：量化流水线延迟；
- `otpu_qscale`（LOC: 61）：scale 计算和应用。

## 7. Collective 层：`otpu_coll`（LOC: 159）

文件：[rtl/top/otpu_coll.sv](rtl/top/otpu_coll.sv)

在 `rtl/top/otpu_top.sv` 中，`COLL_GATHER=1` 时只直接实例化 `otpu_coll`，
`COLL_GATHER=0` 时只直接实例化 `otpu_coll_bar`。

板级 shell 的 `otpu_coll_hub` 和 `otpu_coll_port` 层级为：

```text
otpu_board
├── otpu_coll_hub (LOC: 88)
└── otpu_coll_port[S] (LOC: 74 each)
```

`otpu_coll_hub` 负责多个 slice 的请求仲裁、读数据广播、写数据路由和 gather 协调；
`otpu_coll_port` 把 collective 请求接到每个 slice 的 TMEM/接口。
它们的父模块是 `rtl/shell/otpu_board.sv`，不是 `otpu_coll`。

当 `COLL_GATHER=0` 时只使用同一文件中的 `otpu_coll_bar`，
用于 barrier-only 配置，不实例化 hub/port gather 数据路径。

## 8. 技术映射和可替换实现

核心 wrapper module 不一定直接绑定某一种 FPGA primitive。构建目标会选择不同的技术文件：

| 文件 | 主要 module | 用途 |
|---|---|---|
| `rtl/tech/generic/otpu_tech_ram.sv`  | `otpu_ram_1r1w`、`otpu_ram_1rw`、`otpu_ram_1rw1r` | 通用仿真/综合 RAM |
| `rtl/tech/asic/otpu_tech_ram.sv`     | 同名 RAM wrapper                                  | ASIC memory macro/综合映射 |
| `rtl/tech/generic/otpu_tech_dsp.sv`  | `otpu_pe`、`otpu_mxce`                            | 通用乘法器/DSP model |
| `rtl/tech/xilinx7/otpu_tech_dsp.sv`  | `otpu_pe`、`otpu_mxce`                            | 7-series Xilinx DSP48 映射 |
| `rtl/tech/xilinxus/otpu_tech_dsp.sv` | `otpu_pe`、`otpu_mxce`                            | UltraScale/UltraScale+ DSP 映射 |

这些技术文件通常不能同时编译，因为它们定义了同名 module。具体 Makefile/Vivado 目标决定采用哪一套。

## 9. FPGA 层级

板卡顶层 `otpu_fpga_top_ld`、`otpu_native_sys` 等位于 `rtl/boards/ypcb-00338/`；
BCU1525 还有对应的 `rtl/boards/bcu1525/` 路径。它们负责 PCIe/XDMA、时钟、复位、
DDR/AXI shell 和 `otpu_board` 集成，不应与 core `otpu_top` 的层级混为一谈。

### 9.1 FPGA 板顶层层级

以 YPCB-00338 的 LiteDRAM 构建为例，板级层次可以分成三部分：

```text
otpu_fpga_top_ld (LOC: 238)                                  rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv
├── otpu_bd_wrapper (generated IP; LOC N/A)                  Vivado block design wrapper
├── otpu_litedram (generated IP; LOC N/A)                    LiteDRAM DDR3 controller
├── otpu_native_sys (LOC: 182)                               rtl/boards/ypcb-00338/otpu_native_sys.sv
│   ├── otpu_dma_split (LOC: 111)                            rtl/shell/otpu_axi_split2.sv
│   ├── otpu_mem_ch[2] (LOC: 440 each)                       rtl/shell/otpu_mem_ch.sv
│   └── otpu_board (LOC: 446)                                rtl/shell/otpu_board.sv
│       ├── otpu_axil_iso (LOC: 87)                          rtl/shell/otpu_ctrl.sv
│       ├── otpu_ctrl (LOC: 264)                             rtl/shell/otpu_ctrl.sv
│       ├── otpu_trace (LOC: 217)                            rtl/shell/otpu_trace.sv
│       ├── otpu_slice[S] (LOC: 731 each)                    rtl/top/otpu_slice.sv
│       └── collective: otpu_coll (LOC: 159) / bar (LOC: 45) rtl/top/otpu_coll.sv
└── board I/O                                                clock / reset / I2C / GPIO / transceiver
```

`otpu_fpga_top_ld.sv` 还包含板级时钟缓冲、I/O buffer、生成的 Vivado block design 和 LiteDRAM 连接。
`otpu_native_sys.sv` 是板级 native memory/system wrapper，
直接实例化 `otpu_board`、`otpu_dma_split` 和两个 `otpu_mem_ch`，
负责把 XDMA/board shell 的 native channel 接到实际的 DDR/AXI 系统。

具体的 `otpu_bd_wrapper` 和 `otpu_litedram` 端口来自 Vivado 生成文件，非手写的 RTL。
真实板卡路径不经过 `otpu_top`；`otpu_top` 是仿真/核心集成顶层，而板级 shell 直接复用 `otpu_slice`。

### 9.1.1 `otpu_bd_wrapper`：PCIe/XDMA 与 FPGA 基础设施边界

`otpu_bd_wrapper` 不是 openTPU 的计算 RTL module，而是 Vivado Block Design 生成的 HDL wrapper。
它把多个 Xilinx/板卡 IP 核封装成 `otpu_fpga_top_ld.sv` 可以连接的端口。其主要职责是：

```text
PCIe edge / XDMA IP
    ├── AXI-Lite control master  -> otpu_fpga_top_ld
    ├── AXI memory master        -> otpu_native_sys / otpu_dma_split
    ├── XDMA clock/reset         -> board top clock/reset
    └── PCIe link/status         -> board top status and LEDs

otpu_litedram IP
    ├── DDR3 channel 0
    └── DDR3 channel 1
```

在这条板级路径中，`otpu_bd_wrapper` 通常包含或连接：

- PCIe endpoint 和 Xilinx XDMA bridge；
- BAR0 对应的 AXI-Lite 路径，最终到 `otpu_board` 的控制寄存器链；
- XDMA 的 host-to-card/card-to-host AXI memory master，接到 `otpu_native_sys` 的 `otpu_dma_split`；
- XDMA、控制 AXI、core clock 和 LiteDRAM controller 所需的时钟与复位；
- PCIe link、DMA 和 block-design 状态信号。

因此可以把它理解为 **PCIe/XDMA IP 与 openTPU 手写 RTL 之间的集成边界**，
但不能把 `otpu_bd_wrapper` 本身当作一个可阅读的 openTPU RTL datapath。
它的内部层级由 Vivado `.bd`、IP 配置和生成 HDL 决定，通常不在 `rtl/` 中维护。

`otpu_fpga_top_ld.sv` 的职责是把生成 wrapper、LiteDRAM、板级 I/O 和 `otpu_native_sys` 接起来；
`otpu_native_sys.sv` 再把 XDMA memory path 分到两个 `otpu_mem_ch`，并把控制/执行路径接到 `otpu_board`。
相应的 Vivado 工程文件位于：

- `boards/ypcb-00338/vivado/create_project.tcl`：
  添加 RTL/IP source、设置 FPGA top 和生成工程；
- `boards/ypcb-00338/vivado/bd_native.tcl`：
  创建 PCIe/XDMA、AXI、时钟/复位和 LiteDRAM Block Design 连接；
- `boards/ypcb-00338/litedram/`：
  LiteDRAM 生成的 controller HDL、配置和约束。

离线 lint 使用的 `boards/ypcb-00338/sim/otpu_bd_ld_wrapper_stub.sv` 
只是生成 wrapper 的行为 stub，用来检查 `otpu_fpga_top_ld.sv` 的端口连接；
它不等价于真实 PCIe/XDMA IP，也不能验证 PCIe link、DMA engine 或 DDR3 PHY。

### 9.2 寄存器层级和配置路径

真实板卡的主机可见配置寄存器不在 `otpu_top`、`otpu_slice` 或 MXU/VPU 内，而在 `otpu_ctrl`：

```text
Host Python / Xilinx XDMA
  -> BAR0 AXI-Lite
  -> otpu_board
    -> otpu_axil_iso
      -> otpu_ctrl
      -> register bank
        ├── ID / VERSION / REGMAP / CAPS / CAPS2 / BUILD_ID
        ├── CTRL / STATUS
        ├── PROG_ADDR / PROG_N
        ├── ARG0..ARG7
        ├── CYCLES / ICOUNT / memory counters
        ├── TRACE_CTRL / TRACE_COUNT / TRACE_DROP
        └── temperature / snapshot / memory-calibration status
```

寄存器地址和 bit mask 的软件镜像在 [`opentpu/host/regs.py`](opentpu/host/regs.py)。
`otpu_axil_iso` 和硬件寄存器的实现都位于 [`rtl/shell/otpu_ctrl.sv`](rtl/shell/otpu_ctrl.sv)，其中：

- `otpu_axil_iso` 隔离 XDMA/AXI-Lite master reset 与 core/register slave reset，
  保证 reset 期间已经进入的读写事务能够完成或被丢弃，不把半个 AXI-Lite transaction 传给寄存器 bank；
- `otpu_ctrl` 执行寄存器地址解码、状态汇总和控制信号生成：
  - `VERSION`、`CAPS`、`BUILD_ID` 描述当前 bitstream 的能力和构建信息；
  - `CTRL` 接收 reset、load、run、stop 等主机控制；
  - `PROG_ADDR`、`PROG_N` 触发程序从 DRAM 装载到各 slice 的 IMEM；
  - `ARG0..ARG7` 最终连接 core 的运行参数 `rinit`，对应 ISA 使用的运行时寄存器；
  - `STATUS`、`CYCLES`、`ICOUNT` 和错误位收集 `otpu_board` 下各 `otpu_slice` 的运行状态；
  - trace、ECC、DDR calibration 和温度等属于 shell/板级监控，不属于 core ISA 执行单元。

因此一次真实板卡运行的控制边界是：

```text
XdmaTransport.reg_write/read()
  -> /dev/xdmaN_user
  -> BAR0 / AXI-Lite
  -> otpu_axil_iso
  -> otpu_ctrl register bank
  -> otpu_board control wiring
  -> otpu_slice[S] ld_start / rinit / reset
  -> otpu_seq execute ISA program
```

数据搬运则走另一条路径：

```text
XdmaTransport.mem_write/read()
  -> /dev/xdmaN_h2c_0 / c2h_0
  -> XDMA AXI memory master
  -> otpu_mem_ch[2] / otpu_native_sys
  -> DDR3 / LiteDRAM
  -> otpu_slice DRAM interface
```

### 9.3 寄存器与 core 端口的边界

`otpu_ctrl` 是寄存器协议和 core 控制端口之间的桥梁，不是计算 datapath。
core 内部仍然只看到 module parameter 和端口，例如：

| 寄存器/控制来源 | 连接到 core 的信号 | 作用 |
|---|---|---|
| `CTRL.LOAD`        | `ld_start`                 | 请求 sequencer loader 装载程序 |
| `PROG_ADDR`        | `ld_addr`                  | 程序在 DRAM 中的地址 |
| `PROG_N`           | `ld_n`                     | 程序长度/装载大小 |
| `ARG0..ARG7`       | `rinit[8]`                 | 运行时参数，进入 ISA/seq 执行 |
| `CTRL.RUN` / reset | `rst`、system reset wiring | 启动或停止 core |
| slice halt/error   | `all_halted`、`any_error`  | 返回 `STATUS` |
| slice counters     | `icount` 等统计信号         | 返回计数器寄存器 |

这也解释了为什么 `otpu_top` 源码中看不到 `R_VERSION`、`R_CTRL` 这样的地址常量：
地址解码属于 `otpu_ctrl`，core top 只接收已经解码后的控制信号。

## 10. 仿真 testbench 层

Verilator testbench 实例化并驱动 `otpu_top` ：

```text
tb_top.sv
    -> otpu_top
    -> otpu_top 的 core hierarchy
    -> dump DRAM/TMEM/status
```

`sim/verilator/otpu_native_mem.sv` 和 `sim/verilator/otpu_ldc_mem.sv` 
是 `AXI=1/2` 时被 `otpu_top` 条件引用的仿真 memory backend，属于构建支持文件，
不是 FPGA 板卡的真实 DDR controller。

## 11. 源文件索引

| 层级/功能 | 文件 |
|---|---|
| Core top                  | `rtl/top/otpu_top.sv` |
| Slice integration         | `rtl/top/otpu_slice.sv` |
| ISA package               | `rtl/top/otpu_pkg.sv` |
| Collective                | `rtl/top/otpu_coll.sv` |
| Sequencer                 | `rtl/seq/otpu_seq.sv` |
| DMA                       | `rtl/dma/otpu_dma.sv` |
| DRAM models/adapters      | `rtl/mem/otpu_dram.sv`, `rtl/mem/otpu_native_dram.sv` |
| TMEM / ACT RAM            | `rtl/mem/otpu_tmem.sv`, `rtl/mem/otpu_actram.sv` |
| MXU                       | `rtl/mxu/otpu_mxu.sv` |
| VPU pipeline              | `rtl/vpu/otpu_vpu.sv`, `rtl/vpu/otpu_vtree.sv`, `rtl/vpu/otpu_fpipe.sv` |
| Special functions         | `rtl/vpu/otpu_se_comp.sv`, `rtl/vpu/otpu_se_tail.sv` |
| Quantizer                 | `rtl/vpu/otpu_quant.sv` |
| Technology mapping        | `rtl/tech/` |
| Board shell               | `rtl/shell/`, `rtl/boards/` |
| Verilator memory backends | `sim/verilator/otpu_native_mem.sv`, `sim/verilator/otpu_ldc_mem.sv` |

## 12. 阅读和验证方法

查看一个实例的具体端口连接时，先定位父 module 的实例声明，再跳到目标文件的 `module` 定义。例如：

```sh
rg -n "otpu_slice|otpu_mxu|otpu_vpu" rtl/top/otpu_top.sv rtl/top/otpu_slice.sv
rg -n "^module |u_[a-zA-Z0-9_]+" rtl/top rtl/seq rtl/mem rtl/mxu rtl/vpu
```

实际编译层级还取决于参数（`AXI`、`COLL_GATHER`、`TMEM_IMPL`、`MXU_IMPL`、`HAS_SE` 等）和目标的 source list。
