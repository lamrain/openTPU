# openTPU 与 CPU、FPGA 及 AXI 外设的交互分析

## 1. 结论先行

当前 openTPU 的 FPGA 版本**不需要集成一个用于运行模型的通用 CPU**。模型程序由 openTPU 自己的硬件 sequencer 执行，外部 Linux PC 的 CPU 通过 PCIe 完成配置、数据搬运、启动和结果读取。

当前 FPGA 设计中确实存在一个 CPU，但它属于 LiteDRAM 生成的 DDR3 自校准逻辑：`boards/ypcb-00338/litedram/core.json` 配置了 `--selfcal`，生成的 LiteDRAM 结构中包含 `selfcal -> cpu (VexRiscv)`。这个 CPU 只负责 DDR PHY/controller calibration，不负责执行 openTPU ISA、不负责运行 LLM，也不是主机 CPU 的替代品。

因此当前系统是：

```text
Linux PC CPU
    |
    | PCIe Gen2 x8
    v
Xilinx XDMA
    |-- BAR0 / AXI-Lite: 控制寄存器、状态、计数器
    |-- H2C/C2H DMA:     主机与板卡 DRAM 之间搬运数据
    v
openTPU FPGA
    |-- otpu_ctrl:       接收 RUN、LOAD、ARG 和状态访问
    |-- openTPU sequencer: 执行 8 x 32-bit ISA
    |-- MXU/VPU/DMA:     执行模型计算和显式数据搬运
    |-- otpu_native_dram -> LiteDRAM -> DDR3
    `-- LiteDRAM selfcal VexRiscv: 仅负责内存校准
```

## 2. CPU 参与了哪些事情

### 2.1 主机 CPU：控制面和软件运行时

主机 CPU 执行 `opentpu/host`、模型加载和运行时软件。主要职责包括：

1. 解析 Hugging Face 模型配置和 safetensors 权重；
2. 根据 FPGA 的 VERSION/CAPS 寄存器确定 `D`、`MCOLS`、`LANES` 和 ISA 能力；
3. 量化权重，生成每个 slice 的 DRAM image；
4. 编译 kernel，得到 openTPU ISA 程序；
5. 通过 XDMA 把程序、权重、输入和 KV Cache 写入板卡 DRAM；
6. 通过 BAR0 的 AXI-Lite 寄存器写入运行参数 `ARG0..ARG7`；
7. 写 RUN/LOAD 控制位，等待 HALTED 或 ERROR；
8. 读取输出、logits、性能计数器和硬件 trace；
9. 完成 tokenizer、chat template、sampling 和 UI。

这些职责由 `opentpu/host/board.py`、`opentpu/host/chat.py`、`opentpu/runtime.py` 以及 `opentpu/llm` 共同完成。

### 2.2 FPGA 内部 sequencer：执行面

模型运行时，真正执行指令的不是 CPU，而是 `otpu_slice` 内的 sequencer。它读取 instruction memory 中的固定长度 ISA 指令，控制：

- DMA；
- TMEM；
- ACT RAM；
- MXU；
- VPU；
- quantizer；
- collective unit。

ISA 的 `LI`、`ADDI`、`LOOP`、`HALT`、`WAITW` 和运行时参数寄存器，使 sequencer 能完成地址计算、循环和主机协作，但它不是通用 CPU：

- 没有通用操作系统；
- 不运行 Python、C 或驱动程序；
- 没有软件栈意义上的异常处理和进程；
- 指令集合围绕显式数据流和矩阵/向量计算设计；
- 程序通常由 host compiler 预先生成。

更准确地说，openTPU 是“主机 CPU + FPGA 专用指令处理器”，而不是“主机 CPU + FPGA 内嵌通用 CPU”。

### 2.3 LiteDRAM 校准 CPU：管理面的一小部分

`boards/ypcb-00338/litedram/core.json` 明确记录：

```text
--selfcal
firmware_toolchain: riscv64-elf-gcc
...
selfcal -> cpu (VexRiscv)
```

它的边界是 DDR3 初始化和校准：

- 生成校准命令；
- 调整 PHY 延迟和读写采样；
- 判断 DDR channel 是否 calibrated；
- 向 LiteDRAM CSR 提供校准状态。

`otpu_ctrl.sv` 只读取两个 channel 的 `calib` 状态，并在两路内存校准完成后才允许 RUN 置位。换句话说，openTPU sequencer 依赖校准结果，但不调用这个 VexRiscv 执行模型程序。

## 3. 一次主机到 openTPU 的交互

### 3.1 上电和发现

这一步不是由 openTPU sequencer 或主机 CPU 上的模型运行时完成的，而是由 FPGA
配置、PCIe/XDMA 驱动和 BAR0 寄存器共同完成。对应实现位置如下：

1. **FPGA 上电、时钟和复位**：板卡顶层
   [`rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv`](rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv)
   接收 PCIe 的 `PERST#`/链路复位，生成 `core_clk`、XDMA 时钟及控制逻辑复位，
   实例化 XDMA block design、LiteDRAM 和 `otpu_board`。其中 `pcie_link_up` 和两路
   DDR `calib` 汇总到板卡状态 LED；板级控制和状态连接在
   [`rtl/boards/ypcb-00338/otpu_board.sv`](rtl/boards/ypcb-00338/otpu_board.sv)。
2. **PCIe 枚举、驱动绑定和设备节点**：Linux 主机侧由
   [`opentpu/host/setup_pcie.sh`](opentpu/host/setup_pcie.sh) 安装/加载带 openTPU
   标记的 Xilinx XDMA 驱动。`--rescan` 路径在 FPGA 重新配置后移除并重新扫描 PCIe
   总线、绑定 `xdma` 驱动，并等待 `/dev/xdma0_user` 出现；脚本的 `cards()` 依据
   PCI vendor/device 或 subsystem ID 筛选板卡，`read_id()` 再读取 BAR0 偏移
   `0x00` 的 openTPU ID。
3. **PCIe 到 FPGA 内部总线的落点**：顶层 RTL 将 XDMA 的 BAR0 AXI-Lite 控制窗口
   连接到 `otpu_ctrl`，将 XDMA 的 AXI 主口连接到两路 DRAM 通道；具体时钟、复位、
   AXI-Lite 和 DMA 连接说明见
   [`rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv`](rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv)
   和 [`rtl/boards/ypcb-00338/otpu_native_sys.sv`](rtl/boards/ypcb-00338/otpu_native_sys.sv)。
4. **设备自识别和能力发现**：主机
   [`opentpu/host/board.py`](opentpu/host/board.py) 中的 `XdmaTransport` 通过
   `/dev/xdma0_user` 读写 BAR0，并由 `Board` 校验 `R_ID == ID_OTPU`；`Board.info()`
   读取 `VERSION`、`STATUS`、`REGMAP`、`CAPS`、`CORE_KHZ`、`BUILD_ID` 等寄存器，
   生成运行时设备配置。寄存器地址、ID 常量、状态位和能力位定义在
   [`opentpu/host/regs.py`](opentpu/host/regs.py)。
5. **FPGA 侧寄存器实现**：
   [`rtl/boards/ypcb-00338/otpu_ctrl.sv`](rtl/boards/ypcb-00338/otpu_ctrl.sv)
   （AXI4-Lite BAR0 从设备）实现 `ID`、`VERSION`、`CAPS`、`BUILD_ID` 和
   `STATUS`。`STATUS.CALIB0/CALIB1` 反映两路 DDR 校准状态，且 `CTRL.RUN` 只有在
   两路校准完成后才会生效。因此主机发现到的是“PCIe/XDMA 可访问且寄存器 ID 正确”的
   openTPU bitstream，而不是通过枚举一个 FPGA 内通用 CPU 来发现。

FPGA 配置完成后，通过 XDMA 暴露 PCIe 设备。主机驱动创建：

| 主机设备节点 | FPGA 侧用途 |
|---|---|
| `/dev/xdma0_user` | BAR0，控制寄存器和状态寄存器 |
| `/dev/xdma0_h2c_0` | host-to-card，写板卡 DRAM |
| `/dev/xdma0_c2h_0` | card-to-host，读板卡 DRAM |

主机首先读取：

- `ID`，确认是 openTPU；
- `VERSION`，获取 `D/MCOLS/LANES`；
- `CAPS`，确认 4-bit MM、run arguments、WAITW、GEN、STREAM 等能力；
- `BUILD_ID`，确认 bitstream 版本；
- `STATUS`，确认 DDR calibration、RUN、HALTED、ERROR 和 ECC 状态。

### 3.2 装载程序和数据

Host 通过 H2C DMA 把逻辑 DRAM image 写入板卡。板卡的两个 DDR3 channel 在 host 软件看来是一个逻辑地址空间，`opentpu/host/board.py` 负责：

- 64-byte beat 对齐；
- channel 0/1 interleave；
- `CHASH` channel 映射；
- 大传输分块；
- XDMA DMA lock；
- 非理想 host buffer 的 staging copy。

程序本身也放在 DRAM 中。主机写 `PROG_ADDR` 和 `PROG_N`，再通过 `CTRL.LOAD` 让 FPGA loader 把指定范围复制到 instruction memory。

### 3.3 启动一次 run

主机向 `ARG0..ARG7` 写入 token id、position 或其他运行时地址参数，然后向 `CTRL.RUN` 写 1。

`otpu_ctrl.sv` 将这些参数映射到 sequencer 的 `R8..R15`，并只有在两个 DDR channel 都 calibrated 时才释放 RUN。之后：

```text
host 写 ARG/RUN
    -> otpu_ctrl
    -> core_rst 释放
    -> sequencer 从 IMEM 取指
    -> DMA/MXU/VPU/quantizer 访问 DRAM、TMEM、ACT RAM
    -> HALT
    -> host 轮询 STATUS 并读取结果
```

在 resident decode 模式下，同一个程序可以多次使用，host 每个 token 只更新少量参数。若使用 streamed logits，host 还可以在 FPGA 运行期间通过 C2H DMA 读取已经生成的 logits 片段。

## 4. 当前版本是否需要集成 CPU

### 4.1 对当前 FPGA image：不需要通用 CPU

当前 openTPU FPGA image 已经具备完成一次模型 run 所需的执行能力：

- sequencer 执行 ISA；
- control registers 接收 host 命令；
- DMA 读取和写入 DDR；
- MXU 完成矩阵乘法；
- VPU 完成 fp32 和归约操作；
- quantizer 处理 int8/4-bit 数据；
- collective unit 处理 slice 间数据交换。

因此，增加 MicroBlaze、RISC-V 或 ARM 软核并不是当前功能闭环的必要条件。它不会自动替代 compiler、host runtime 或 sequencer，反而会增加：

- FPGA 资源消耗；
- BRAM/TCM/DDR 地址空间规划；
- 固件启动和下载流程；
- CPU 与 accelerator 的共享内存一致性问题；
- 中断和 mailbox 协议；
- 软件调试和验证面。

### 4.2 什么情况下可能需要 FPGA 内 CPU

如果目标从“主机驱动的 PCIe 加速卡”变成“独立 SoC 或边缘设备”，集成 CPU 才可能有明显价值。例如：

- 没有外部 Linux PC，设备需要独立启动；
- 需要在卡上运行 tokenizer、sampling、任务队列或模型管理；
- 需要处理多个模型和动态内存分配；
- 需要通过 Ethernet、USB、显示或摄像头接口独立提供服务；
- 需要把 PCIe 断开后的控制和故障恢复放在设备侧。

即使那时加入 CPU，它更适合做 control-plane，而不是替换 MXU/VPU 或让 CPU 执行矩阵计算。一个合理的结构会是：

```text
embedded CPU
    | mailbox / command queue / shared DRAM
    v
openTPU sequencer + MXU/VPU/DMA
```

### 4.3 一个容易混淆的例外

“工程里已经有 CPU”与“openTPU 需要 CPU 才能运行”是两件事：LiteDRAM 的 VexRiscv 是内存校准固件处理器，属于 DDR subsystem；它不承担 openTPU 程序执行，也不处理主机的模型 runtime。

## 5. 与 JPEG decoder 等 AXI 总线设备的区别

参考 `uvm_jpeg/rtl/jpeg/jpeg_wrapper/jpeg_decoder_fpga.v`，JPEG decoder 的典型边界是：

```text
CPU/系统互连
    |
    +-- AXI-Lite slave -> JPEG 配置寄存器
    |
    +-- JPEG AXI master -> 系统 DDR / AXI memory
                              |
                              v
                         JPEG decoder core
```

JPEG wrapper 通常包含：

- 一个 AXI-Lite slave 配置口；
- 一个 AXI master 数据口；
- AXI-Lite 到 APB 的桥；
- JPEG core；
- 完成中断 `jpeg_irq_pend`。

JPEG 的软件交互一般是：

```text
CPU 写源地址、目标地址、长度、宽高和 start
    -> JPEG 内部 AXI master 读取压缩数据
    -> JPEG 解码并通过 AXI master 写回像素
    -> JPEG 产生 IRQ
    -> CPU 读取状态/尺寸并处理结果
```

这和 openTPU 有相似之处：两者都可以采用“寄存器配置 + 数据 master + 完成通知”的加速器模型。但两者在执行模型、数据组织和控制粒度上明显不同。

## 6. 关键区别对照

| 维度 | openTPU 当前 FPGA 版本 | JPEG decoder AXI 外设 |
|---|---|---|
| 外部控制者 | Linux PC CPU，经 PCIe/XDMA | SoC CPU，经 AXI-Lite/APB 或系统互连 |
| 配置接口 | PCIe BAR0 后的 AXI4-Lite `otpu_ctrl` | AXI-Lite slave，通常桥到 APB 寄存器 |
| 数据接口 | PCIe XDMA 可搬运 DRAM；openTPU 内部再经 native DRAM/LiteDRAM 访问 DDR3 | JPEG core 自己作为 AXI master 访问系统 DDR |
| 指令执行 | 有专用 openTPU ISA 和 sequencer | 通常由固定 JPEG FSM/控制器执行，不是可编程通用 ISA |
| 计算组织 | DMA、MXU、VPU、quantizer、TMEM、ACT RAM 协同执行 | 熵解码、IDCT、颜色转换、FIFO 等固定数据流 |
| 数据布局 | 权重、scale、KV Cache、TMEM、ACT RAM、slice shard 都由 compiler/runtime 规划 | 通常是 source buffer、destination buffer、stride/尺寸等较简单的 buffer 描述 |
| 运行时间 | 可执行不同模型、不同 kernel 和不同 ISA program | 主要执行 JPEG decode 这一固定功能 |
| 主机参与 | 编译、量化、image layout、sampling、程序装载和运行控制 | 写寄存器、提交 buffer、等待 IRQ、消费输出 |
| 设备内部 CPU | 模型执行不需要；LiteDRAM 只有校准 CPU | 通常不需要内部 CPU |
| 结果同步 | HALTED、STATUS、counter、trace、streamed logits、WAITW | IRQ、状态寄存器和 AXI 写响应 |
| 存储系统 | 板卡自带双 DDR3，openTPU 直接控制 native memory adapter | 多数系统共享 SoC DDR 和 AXI interconnect |

## 7. openTPU 不是简单的 AXI peripheral

从总线连接形式看，openTPU 当然也是一个 AXI 外设的一部分：

- `otpu_board` 向上提供 AXI4-Lite control slave；
- PCIe XDMA 将 host BAR 访问转换为 AXI-Lite；
- openTPU memory path 向 LiteDRAM 提供 native memory master；
- 某些历史或仿真路径可以接 AXI memory。

但从系统抽象看，它更接近**带专用 ISA 和本地内存系统的 accelerator subsystem**，而不是一个只有几个寄存器和一个固定数据通路的 AXI peripheral。

区别在于 openTPU 的配置不是简单的“填写一组寄存器后启动固定算法”：

```text
编译器生成 program
  -> program 放入 instruction memory
  -> 权重、scale、KV Cache 放入 DRAM
  -> runtime 设置地址参数
  -> sequencer 执行可变化的 ISA program
```

JPEG decoder 的配置更接近：

```text
CPU 填 source/destination/length/format
  -> 固定 JPEG datapath 启动
  -> 完成后 IRQ
```

因此 openTPU 的 AXI-Lite 寄存器只是 control plane 的入口，不能代表整个软件/硬件接口；真正的应用接口是 `compiler + ISA + memory image + runtime protocol`。

## 8. openTPU 与 JPEG 的交互时序对比

### JPEG

```text
CPU
 |-- AXI-Lite: 配置寄存器
 |-- 等待 IRQ
 |<-- AXI-Lite: 读取状态
 |
JPEG AXI master <--> 系统 DDR
```

CPU 通常只提交一个任务，JPEG 内部 FSM 负责整个处理过程。

### openTPU

```text
CPU
 |-- PCIe BAR/AXI-Lite: VERSION/CAPS/ARG/RUN/STATUS
 |-- H2C DMA: program + weights + inputs -> board DRAM
 |-- C2H DMA: outputs/logits <- board DRAM
 |-- 可在运行期间流式读取 logits
 |
openTPU sequencer
 |-- 从 IMEM 取 ISA
 |-- 控制 DMA/TMEM/ACT/MXU/VPU/QNT
 |
otpu_native_dram <--> LiteDRAM <--> DDR3 channel 0/1
```

openTPU 的 CPU 交互频率可以很低：程序和模型 image 预先加载后，一个 resident program 可以连续服务多个 token。CPU 更像 host runtime 和调度者，而不是每个硬件算子都参与控制。

## 9. 设计判断

### 当前项目是否应该加入 FPGA CPU

对当前 YPCB-00338 PCIe 加速卡，建议保持现状：

- 使用外部 PC CPU 负责模型加载、编译和服务控制；
- 使用 openTPU sequencer 负责设备内 program execution；
- 保留 LiteDRAM 自带的校准 CPU；
- 不额外加入通用 FPGA CPU。

这样可以保持控制边界清晰，并把 FPGA 资源集中给 MXU、VPU、TMEM、FIFO、DMA 和 DDR 带宽。

只有当产品目标变成脱离 PC 独立运行时，才值得评估增加嵌入式 CPU。此时应先定义 mailbox、共享内存、固件启动、异常恢复和任务队列，而不是简单地把一个 CPU 接到 AXI 总线上。

### 一句话总结

JPEG decoder 是“CPU 配置的固定功能 AXI 加速器”；当前 openTPU 是“CPU 通过 PCIe 驱动的、带专用 ISA、编译器和本地 DRAM 数据流的可编程 FPGA 加速子系统”。