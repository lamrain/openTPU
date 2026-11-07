# 已经测试的命令
# RTL tests also need Verilator 5 回归测试
python3 -m pytest -q          

# 4-bit layers ISA 与 CPU golden 比对
python3 tools/validate.py --model qwen3
python3 tools/validate.py --model lfm2 --wformat fp4 --head-format int8

# 运行 RTL 与 ISA 比对
python3 tools/validate.py --model lfm2 --backend rtl --against isa --tokens 3 "The capital of France is"

# chat on the simulator 基于 ISA 的对话窗口
otpu-chat --model qwen3 --backend isa
otpu-chat --model lfm2  --backend isa

# 异常：MemoryError: model image needs 514 MiB per slice, DRAM_BYTES is 512 MiB
otpu-chat --model lfm2  --backend board-sim
# 太慢
otpu-chat --model qwen3 --backend board-sim

# RTL 检查
cd ~/Workspace/openTPU/boards/ypcb-00338
make lint
# Vivado 生成 bit，Windows 安装 Git bash，Python 3.12.10，numpy，boards/ypcb-00338 目录文件已梳理
make bit