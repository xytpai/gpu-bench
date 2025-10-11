import sys
import glob
from subprocess import check_call, check_output


build_tool_map = {
    'rocm': 'build_rocm.sh',
    'hip': 'build_rocm.sh',
    'cuda': 'build_cuda.sh',
    'nvcc': 'build_cuda.sh',
    'cu': 'build_cuda.sh',
}
sources = sorted(glob.glob(f"standard/*.cpp"))


model = sys.argv[1].strip()
build_tool = build_tool_map[model]
for source in sources:
    cmd = f"bash {build_tool} {source}"
    print(f"\n{cmd}\n")
    cmd = cmd.split(' ')
    check_call(cmd)
    check_call(['./a.out'])
