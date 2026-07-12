import os
import subprocess
import shutil
import numpy
import sysconfig
import platform
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Tell Hatchling this is a compiled C extension, not pure Python
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        # Inject Python and NumPy header locations into Zig's environment
        # os.environ["PYTHON_INCLUDE_DIR"] = sysconfig.get_path("include")
        # os.environ["NUMPY_INCLUDE_DIR"] = numpy.get_include()

        # Run your build step (The "zig build" part)
        cmd = ["zig", "build", "-Doptimize=ReleaseFast"]
        if platform.system() == "Darwin":
            cmd.append("-Dtarget=aarch64-macos.14")
        print(f"Compiling ExSHalos native extensions... ({' '.join(cmd)})")
        subprocess.check_call(cmd)

        # Copy the artifact into your package directory (The "cp" part)
        zig_out = os.path.join("zig-out", "lib")
        pkg_dir = "pyexshalos/lib"
        os.makedirs(pkg_dir, exist_ok=True)

        for file in os.listdir(zig_out):
            shutil.copyfile(os.path.join(zig_out, file), os.path.join(pkg_dir, file))
