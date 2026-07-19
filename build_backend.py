import os
import subprocess
import shutil
import platform
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Tell Hatchling this is a compiled C extension, not pure Python
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        # Run your build step (The "zig build" part)
        cmd = ["zig", "build", "-Doptimize=ReleaseFast"]
        if platform.system() == "Darwin":
            cmd.append("-Dtarget=aarch64-macos.14.0.0")
        print(f"Compiling ExSHalos native extensions... ({' '.join(cmd)})")
        subprocess.check_call(cmd)

        # Copy the artifact into your package directory
        zig_out = os.path.join("zig-out", "lib")
        pkg_dir = "pyexshalos/lib"
        os.makedirs(pkg_dir, exist_ok=True)

        for file in os.listdir(zig_out):
            shutil.copyfile(os.path.join(zig_out, file), os.path.join(pkg_dir, file))
