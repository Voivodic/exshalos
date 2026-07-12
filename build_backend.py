import os
import subprocess
import shutil
import numpy
import sysconfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Inject Python and NumPy header locations into Zig's environment
        # os.environ["PYTHON_INCLUDE_DIR"] = sysconfig.get_path("include")
        # os.environ["NUMPY_INCLUDE_DIR"] = numpy.get_include()

        # Run your build step (The "zig build" part)
        print("Compiling ExSHalos native extensions via Zig...")
        subprocess.check_call(["zig", "build", "-Doptimize=ReleaseFast"])

        # Copy the artifact into your package directory (The "cp" part)
        zig_out = os.path.join("zig-out", "lib")
        pkg_dir = "pyexshalos/lib"
        os.makedirs(pkg_dir, exist_ok=True)

        for file in os.listdir(zig_out):
            shutil.copyfile(os.path.join(zig_out, file), os.path.join(pkg_dir, file))
