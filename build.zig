//! Build script for pyexshalos C/C++ extension modules using Zig.
//!
//! Produces Python-importable shared libraries (.so) for:
//!   spectrum, exshalos, hod, analytical, finder
//!
//! Usage:
//!   zig build                     # builds all modules into zig-out/pyexshalos/lib/
//!   zig build -Ddouble-precision  # use fftw3/fftw3_omp instead of fftw3f/fftw3f_omp

const std = @import("std");

// ─── Per-module configuration ────────────────────────────────────────────

const ExtModule = struct {
    name: []const u8,
    src_dir: []const u8,
    files: []const []const u8,
    include_subdir: []const u8,
    libs: []const []const u8,
    cpp: bool = false,
    voro: bool = false,
};

const modules = [_]ExtModule{
    .{
        .name = "spectrum",
        .src_dir = "src/spectrum",
        .files = &.{
            "spectrum_h.c", "abundance.c", "gridmodule.c", "powermodule.c",
            "bimodule.c",   "trimodule.c", "bias.c",       "spectrum.c",
        },
        .include_subdir = "spectrum",
        .libs = &.{ "m", "gsl", "gslcblas" },
    },
    .{
        .name = "exshalos",
        .src_dir = "src/exshalos",
        .files = &.{
            "fftlog.c",     "exshalos_h.c",       "density_grid.c",
            "find_halos.c", "cells_in_spheres.c", "lpt.c",
            "box.c",        "exshalos.c",
        },
        .include_subdir = "exshalos",
        .libs = &.{ "m", "fftw3", "gsl", "gslcblas" },
    },
    .{
        .name = "hod",
        .src_dir = "src/hod",
        .files = &.{ "hod_h.c", "populate_halos.c", "split_galaxies.c", "hod.c" },
        .include_subdir = "hod",
        .libs = &.{ "m", "gsl", "gslcblas" },
    },
    .{
        .name = "analytical",
        .src_dir = "src/analytical",
        .files = &.{ "fftlog.c", "analytical_h.c", "clpt.c", "analytical.c" },
        .include_subdir = "analytical",
        .libs = &.{ "m", "fftw3", "gsl", "gslcblas" },
    },
    .{
        .name = "halovoid",
        .src_dir = "src/halovoid",
        .files = &.{ "finder.cpp", "halovoid_h.cpp", "halovoid.cpp" },
        .include_subdir = "halovoid",
        .libs = &.{"m"},
        .cpp = true,
        .voro = true,
    },
};

// ─── Environment detection ─────────────────────────────────────────────────

const EnvInfo = struct {
    py_include: []const u8,
    numpy_include: []const u8,
    ext_suffix: []const u8,
    lib_paths: []const []const u8,
    gomp_dir: []const u8,
    gcc_include: []const u8,
};

fn runCapture(b: *std.Build, argv: []const []const u8) ![]const u8 {
    const result = std.process.run(b.graph.arena, b.graph.io, .{
        .argv = argv,
        .environ_map = &b.graph.environ_map,
    }) catch return error.RunFailed;
    return std.mem.trim(u8, result.stdout, " \r\n");
}

fn detectEnv(b: *std.Build) !EnvInfo {
    // Python + numpy in one subprocess
    const py_script =
        \\import sysconfig, numpy
        \\print(sysconfig.get_path("include"))
        \\print(sysconfig.get_config_var("EXT_SUFFIX"))
        \\print(numpy.get_include())
    ;
    const py_out = blk: {
        const r = std.process.run(b.graph.arena, b.graph.io, .{
            .argv = &.{ "python3", "-c", py_script },
            .environ_map = &b.graph.environ_map,
        }) catch break :blk null;
        break :blk r.stdout;
    } orelse {
        std.debug.print("error: could not run 'python3'. Run inside `nix develop` or activate .venv.\n", .{});
        return error.PythonNotFound;
    };

    var it = std.mem.splitScalar(u8, py_out, '\n');
    const py_include = std.mem.trim(u8, it.next() orelse "", " \r");
    const ext_suffix = std.mem.trim(u8, it.next() orelse "", " \r");
    const numpy_include = std.mem.trim(u8, it.next() orelse "", " \r");
    if (py_include.len == 0 or ext_suffix.len == 0 or numpy_include.len == 0) {
        std.debug.print("error: unexpected python3 output: {s}\n", .{py_out});
        return error.PythonOutputInvalid;
    }

    // Parse $LIBRARY_PATH
    var lib_paths: []const []const u8 = &.{};
    if (b.graph.environ_map.get("LIBRARY_PATH")) |lp| {
        var count: usize = 0;
        var splitter = std.mem.splitScalar(u8, lp, ':');
        while (splitter.next()) |part| {
            if (part.len > 0) count += 1;
        }
        const arr = b.graph.arena.alloc([]const u8, count) catch @panic("OOM");
        var i: usize = 0;
        splitter = std.mem.splitScalar(u8, lp, ':');
        while (splitter.next()) |part| {
            if (part.len > 0) {
                arr[i] = part;
                i += 1;
            }
        }
        lib_paths = arr;
    }

    // Find libgomp via the C compiler
    var gomp_dir: []const u8 = "";
    if (runCapture(b, &.{ "gcc", "-print-file-name=libgomp.so" })) |path| {
        // gcc returns full path like /nix/store/.../lib/libgomp.so — extract dir
        if (std.fs.path.dirname(path)) |dir| {
            if (dir.len > 0 and !std.mem.eql(u8, dir, ".")) gomp_dir = dir;
        }
    } else |_| {}

    // GCC's private include dir (omp.h, etc.)
    var gcc_include: []const u8 = "";
    if (runCapture(b, &.{ "gcc", "-print-file-name=include" })) |path| {
        if (path.len > 0 and !std.mem.eql(u8, path, "include")) gcc_include = path;
    } else |_| {}

    return .{
        .py_include = py_include,
        .numpy_include = numpy_include,
        .ext_suffix = ext_suffix,
        .lib_paths = lib_paths,
        .gomp_dir = gomp_dir,
        .gcc_include = gcc_include,
    };
}

// ─── Build ─────────────────────────────────────────────────────────────────

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const double_precision = b.option(
        bool,
        "double-precision",
        "Use double-precision FFTW3 (fftw3/fftw3_omp) instead of float (fftw3f/fftw3f_omp)",
    ) orelse false;

    const env = detectEnv(b) catch @panic("Environment detection failed");

    const fftw_libs: []const []const u8 = if (double_precision)
        &.{ "fftw3", "fftw3_omp" }
    else
        &.{ "fftw3f", "fftw3f_omp" };

    const c_flags: []const []const u8 = &.{ "-O2", "-funroll-loops", "-fopenmp" };
    const cpp_flags: []const []const u8 = &.{ "-O2", "-funroll-loops", "-fopenmp", "-std=c++23" };

    // ─── Voro++ static libraries (compiled from fetched dependency source) ────
    const voro_dep = b.dependency("voro", .{ .target = target, .optimize = optimize });
    const voro_3d = blk: {
        const m = b.createModule(.{ .target = target, .optimize = optimize, .link_libcpp = true, .pic = true });
        m.addCSourceFiles(.{
            .root = voro_dep.path("src"),
            .files = &.{ "cell.cc", "common.cc", "container.cc", "unitcell.cc", "v_compute.cc", "c_loops.cc", "v_base.cc", "wall.cc", "pre_container.cc", "container_prd.cc" },
            .flags = &.{ "-O2", "-fopenmp" },
            .language = .cpp,
        });
        m.addIncludePath(voro_dep.path("src"));
        break :blk b.addLibrary(.{ .name = "voro++", .root_module = m, .linkage = .static });
    };
    const voro_2d = blk: {
        const m = b.createModule(.{ .target = target, .optimize = optimize, .link_libcpp = true, .pic = true });
        m.addCSourceFiles(.{
            .root = voro_dep.path("2d/src"),
            .files = &.{ "common.cc", "cell_2d.cc", "container_2d.cc", "v_base_2d.cc", "v_compute_2d.cc", "c_loops_2d.cc", "wall_2d.cc", "cell_nc_2d.cc", "ctr_boundary_2d.cc", "ctr_quad_2d.cc", "quad_march.cc" },
            .flags = &.{ "-O2", "-fopenmp" },
            .language = .cpp,
        });
        m.addIncludePath(voro_dep.path("2d/src"));
        break :blk b.addLibrary(.{ .name = "voro++_2d", .root_module = m, .linkage = .static });
    };

    for (&modules) |ext| {
        // 1 — Create module
        const mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = if (ext.cpp) true else null,
            .pic = true,
        });

        // 2 — C/C++ sources
        mod.addCSourceFiles(.{
            .root = b.path(ext.src_dir),
            .files = ext.files,
            .flags = if (ext.cpp) cpp_flags else c_flags,
            .language = if (ext.cpp) .cpp else .c,
        });

        // 3 — Include dirs: Python, numpy, project headers, gcc (omp.h)
        mod.addSystemIncludePath(.{ .cwd_relative = env.py_include });
        mod.addSystemIncludePath(.{ .cwd_relative = env.numpy_include });
        if (env.gcc_include.len > 0) mod.addSystemIncludePath(.{ .cwd_relative = env.gcc_include });
        
        // --- MACOS FIX: Add Homebrew OpenMP Include Paths ---
        if (target.result.os.tag == .macos) {
            if (target.result.cpu.arch == .aarch64) {
                mod.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/libomp/include" });
            } else {
                mod.addSystemIncludePath(.{ .cwd_relative = "/usr/local/opt/libomp/include" });
            }
        }
        // ----------------------------------------------------

        const inc_path = b.fmt("include/{s}", .{ext.include_subdir});
        mod.addIncludePath(b.path(inc_path));
        if (ext.voro) {
            mod.addIncludePath(voro_dep.path("src"));
            mod.addIncludePath(voro_dep.path("2d/src"));
        }

        // 4 — Library search paths (from $LIBRARY_PATH + gcc libgomp dir)
        for (env.lib_paths) |p| mod.addLibraryPath(.{ .cwd_relative = p });
        if (env.gomp_dir.len > 0) mod.addLibraryPath(.{ .cwd_relative = env.gomp_dir });

        // --- MACOS FIX: Add Homebrew OpenMP Library Paths ---
        if (target.result.os.tag == .macos) {
            if (target.result.cpu.arch == .aarch64) {
                mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/libomp/lib" });
            } else {
                mod.addLibraryPath(.{ .cwd_relative = "/usr/local/opt/libomp/lib" });
            }
        }
        // ----------------------------------------------------

        // 5 — Libraries (.needed = true forces DT_NEEDED even if only an
        //     indirect dep needs it — e.g. libgsl needs cblas_* from gslcblas)
        if (ext.voro) {
            mod.linkLibrary(voro_3d);
            mod.linkLibrary(voro_2d);
        }
        for (ext.libs) |lib| {
            mod.linkSystemLibrary(lib, .{ .needed = true });
        }
        for (fftw_libs) |lib| {
            mod.linkSystemLibrary(lib, .{ .needed = true });
        }
        mod.linkSystemLibrary("gomp", .{ .needed = true }); // OpenMP runtime

        if (double_precision) {
            mod.addCMacro("DOUBLEPRECISION_FFTW", "");
        }

        // 6 — Shared library
        const lib = b.addLibrary(.{
            .name = ext.name,
            .root_module = mod,
            .linkage = .dynamic,
        });
        
        // --- MACOS FIX: Allow undefined symbols for Python C-API ---
        if (target.result.os.tag == .macos) {
            lib.linker_allow_shlib_undefined = true;
        }
        // -----------------------------------------------------------

        // 7 — Install with Python extension suffix
        const dest = b.fmt("lib/{s}{s}", .{ ext.name, env.ext_suffix });
        const install = b.addInstallFile(lib.getEmittedBin(), dest);
        install.step.dependOn(&lib.step);
        b.getInstallStep().dependOn(&install.step);
    }
}
