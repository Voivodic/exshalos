{
  description = "Dev shell for pyexshalos (builds C/C++ setuptools extensions with voro++, GSL, FFTW3)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # ponytail: C extensions don't compile against python 3.14 yet; pin 3.13.
      # Revisit when upstream supports 3.14 / numpy 3.14 headers.
      py = pkgs.python313;
      pypkgs = py.pkgs;

      # shared: env vars so the native `zig build` finds GSL/FFTW/OpenMP.
      # ponytail: one block reused by devShell and the test app (DRY).
      libEnv = ''
        export CC=gcc
        export CXX=g++
        export C_INCLUDE_PATH="${pkgs.gsl.dev}/include:${pkgs.fftw.dev}/include:${pkgs.fftwFloat.dev}/include:''${C_INCLUDE_PATH:-}"
        export CPLUS_INCLUDE_PATH="''${C_INCLUDE_PATH:-}:''${CPLUS_INCLUDE_PATH:-}"
        export LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.fftw}/lib:${pkgs.fftwFloat}/lib:${pkgs.llvmPackages.openmp}/lib:''${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.fftw}/lib:${pkgs.fftwFloat}/lib:${pkgs.llvmPackages.openmp}/lib:''${LD_LIBRARY_PATH:-}"
      '';
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          gnumake
          gsl
          fftw
          fftwFloat
          py
          pypkgs.numpy
          pypkgs.scipy
          pypkgs.setuptools
          pypkgs.wheel
          pypkgs.pip
          pypkgs.hatchling
          pypkgs.pytest
          git
          zig
          llvmPackages.openmp
        ];

        shellHook = libEnv + ''
          # ponytail: venv sidesteps PEP 668 (externally-managed) so pip works.
          # numpy/scipy inherited from nix via --system-site-packages.
          if [ ! -d .venv ]; then
            python -m venv --system-site-packages .venv
          fi
          source .venv/bin/activate
          # reinstall only when pyproject.toml changed
          if [ pyproject.toml -nt .venv/.installed ]; then
            pip install -e . --no-build-isolation
            touch .venv/.installed
          fi

          echo "Ready. Run tests with:"
          echo "  python -m pytest tests/test_finder.py"
        '';
      };

      # `nix run .#test-halovoid` — installs pyexshalos into a throwaway venv
      # and runs tests/test_halovoid.py with pytest.
      # ponytail: impure build (nix run runs outside the sandbox, so the zig
      # dependency fetch for voro++ works). Make a real derivation once the
      # voro fetch is pre-cached via fetchurl + ZIG_GLOBAL_CACHE_DIR.
      apps.${system}.test-halovoid =
        let
          pythonEnv = py.withPackages (ps: with ps; [
            numpy scipy pytest hatchling setuptools pip wheel
          ]);
          runtimeDeps = with pkgs; [
            gcc zig gnumake gsl fftw fftwFloat llvmPackages.openmp pythonEnv
          ];
        in
        {
          type = "app";
          program = "${pkgs.writeShellApplication {
            name = "test-halovoid";
            excludeShellChecks = [ "SC1091" ]; # activate script is generated at runtime
            runtimeInputs = runtimeDeps;
            text = libEnv + ''
              set -euo pipefail
              WORK="$(mktemp -d)"
              # flake source in the store is read-only — copy to a writable tree
              # so the build (zig-out/, pyexshalos/lib/) can write artifacts.
              cp -aT "${self}" "$WORK"
              chmod -R +w "$WORK"
              cd "$WORK"
              python -m venv --system-site-packages .venv
              source .venv/bin/activate
              pip install -e . --no-build-isolation
              # ponytail: must run pytest via venv python (python -m pytest), not the
              # bare `pytest` on PATH — that binary's shebang points at the nix python,
              # which never loads .venv/site-packages/_pyexshalos.pth (editable hook).
              python -m pytest tests/test_halovoid.py
            '';
          }}/bin/test-halovoid";
        };
    };
}
