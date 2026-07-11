{
  description = "Dev shell for pyexshalos (builds C/C++ setuptools extensions with voro++, GSL, FFTW3)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          gnumake
          gsl
          fftw
          fftwFloat
          python3
          python3Packages.numpy
          python3Packages.scipy
          python3Packages.setuptools
          python3Packages.wheel
          python3Packages.pip
          python3Packages.hatchling
          git
          zig
          llvmPackages.openmp
        ];

        shellHook = ''
          export CC=gcc
          export CXX=g++
          # setup.py hardcodes /usr-style search paths that are empty under Nix;
          # make GSL + FFTW discoverable via the standard env vars.
          export C_INCLUDE_PATH="${pkgs.gsl}/include:${pkgs.fftw.dev}/include:${pkgs.fftwFloat.dev}/include:$C_INCLUDE_PATH"
          export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"
          export LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.fftw}/lib:${pkgs.fftwFloat}/lib:${pkgs.llvmPackages.openmp}/lib:$LIBRARY_PATH"
          export LD_LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.fftw}/lib:${pkgs.fftwFloat}/lib:${pkgs.llvmPackages.openmp}/lib:$LD_LIBRARY_PATH"

          # ponytail: venv sidesteps PEP 668 (externally-managed) so pip works.
          # numpy/scipy inherited from nix via --system-site-packages.
          if [ ! -d .venv ]; then
            python -m venv --system-site-packages .venv
          fi
          source .venv/bin/activate
          # reinstall only when setup.py / pyproject.toml changed
          if [ setup.py -nt .venv/.installed ] || [ pyproject.toml -nt .venv/.installed ]; then
            pip install -e . --no-build-isolation
            touch .venv/.installed
          fi

          echo "Ready. Run tests with:"
          echo "  python tests/test_finder.py"
        '';
      };
    };
}
