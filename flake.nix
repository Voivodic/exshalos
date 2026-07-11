{
  description = "Dev shell for pyexshalos (builds C/C++ setuptools extensions with voro++, GSL, FFTW3)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # voro++ source pinned to main @ 2026-03-04
      voroSrc = pkgs.fetchFromGitHub {
        owner = "chr1shr";
        repo = "voro";
        rev = "b0dac575a47af0f90b5b100e6dc199a493c7cb83";
        sha256 = "sha256-hFTaPqF2PMW1HRMZjURGxfDyEpxa9spiN6cDYyTOefg=";
      };

      # Build the 3D voro++ shared library via its CMakeLists.txt
      voroLib = pkgs.stdenv.mkDerivation {
        pname = "voro";
        version = "0.4.6";
        src = voroSrc;
        nativeBuildInputs = [ pkgs.cmake ];
        cmakeFlags = [ "-DBUILD_SHARED_LIBS=ON" ];
        # headers live in src/, exposed via the cmake install rule
      };

      # Build the 2D voro++ static library (no CMake for 2D; use its Makefile
      # object list directly). libvoro++_2d.a exposes container_2d,
      # voronoicell_neighbor_2d, c_loop_all_2d, etc.
      voroLib2d = pkgs.stdenv.mkDerivation {
        pname = "voro-2d";
        version = "0.4.6";
        src = "${voroSrc}/2d";
        # 2d/src/Makefile object list (excluding the broken v_connect etc.)
        objs = [ "common" "cell_2d" "container_2d" "v_base_2d" "v_compute_2d"
                 "c_loops_2d" "wall_2d" "cell_nc_2d" "ctr_boundary_2d"
                 "ctr_quad_2d" "quad_march" ];
        buildPhase = ''
          cd src
          for o in $objs; do
            g++ -fopenmp -O2 -I. -c $o.cc -o $o.o
          done
          ar rs libvoro++_2d.a *.o
        '';
        installPhase = ''
          mkdir -p $out/lib
          cp libvoro++_2d.a $out/lib/
        '';
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          gnumake
          cmake
          pkg-config
          gsl
          fftw
          fftwFloat
          python3
          python3Packages.numpy
          python3Packages.scipy
          python3Packages.setuptools
          python3Packages.wheel
          python3Packages.pip
          git
        ];

        shellHook = ''
          export CC=gcc
          export CXX=g++
          export VORO_INC="${voroSrc}/src"
          export VORO_LIB="${voroLib}/lib"
          export VORO_2D_SRC="${voroSrc}/2d/src"
          export VORO_2D_LIB="${voroLib2d}/lib"
          # setup.py hardcodes /usr-style search paths that are empty under Nix;
          # make GSL + FFTW discoverable via the standard env vars.
          export C_INCLUDE_PATH="${pkgs.gsl}/include:${pkgs.fftw.dev}/include:${pkgs.fftwFloat.dev}/include:$C_INCLUDE_PATH"
          export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"
          export LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.fftw}/lib:${pkgs.fftwFloat}/lib:$LIBRARY_PATH"
          export LD_LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.fftw}/lib:${pkgs.fftwFloat}/lib:$LD_LIBRARY_PATH"

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
