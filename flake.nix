{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          system = "x86_64-linux";
          config.allowUnfree = true;
          config.cudaSupport = true;
        };      in
      {
        devShells.default = pkgs.mkShell { packages = with pkgs; [ 
          python312Packages.numpy
          python312Packages.matplotlib
          python312Packages.pandas
	  python312Packages.cupy
          ]; 
        };
      }
    );
}
