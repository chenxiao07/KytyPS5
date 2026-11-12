#!/usr/bin/env bash
# FSR 4 on Linux: builds ffx-vk-linux (github.com/DuIslingr/ffx-vk-linux, MIT), the small PE loader that runs AMD's
# Windows FSR DLLs in this native process, and puts libffx_vk_linux.so in the emulator's "fsr4" folder.
#   ./get-fsr4-linux.sh [<the emulator's folder>]     (default: _Build/linux, where build-linux.sh puts it)
#
# It does NOT download AMD's DLLs: their licence allows redistribution in binary form only, so they are yours to
# supply. Put these in the same fsr4 folder:
#   amd_fidelityfx_loader_vk.dll, amd_fidelityfx_upscaler_vk.dll   AMD FSR SDK 2.4: Kits/FidelityFX/signedbin
#                                                                  (github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
#   amdvkffx64.dll                                                 AMD Adrenalin driver for Windows (26.10.1 was
#                                                                  the one ffx-vk-linux was tested with): the FSR 4
#                                                                  model itself comes from the driver
# Then build the emulator again (./build-linux.sh: CMake finds _Build/deps/ffx-vk-linux), and pick "FSR 4" as the
# upscaler (launcher: Graphics page; or KYTY_UPSCALER=fsr KYTY_FSR4=1).
#
# Needs git, cmake, a C++20 compiler, python3, pkg-config and the development packages of Vulkan (headers, loader and
# vk.xml), OpenSSL, libdrm with libdrm_amdgpu, and SPIRV-Tools, e.g. on Debian/Ubuntu:
#   sudo apt-get install git cmake g++ python3 pkg-config libvulkan-dev vulkan-headers libssl-dev libdrm-dev \
#        spirv-tools libspirv-tools-dev spirv-headers
# and on Arch: sudo pacman -S --needed git cmake gcc python pkgconf vulkan-headers vulkan-icd-loader openssl libdrm \
#        spirv-tools spirv-headers
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
EMU="${1:-${KYTY_BUILD_DIR:-$ROOT/_Build/linux}}"
SRC="$ROOT/_Build/deps/ffx-vk-linux"
BUILD="$SRC/build"
REPO="${FFX_VK_LINUX_REPO:-https://github.com/DuIslingr/ffx-vk-linux}"
# The ffx-vk-linux commit this builds: code from outside this repository that loads AMD's DLLs into the emulator's process,
# so it is pinned, not "whatever the default branch has today". This is the one the OpenSSL patch below was made against
# and that was built with this emulator. FFX_VK_LINUX_REF=<full commit id, tag or branch> tries another (a newer one: then
# the patch may not apply, and nothing here has been tested with it).
PINNED_REF=4d339b8d4540744d9c0d0c6b961074f1f266867c
REF="${FFX_VK_LINUX_REF:-$PINNED_REF}"

for tool in git cmake c++ python3 pkg-config; do
	command -v "$tool" >/dev/null 2>&1 || { echo "$tool not found (see the top of this script)." >&2; exit 1; }
done
if [ "$(uname -m)" != "x86_64" ]; then
	echo "ffx-vk-linux loads x86-64 Windows DLLs: this needs Linux on x86-64." >&2
	exit 1
fi

# The SPIR-V headers are a separate package on some distributions, and CMake does not look for them.
if ! printf '#include <spirv/unified1/spirv.hpp11>\n' | c++ -x c++ -std=c++20 -fsyntax-only - 2>/dev/null; then
	echo "spirv/unified1/spirv.hpp11 not found: install the SPIR-V headers (Debian/Ubuntu: spirv-headers, Arch: spirv-headers)." >&2
	exit 1
fi

# Fetch exactly REF (GitHub serves a full commit id as well as a branch or tag) and check it out detached; an existing
# checkout is moved to it, its earlier patch dropped (applied again below). No pull: nothing moves on its own.
if [ -d "$SRC/.git" ]; then
	echo "Updating $SRC"
	git -C "$SRC" checkout -q -- .
	git -C "$SRC" remote set-url origin "$REPO"
else
	mkdir -p "$SRC"
	git -C "$SRC" init -q
	git -C "$SRC" remote add origin "$REPO"
fi
git -C "$SRC" fetch -q --depth 1 origin "$REF"
git -C "$SRC" checkout -q --detach FETCH_HEAD
HEAD_ID="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$REF" =~ ^[0-9a-f]{40}$ && "$HEAD_ID" != "$REF" ]]; then
	echo "ffx-vk-linux: expected $REF, got $HEAD_ID" >&2
	exit 1
fi
echo "ffx-vk-linux $HEAD_ID"
if [ "$HEAD_ID" != "$PINNED_REF" ]; then
	echo "NOTE: this is not the commit this was tested with ($PINNED_REF)." >&2
fi

# OpenSSL 4 made its ASN.1 structures opaque (and X509_get_X509_PUBKEY's result const): ffx-vk-linux's signature check read
# their members directly and does not build there. The patch (kept here: tools/local/fsr4/ffx-vk-linux-openssl4.patch) uses
# the accessor functions, which OpenSSL 1.1 and 3 have too. Skipped when the sources already have it.
FIX="$ROOT/tools/local/fsr4/ffx-vk-linux-openssl4.patch"
if [ -f "$FIX" ] && git -C "$SRC" apply --check "$FIX" 2>/dev/null; then
	git -C "$SRC" apply "$FIX"
	echo "Applied $(basename "$FIX") to ffx-vk-linux"
else
	echo "NOTE: $(basename "$FIX") not applied (the sources already have it, or differ from the pinned commit)." >&2
fi

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target ffx_vk_linux --parallel

mkdir -p "$EMU/fsr4"
cp -f "$BUILD/libffx_vk_linux.so" "$EMU/fsr4/libffx_vk_linux.so"
echo
echo "Built: $EMU/fsr4/libffx_vk_linux.so"
missing=0
for dll in amd_fidelityfx_loader_vk.dll amd_fidelityfx_upscaler_vk.dll amdvkffx64.dll; do
	if [ -f "$EMU/fsr4/$dll" ]; then
		echo "  found   $dll"
	else
		echo "  missing $dll"
		missing=1
	fi
done
if [ "$missing" = 1 ]; then
	echo "Copy the missing DLLs into $EMU/fsr4 (they are AMD's, not downloaded here)."
fi
if [ ! -f "$ROOT/_Build/linux/build.ninja" ] || ! grep -q "KYTY_FSR4_LINUX" "$ROOT/_Build/linux/build.ninja" 2>/dev/null; then
	echo "The emulator needs building again to include FSR 4: ./build-linux.sh (re-run cmake first if it was configured before: cmake -S . -B _Build/linux)."
fi
