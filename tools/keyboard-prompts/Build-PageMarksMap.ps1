#Requires -Version 7.5
# Generate the page-marks pixel map for global_pages.ctxr (the square and triangle page-switch marks) with Aqua Toolset.
# Marker round trip: every pixel of the sheet stores its own index, Aqua decodes it, and the two 66x66 regions are read back.
# Only file offsets, the sheet size and a hash are emitted; -OriginalOut also writes the original region bytes for the
# local tests (tests/KeyboardPromptsTests.cs). Never commit that file.
# Expects a sheet whose two marks are original; it refuses when the launcher's saved copy beside the game says they are not.
# Needs PowerShell 7.5 or later (the Aqua Toolset build it loads targets .NET 9).
param([Parameter(Mandatory)][string]$AquaToolsetDirectory,
      [Parameter(Mandatory)][string]$Atlas,
      [Parameter(Mandatory)][string]$OutputDirectory,
      [string]$OriginalOut)
$ErrorActionPreference='Stop'
foreach ($name in @('Reloaded.Memory','DrSwizzler','AquaModelLibrary.Helpers','AquaModelLibrary.Data','ZstdNet','ICSharpCode.SharpZipLib')) {
    [Reflection.Assembly]::LoadFrom((Join-Path $AquaToolsetDirectory "$name.dll")) | Out-Null
}
Add-Type -TypeDefinition @'
public static class PageMarksMap {
    // Every 4-byte pixel at d + 4g stores g.
    public static byte[] Markers(byte[] src, int d, int count) {
        var b = (byte[])src.Clone();
        for (int g = 0; g < count; g++) { int at = d + 4 * g; b[at] = (byte)g; b[at + 1] = (byte)(g >> 8); b[at + 2] = (byte)(g >> 16); b[at + 3] = (byte)(g >> 24); }
        return b;
    }
    // Decoded pixels whose marker is outside [lo, lo + n) or repeated (0 = exact permutation).
    public static long NotPermutation(byte[] dec, long lo, int n) {
        var seen = new bool[n]; long bad = 0;
        for (int p = 0; p < n; p++) { long r = (long)System.BitConverter.ToUInt32(dec, p * 4) - lo; if (r < 0 || r >= n || seen[r]) { bad++; continue; } seen[r] = true; }
        return bad;
    }
    // Pixels where the decoded sheet differs from the raw file bytes at the mapped offset.
    public static long Mismatches(byte[] realDec, byte[] markDec, byte[] raw, int d) {
        long bad = 0;
        for (int p = 0; p < realDec.Length / 4; p++) {
            long at = d + 4L * System.BitConverter.ToUInt32(markDec, p * 4);
            for (int k = 0; k < 4; k++) if (raw[at + k] != realDec[p * 4 + k]) { bad++; break; }
        }
        return bad;
    }
    public static int[] Map(byte[] markDec, int width, int d, int x0, int y0, int side) {
        var r = new int[side * side]; int i = 0;
        for (int y = y0; y < y0 + side; y++) for (int x = x0; x < x0 + side; x++) r[i++] = d + 4 * (int)System.BitConverter.ToUInt32(markDec, (y * width + x) * 4);
        return r;
    }
}
'@
$C=[AquaModelLibrary.Data.BluePoint.CTXR.CTXR]
$raw=[IO.File]::ReadAllBytes($Atlas)
$tex=$C::new($raw)
$W=$C::GetDesResolutionComponent($tex.WidthBaseByte,$tex.WidthMultiplierByte,0xC0)
$H=$C::GetDesResolutionComponent($tex.HeightBaseByte,$tex.HeightMultiplierByte,0x80)
# Five 4096x4096 R8G8B8A8 layers, one mip each; the layer Aqua lists last (index 4) is stored first, after the 65-byte header.
$N=$W*$H; $D=[int]($raw.Length-5*[long]$N*4-12)
if ($tex.textureFormat -ne 0 -or $W -ne 4096 -or $H -ne 4096 -or $tex.mipMapsList.Count -ne 5 -or @($tex.mipMapsList|?{$_.Count -ne 1}).Count -ne 0 -or
    $D -ne 65 -or $tex.bufferOffsets[4] -ne 65) { throw 'Unsupported global_pages.ctxr' }
$real=$tex.mipMapsList[4][0]; $tex=$null; [GC]::Collect()
$marked=$C::new([PageMarksMap]::Markers($raw,$D,5*$N)); $mark=$marked.mipMapsList[4][0]; $marked=$null; [GC]::Collect()
if ([PageMarksMap]::NotPermutation($mark,0,$N) -ne 0) { throw 'Markers are not a permutation of the first layer' }
if ([PageMarksMap]::Mismatches($real,$mark,$raw,$D) -ne 0) { throw 'Decoded sheet does not match the mapped file bytes' }
# Square then triangle, row-major. Positions in the decoded layer.
$offsets=[Collections.Generic.List[int]]::new()
# The two regions were found by decoding the sheet with Aqua and taking the alpha bounds of the square-in-circle and
# triangle-in-circle sprites in the layer Aqua lists as index 4; a scan of all five layers found no other copy.
foreach ($x in @(2372,2450)) { $offsets.AddRange([PageMarksMap]::Map($mark,$W,$D,$x,2946,66)) }
$offsets=$offsets.ToArray()
if ($offsets.Length -ne 8712 -or ($offsets|Sort-Object -Unique).Count -ne 8712 -or @($offsets|?{$_ -lt 65 -or $_+4 -gt 65+[long]$N*4 -or ($_-65)%4 -ne 0}).Count -ne 0) { throw 'Invalid page-marks offsets' }
$original=[byte[]]::new($offsets.Length*4)
for ($i=0;$i -lt $offsets.Length;$i++) { [Array]::Copy($raw,$offsets[$i],$original,$i*4,4) }
# A sheet the launcher has relabeled would give the wrong original: its saved copy is in the game folder, four levels
# above the sheet's own folder (ui/textures/_ps5/_ps5).
$game=(Get-Item -LiteralPath $Atlas).Directory
foreach ($level in 1..4) { if ($game) { $game=$game.Parent } }
if ($game) {
    $saved=Join-Path $game.FullName '.kyty-keyboard-prompts-v1/page-marks.bin'
    if ((Test-Path -LiteralPath $saved -PathType Leaf) -and -not [Linq.Enumerable]::SequenceEqual([byte[]][IO.File]::ReadAllBytes($saved),$original)) {
        throw "The two marks in this sheet are not the original ones (the launcher's saved copy beside the game differs). Turn Keyboard and mouse prompts off, press Play once, then run this script again."
    }
}
$hash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($original)).ToLowerInvariant()
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
function Gzip($name,[byte[]]$bytes){
    $file=[IO.File]::Create((Join-Path $OutputDirectory $name))
    $zip=[IO.Compression.GZipStream]::new($file,[IO.Compression.CompressionMode]::Compress)
    try{$zip.Write($bytes,0,$bytes.Length)}finally{$zip.Dispose();$file.Dispose()}
}
$map=[byte[]]::new($offsets.Length*4);[Buffer]::BlockCopy($offsets,0,$map,0,$map.Length)
Gzip 'page-marks-offsets.bin.gz' $map
Gzip 'page-marks.txt.gz' ([Text.Encoding]::ASCII.GetBytes("ui/textures/_ps5/_ps5/global_pages.ctxr $($raw.Length) $hash`n"))
if ($OriginalOut) { [IO.File]::WriteAllBytes($OriginalOut,$original) }
Write-Output "Generated $($offsets.Length) page-marks offsets; original region sha256 $hash."
