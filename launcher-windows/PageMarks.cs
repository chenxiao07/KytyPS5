// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
namespace KytyLauncher {
// The square and triangle page-switch marks at the bottom of the inventory and status panels: two 66x66
// R8G8B8A8 sprites inside the UI sheet named in page-marks.txt, rewritten in place at the file offsets
// listed in page-marks-offsets.bin (square region then triangle region, row-major).
static class PageMarks {
	public const int Side = 66;
	public const string BackupName = "page-marks.bin";
	public const string PendingName = "page-marks.pending";
	public const string RecordName = "page-marks.generated";
	const int Regions = 2;
	static int[] offsets;
	static string relativePath, originalHash;
	static long size;
	static byte[] Resource(string name) {
		using (var input = typeof(PageMarks).Assembly.GetManifestResourceStream("KytyLauncher.KeyboardPrompts." + name)) {
			if (input == null)
				throw new IOException("Missing prompt texture resource: " + name);
			using (var output = new MemoryStream()) { input.CopyTo(output); return output.ToArray(); }
		}
	}
	// Loaded once, on first use, so a damaged resource is reported as an IOException like other prompt failures.
	static void Load() {
		if (offsets != null)
			return;
		// One line: relative sheet path, expected file size, SHA-256 of the original region bytes.
		var fields = Encoding.ASCII.GetString(Resource("page-marks.txt")).TrimEnd('\r', '\n').Split(' ');
		long length;
		// The relative path is checked before any Path call: lowercase letters, digits, '_', '.' and '/' only.
		if (fields.Length != 3 || !Regex.IsMatch(fields[0], @"^[a-z0-9_./]+\z") || fields[0].Contains("//") || fields[0].StartsWith("/") ||
			fields[0].Split('/').Contains("..") || !long.TryParse(fields[1], NumberStyles.None, CultureInfo.InvariantCulture, out length) ||
			fields[2].Length != 64 || !fields[2].All(c => (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
			throw new IOException("Invalid page marks resource");
		var data = Resource("page-marks-offsets.bin");
		if (data.Length != Regions * Side * Side * 4)
			throw new IOException("Invalid page marks layout");
		var result = new int[Regions * Side * Side];
		Buffer.BlockCopy(data, 0, result, 0, data.Length);
		// Pixel data starts after the 65-byte header; every pixel is 4 bytes.
		if (result.Distinct().Count() != result.Length || result.Any(x => x < 65 || (x - 65) % 4 != 0 || x > length - 4))
			throw new IOException("Invalid page marks pixel offset");
		relativePath = fields[0];
		size = length;
		originalHash = fields[2];
		offsets = result;
	}
	public static string OriginalHash { get { Load(); return originalHash; } }
	public static string RelativePath { get { Load(); return relativePath; } }
	public static string AtlasPath(GameInfo game) {
		Load();
		return Path.Combine(game.Path, relativePath.Replace('/', Path.DirectorySeparatorChar));
	}
	public static bool Exists(GameInfo game) { return File.Exists(AtlasPath(game)); }
	public static bool SizeMatches(string path) {
		Load();
		return new FileInfo(path).Length == size;
	}
	[StructLayout(LayoutKind.Sequential)]
	struct FileInformation {
		public uint Attributes;
		public System.Runtime.InteropServices.ComTypes.FILETIME Created, Accessed, Written;
		public uint VolumeSerial, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
	}
	[DllImport("kernel32.dll", SetLastError = true)]
	static extern bool GetFileInformationByHandle(Microsoft.Win32.SafeHandles.SafeFileHandle file, out FileInformation information);
	// True when an untouched sheet can be written in place: it opens for writing and has no second name on the
	// disk (a hard link), since an in-place write would change that other file too.
	public static bool Startable(string path) {
		try {
			using (var file = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.Read)) {
				FileInformation information;
				return GetFileInformationByHandle(file.SafeFileHandle, out information) && information.Links == 1;
			}
		}
		catch (Exception error) when (error is IOException || error is UnauthorizedAccessException) { return false; }
	}
	// True when the sheet, or a folder between it and the game folder, is a link (a symlink or a junction).
	public static bool Linked(GameInfo game) {
		var root = Path.GetFullPath(game.Path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
		for (var current = Path.GetFullPath(AtlasPath(game)); current != null && current.Length > root.Length; current = Path.GetDirectoryName(current))
			if ((File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
				return true;
		return false;
	}
	// The two spots of a sheet the launcher can look at, or null when the sheet is absent, linked, of another size
	// or cannot be read. "gone" says there is no sheet of the expected size here at all.
	public static byte[] Spots(GameInfo game, out bool gone) {
		gone = false;
		try {
			if (!Exists(game)) { gone = true; return null; }
			if (Linked(game))
				return null;
			var path = AtlasPath(game);
			if (!SizeMatches(path)) { gone = true; return null; }
			return ReadRegions(path);
		}
		catch (Exception error) when (error is IOException || error is UnauthorizedAccessException) { return null; }
	}
	// Each label is drawn like a 128-pixel icon and resampled like the icons' mips; R,G,B,A, square then triangle.
	public static byte[] Render(string squareLabel, string triangleLabel) {
		var result = new byte[Regions * Side * Side * 4];
		int at = 0;
		foreach (var label in new[] { squareLabel, triangleLabel })
			using (var canvas = PromptTexture.Draw(label))
			using (var mark = new Bitmap(Side, Side, PixelFormat.Format32bppArgb)) {
				using (var g = Graphics.FromImage(mark)) { g.CompositingMode = CompositingMode.SourceCopy; g.InterpolationMode = InterpolationMode.HighQualityBicubic; g.DrawImage(canvas, 0, 0, Side, Side); }
				for (int y = 0; y < Side; y++)
					for (int x = 0; x < Side; x++) {
						var c = mark.GetPixel(x, y);
						result[at++] = c.R;
						result[at++] = c.G;
						result[at++] = c.B;
						result[at++] = c.A;
					}
			}
		return result;
	}
	static IOException UnexpectedSize() {
		return new IOException("Unrecognized or modified prompt: global_pages.ctxr has an unexpected size.");
	}
	// Only a read of the two spots, so it shares the sheet with any other open handle.
	public static byte[] ReadRegions(string path) {
		Load();
		var result = new byte[offsets.Length * 4];
		using (var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete)) {
			if (file.Length != size)
				throw UnexpectedSize();
			for (int i = 0; i < offsets.Length; i++) {
				file.Position = offsets[i];
				if (file.Read(result, i * 4, 4) != 4)
					throw UnexpectedSize();
			}
		}
		return result;
	}
	// One pass over the open sheet, flushed to disk; other programs may read it meanwhile, and one that holds it
	// without sharing writes makes the open fail. "opened" runs once the sheet is open for writing and before its
	// first byte changes. A failed batch restores the previous bytes the same way.
	public static void WriteRegions(string path, byte[] bytes, Action opened = null) {
		Load();
		if (bytes.Length != offsets.Length * 4)
			throw new IOException("Invalid page marks data");
		using (var file = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.Read)) {
			if (file.Length != size)
				throw UnexpectedSize();
			if (opened != null)
				opened();
			for (int i = 0; i < offsets.Length; i++) {
				file.Position = offsets[i];
				file.Write(bytes, i * 4, 4);
			}
			file.Flush(true);
		}
	}
}
}
