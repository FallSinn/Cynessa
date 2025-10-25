using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace GameLauncher;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        var config = LauncherConfiguration.Load("appsettings.json");
        Application.Run(new LauncherForm(config));
    }
}

public sealed class LauncherForm : Form
{
    private readonly LauncherConfiguration _config;
    private readonly TextBox _logBox = new() { Multiline = true, ScrollBars = ScrollBars.Vertical, ReadOnly = true, Dock = DockStyle.Bottom, Height = 160 };

    public LauncherForm(LauncherConfiguration config)
    {
        _config = config;
        Text = "Cynessa Portable Launcher";
        Width = 480;
        Height = 360;
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;

        var playButton = CreateButton("Play", 20, (s, e) => LaunchGame());
        var settingsButton = CreateButton("Settings", 60, (s, e) => OpenSettings());
        var verifyButton = CreateButton("Verify Files", 100, (s, e) => VerifyFiles());
        var updateButton = CreateButton("Update", 140, (s, e) => ApplyUpdates());
        var savesButton = CreateButton("Open Saves", 180, (s, e) => OpenSaves());
        var quitButton = CreateButton("Quit", 220, (s, e) => Close());

        Controls.AddRange(new Control[] { playButton, settingsButton, verifyButton, updateButton, savesButton, quitButton, _logBox });
    }

    private Button CreateButton(string text, int top, EventHandler handler)
    {
        var button = new Button
        {
            Text = text,
            Left = 20,
            Top = top,
            Width = 180,
            Height = 32
        };
        button.Click += handler;
        return button;
    }

    private void LaunchGame()
    {
        try
        {
            Log("Launching game...");
            var startInfo = new ProcessStartInfo
            {
                FileName = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.PythonExecutable)),
                Arguments = $"-I -s -S \"{_config.Game.GameScript}\"",
                WorkingDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.WorkingDirectory)),
                UseShellExecute = false,
            };
            Process.Start(startInfo);
        }
        catch (Exception ex)
        {
            Log($"Failed to launch: {ex.Message}");
        }
    }

    private void OpenSettings()
    {
        try
        {
            var path = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.SettingsFile));
            if (!File.Exists(path))
            {
                Log("Settings file not found");
                return;
            }
            Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
        }
        catch (Exception ex)
        {
            Log($"Failed to open settings: {ex.Message}");
        }
    }

    private void OpenSaves()
    {
        try
        {
            var path = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.SaveDirectory));
            Directory.CreateDirectory(path);
            Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
        }
        catch (Exception ex)
        {
            Log($"Failed to open saves: {ex.Message}");
        }
    }

    private void VerifyFiles()
    {
        try
        {
            var manifestPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.ManifestFile));
            if (!File.Exists(manifestPath))
            {
                Log("Manifest file missing");
                return;
            }
            var manifest = JsonSerializer.Deserialize<FileManifest>(File.ReadAllText(manifestPath));
            if (manifest is null)
            {
                Log("Unable to parse manifest");
                return;
            }

            var baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.WorkingDirectory));
            var invalid = new List<string>();
            foreach (var entry in manifest.Files)
            {
                var path = Path.Combine(baseDir, entry.Path.Replace('/', Path.DirectorySeparatorChar));
                if (!File.Exists(path))
                {
                    invalid.Add($"Missing: {entry.Path}");
                    continue;
                }
                using var stream = File.OpenRead(path);
                var hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
                if (!string.Equals(hash, entry.Sha256, StringComparison.OrdinalIgnoreCase))
                {
                    invalid.Add($"Mismatch: {entry.Path}");
                }
            }
            Log(invalid.Count == 0 ? "All files validated." : string.Join(Environment.NewLine, invalid));
        }
        catch (Exception ex)
        {
            Log($"Verification failed: {ex.Message}");
        }
    }

    private void ApplyUpdates()
    {
        try
        {
            var manifestPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.UpdateManifest));
            if (!File.Exists(manifestPath))
            {
                Log("Update manifest missing");
                return;
            }
            var update = JsonSerializer.Deserialize<UpdateManifest>(File.ReadAllText(manifestPath));
            if (update is null)
            {
                Log("Unable to parse update manifest");
                return;
            }
            var baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _config.Game.WorkingDirectory));
            foreach (var file in update.Files)
            {
                var source = Path.Combine(Path.GetDirectoryName(manifestPath) ?? string.Empty, file.Source ?? file.Path);
                var destination = Path.Combine(baseDir, file.Path.Replace('/', Path.DirectorySeparatorChar));
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                File.Copy(source, destination, overwrite: true);
            }
            Log($"Applied update version {update.Version}");
        }
        catch (Exception ex)
        {
            Log($"Update failed: {ex.Message}");
        }
    }

    private void Log(string message)
    {
        _logBox.AppendText($"[{DateTime.Now:T}] {message}{Environment.NewLine}");
    }
}

public sealed class LauncherConfiguration
{
    public GameConfiguration Game { get; set; } = new();

    public static LauncherConfiguration Load(string path)
    {
        var fullPath = Path.Combine(AppContext.BaseDirectory, path);
        var json = File.ReadAllText(fullPath);
        return JsonSerializer.Deserialize<LauncherConfiguration>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }) ?? new LauncherConfiguration();
    }
}

public sealed class GameConfiguration
{
    public string PythonExecutable { get; set; } = string.Empty;
    public string GameScript { get; set; } = string.Empty;
    public string WorkingDirectory { get; set; } = string.Empty;
    public string SettingsFile { get; set; } = string.Empty;
    public string ManifestFile { get; set; } = string.Empty;
    public string UpdateManifest { get; set; } = string.Empty;
    public string SaveDirectory { get; set; } = "../game/saves";
}

public sealed class FileManifest
{
    public List<FileManifestEntry> Files { get; set; } = new();
}

public sealed class FileManifestEntry
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("sha256")]
    public string Sha256 { get; set; } = string.Empty;
}

public sealed class UpdateManifest
{
    public string Version { get; set; } = "0.0.0";
    public List<UpdateFileEntry> Files { get; set; } = new();
}

public sealed class UpdateFileEntry
{
    [JsonPropertyName("path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("source")]
    public string? Source { get; set; }
}
