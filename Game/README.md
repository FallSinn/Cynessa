# Cynessa Portable Shooter

Double-click `launcher/GameLauncher.exe` and press **Play** to start the game.

## Included directories
- `launcher/` – Windows launcher source and configuration. Publish with `dotnet publish` to produce the self-contained `GameLauncher.exe`.
- `runtime/` – Place the Python 3.11 embeddable distribution here (already wired with `sitecustomize.py`).
- `game/` – Python source, assets, configs, saves, mods, and logs.
- `updates/` – Offline update payloads and manifests.
- `tools/` – Packaging scripts (`build_zip.bat`, `verify_files.ps1`, `gen_manifest.py`, `installer.nsi`).
- `dist/` – Build output target for the ZIP and installer.

## Building the launcher
1. Install the .NET 8 SDK.
2. From `Game/launcher`, run:
   ```powershell
   dotnet publish -c Release -r win-x64 -p:PublishSingleFile=true -p:SelfContained=true -p:PublishTrimmed=true
   ```
3. Copy the produced `GameLauncher.exe` from `bin/Release/net8.0-windows/win-x64/publish/` into `Game/launcher/`.

## Packaging the portable build
Run `tools/build_zip.bat` from a Windows shell to produce `dist/Game_Portable.zip`.

## Generating file manifests
Use the Python script to calculate hashes:
```powershell
python tools/gen_manifest.py Game updates/file_manifest.json
```

## File verification
Execute `tools/verify_files.ps1` (PowerShell) to validate files against the manifest.

## Troubleshooting
- Install the latest Visual C++ Redistributable if SDL or Pygame DLLs report missing dependencies.
- Adjust resolution or toggle fullscreen by editing `game/config/settings.json`.
- Enable/disable VSync via the same settings file (`"vsync"`).

