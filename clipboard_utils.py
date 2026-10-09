import subprocess
import os
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QUrl, QMimeData

logger = logging.getLogger(__name__)

def _powershell_command(filepath):
    env = os.environ.copy()
    env["QUICK_AUDIO_RECORDER_PATH"] = filepath
    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Set-Clipboard -Path $env:QUICK_AUDIO_RECORDER_PATH",
    ]
    return command, env

def copy_file_to_clipboard(filepath):
    """
    Copies a file to the clipboard using PowerShell.
    This mimics the behavior of selecting a file in Explorer and pressing Ctrl+C.
    
    Fallback to Qt's clipboard mechanism if PowerShell fails.
    """
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        return False, "File does not exist"

    # Strategy 1: PowerShell (Native / Robust)
    # Set-Clipboard -Path "..." mimics Explorer copy perfectly.
    try:
        cmd, env = _powershell_command(filepath)
        
        # subprocess.run ensures we wait for it to finish
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        
        if result.returncode == 0:
            return True, "Copied via PowerShell"
        else:
            logger.error("PowerShell clipboard error: %s", result.stderr.strip())
            # Don't return yet, try Strategy 2
            
    except Exception as e:
        logger.exception("PowerShell clipboard execution failed")

    # Strategy 2: Qt (Fallback)
    try:
        data = QMimeData()
        url = QUrl.fromLocalFile(filepath)
        data.setUrls([url])
        QApplication.clipboard().setMimeData(data)
        return True, "Copied via Qt (Fallback)"
    except Exception as e:
        logger.exception("Qt clipboard fallback failed")
        return False, f"All methods failed. Qt Error: {e}"
