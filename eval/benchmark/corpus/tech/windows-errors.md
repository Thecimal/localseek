# Windows errors

Error 0x80070057 ("the parameter is incorrect") during updates or backups: open an elevated prompt and run sfc /scannow, then DISM /Online /Cleanup-Image /RestoreHealth, and reboot. If the update still fails, reset the update components and try again.
