# Home server

A small Ubuntu 22.04 mini PC running Nextcloud and Jellyfin. Data lives in /srv/data. Nightly backups run with restic to an external USB drive, and a weekly copy goes to the offsite NAS at a friend's house. If a backup job reports that the repository is locked, run restic unlock and start it again. Apply system updates with apt every other Sunday and reboot afterwards.
