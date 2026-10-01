# Backup policy

Follow the 3-2-1 rule: three copies of the data, on two different media, with one copy offsite. Retention is set with restic forget --keep-daily 7 --keep-weekly 4 --keep-monthly 12 followed by a prune. Test a restore every quarter, because an untested backup is only a hope. Phone photos are synced to the NAS weekly.
