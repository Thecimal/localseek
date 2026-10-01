# Certificate errors

ERR_CERT_DATE_INVALID means the certificate has expired or the device clock is wrong, so check the system date first. NET::ERR_CERT_AUTHORITY_INVALID usually points to a self-signed certificate or a missing intermediate. To renew a Let's Encrypt certificate run sudo certbot renew, and test it first with certbot renew --dry-run. A cron entry runs the renewal twice a day, so certificates renew by themselves about thirty days before they expire.
