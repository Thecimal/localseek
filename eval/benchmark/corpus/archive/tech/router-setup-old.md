# Router setup (previous provider)

Last updated 2022. Admin page: 192.168.0.1. The main network Maple-Home uses WPA2. Guest network: SSID Maple-Guest, password green-heron-17, bandwidth limit 10 Mbps. Change the admin password every year and check for firmware updates on the first of the month.

The ISP modem runs in bridge mode and the router handles all routing. DHCP reservations are made for the printer and the NAS so their addresses never change. Port forwarding is not used. Remote administration is disabled, and the router keeps its logs for seven days. When replacing the router, export the configuration first and store the file in the password manager notes. The main network and the guest network use separate passphrases, and both are rotated after any visitor stays longer than a week.
