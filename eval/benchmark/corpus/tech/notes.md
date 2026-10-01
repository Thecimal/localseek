# Home network notes

Subnet 192.168.1.0/24. Static assignments: router 192.168.1.1, NAS 192.168.1.20, Pi-hole DNS 192.168.1.53, printer 192.168.1.60. The DHCP pool is .100 to .199. Smart-home gadgets live on a separate VLAN 30 so they cannot see the laptops. Pi-hole blocks ads for the whole house and forwards everything else to the upstream resolver 1.1.1.1.
