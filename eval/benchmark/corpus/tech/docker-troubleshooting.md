# Docker troubleshooting

Error "permission denied while trying to connect to the Docker daemon socket": add your user to the docker group with sudo usermod -aG docker $USER, then log out and back in. Error "port is already allocated": find the process holding the port with ss -ltnp and stop it, or map the container to a different host port. If images fill the disk, run docker system prune after checking which volumes you still need.
