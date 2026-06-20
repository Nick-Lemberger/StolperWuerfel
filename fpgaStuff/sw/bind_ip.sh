#!/bin/bash

echo "Setting up network adapter enp0s25 and binding static IP address..."

sudo ip addr add 192.168.178.10/24 dev enp0s25
sudo ip link set enp0s25 up

echo "Done."
