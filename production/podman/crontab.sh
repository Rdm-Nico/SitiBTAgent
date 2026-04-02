#!/bin/bash


# script per schedulare il podman-compose e avviarlo ogni giorno.
# guardare il README.md per una guida completa

# Imposta il PATH 
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
# dobbiamo entrare nella directory attraverso l'abs path
cd /home/tecnico/SitiBT-Agent-AI/production/podman || exit 1
# trova l'eseguibile di podman-compose
PODMAN_COMPOSE_CMD=$(which podman-compose 2>&1)

# podman-compose dei servizi per il tecnico
$PODMAN_COMPOSE_CMD restart whatsapp-agent-tecnico statsd-exporter ngrok postgres

# podman-compose dei servizi per il commerciale
#$PODMAN_COMPOSE_CMD start whatsapp-agent-follow-up

