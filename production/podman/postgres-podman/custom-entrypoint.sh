#!/bin/bash

# Entrypoint costum per fare il drop della tabelle commesse e followups ogni volta che il container postgres si avvia/riavvia.

set -e 

# eseguire la manutenzione
run_maintenance() {
    echo "Waiting postgres to be ready..."
    until pg_isready -U postgres > /dev/null 2>&1; do
        sleep 1
    done

    echo "running maintenance..."
    psql -U postgres -f ./maintenance.sql
    echo "maintenance completed"
}

# esegui manutenzione(background) e entrypoint originale con gli argomenti corretti
run_maintenance & 
exec /usr/local/bin/docker-entrypoint.sh "$@"
 
