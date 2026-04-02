"""File per fare il run del bot whatsapp per i tecnici"""
from whatsapp_agent_tecnico import Whatsapp_Agent as WA_tecnico
from whatsapp_agent_follow_up import Whatsapp_Agent as WA_follow
from utils.configurator import Configurator
from utils.logger import Logger
import argparse
import os 
logger = Logger(save=True).getLogger()

# Configurazione globale (caricata una volta)
configurator = None
conf = None
agent = None
app = None

def create_app():
    """Factory function per creare l'app"""
    global configurator, conf, agent, app
    logger.info("dentro al factory method")

    conf_file = os.getenv('APP_CONFIG', 'config.yaml')
    
    if configurator is None:
        configurator = Configurator(file=conf_file)
        conf = configurator.get_file()
        if conf.app.name == "tecnico":
            agent = WA_tecnico(conf)
        else:
            agent = WA_follow(conf)

    return agent.app

# Chiama la factory per inizializzare l'app
app = create_app()

# Per usare con Gunicorn:
# gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 run_production:create_app()
# ma in produzione si userà il file run.py