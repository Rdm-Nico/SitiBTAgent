import sys
import os

# Aggiunge la directory root del progetto al path di Python per trovare i moduli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from whatsapp_agent_tecnico import Whatsapp_Agent as WA_Tecnico
from whatsapp_agent_follow_up import Whatsapp_Agent as WA_follow
from utils.configurator import Configurator
from utils.logger import Logger
import argparse


logger = Logger(save=True).getLogger()
if __name__ == "__main__":
    # add the config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file", help="Path to config file", default="config.yaml"
    )
    args = parser.parse_args()
    configurator = Configurator(file=args.conf_file)

    conf = configurator.get_file()

    logger.info("=" * 50)
    logger.info(f"Avvio SCHEDULER MODE FOR APP {conf.app}")
    logger.info("=" * 50)

    if conf.app.name == "tecnico":
        agent = WA_Tecnico(conf, is_scheduler=True)
    elif conf.app.name == "follow-up":
        agent = WA_follow(conf, is_scheduler=True)
    else:
        logger.error(f"nome dell applicativo non valido: {conf.app.name} può essere solo 'tecnico' oppure 'follow-up")
        exit
    
    # dopo aver completato start() -> dovrebbe in teoria uscire da solo dal seguente loop
    while(agent.is_running()):
        continue