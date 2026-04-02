from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError
from utils.logger import Logger

logger = Logger(save=True).getLogger()
class Configurator:
    """
    Classe per convertire il file di configurazione YAML e controllare che siano presenti tutti i campi
    """
    def __init__(self, file:str, verify:bool=True):
        self.conf = OmegaConf.load(file)
        self.logger = logger
        if verify:
            self.check(file)
        
        

    def get_file(self) -> object:
        """
        Ritorna l'oggetto del configuratore
        """
        return self.conf
    def load_file(self, file:str):
        """
        Aggiunge il file di configurazione
        """
        self.conf = OmegaConf.load(file)

    def check(self, file):
        """
        Controllo che ogni campo sia specificato 
        """
        output = "\n"
        output += f"{'-' * 15} YAML file config:{file} {'-' * 15}\n"
        try:
            for group_name, group in self.conf.items():
                # convert to dict and interpolate the result
                dict_group = OmegaConf.to_container(group, resolve=True)
                output += f"modulo:{group_name}\n"
                for key, value in dict_group.items():
                    output += f"\t{key}:{value}\n"
                output += "\n"
            output += f"{'-' * 15} End of YAML file {'-' * 15}\n"
            self.logger.info(output)
        except InterpolationResolutionError as e:
            self.logger.exception(f"Errore: la variabile d'ambiente {e.full_key} non è stata settata")