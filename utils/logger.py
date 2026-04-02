import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pytz import timezone
import os
import sys
"""
File per creare un logger (singleton)
"""


class ColorFormatter(logging.Formatter):
    """
    Classe per colorare qualche log level
    """
    red = "\x1b[31;20m"
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    yellow = "\x1b[33;20m"


    
    

    def __init__(self, fmt = None, datefmt = None, style = "%", validate = True, *, defaults = None):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        # define formats
        self.formats = {
        logging.DEBUG: self.green + fmt + self.reset,
        logging.INFO: self.grey + fmt + self.reset,
        logging.ERROR: self.red + fmt + self.reset,
        logging.WARNING: self.yellow + fmt + self.reset
        }
    
    def tz_converter(self, timestamp):
        """Funzione per aggiungere un datetime aware"""
        dt = datetime.fromtimestamp(timestamp)
        tzinfo = timezone('Europe/Rome')
        return tzinfo.localize(dt)
    
    def formatTime(self, record, datefmt=None):
        dt = self.tz_converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)



class Logger:
    _instance  = None
    _initialized = False
    _is_main_configured = False  # flag per sapere se il __main__ è configurato

    def __new__(cls, save:bool=True, consoleLevel:str="DEBUG", fileLevel:str="INFO"):
        """
        Metodo chiamato prima di __init__ per garantire che esiste solo un istanza del Logger
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def _get_caller_module():
        """
        Identifica il modulo che ha chiamato Logger()
        """
        import inspect
        try:
            # Risali lo stack per trovare il chiamante
            for frame_info in inspect.stack():
                frame_globals = frame_info.frame.f_globals
                module_name = frame_globals.get('__name__')
                # Salta i frame interni della classe Logger
                if module_name and module_name != Logger.__module__:
                    return module_name == '__main__'
        except Exception:
            return False
        return False
    
    
    def __init__(self, save:bool=True, consoleLevel:str="DEBUG", fileLevel:str="INFO"):
        """
        Accetta:
            - save: se salvare oppure no il log
            - consoleLevel: il log level della console
            - fileLevel: il log level del file 

        Log level accettati: 
            - DEBUG
            - INFO
            - WARNING
            - ERROR
            - CRITICAL
        """

        is_main = self._get_caller_module()
        # se esiste non lo crea nuovamente 
        if Logger._initialized:
            # se chiama dal main e non è configurato
            if is_main and not Logger._is_main_configured:
                Logger._is_main_configured = True
                self._reconfigure(save, consoleLevel, fileLevel)
            return 
        
        # prima inizializzazione -> procedere normalmente

        Logger._initialized = True
        if is_main:
            Logger._is_main_configured = True

        self._configure(save, consoleLevel, fileLevel)
        

    def _configure(self, save: bool, consoleLevel: str, fileLevel: str):
        """Configurazione del logger per la prima volta"""

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.logger.setLevel("DEBUG")

        # salva i parametri
        self.save = save 
        self.consoleLevel = consoleLevel
        self.fileLevel = fileLevel

        # add formatters
        colorformatter = ColorFormatter(
            "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )

        fileformatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )
        # add handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(consoleLevel)
        if save:
            # check if the folder logs exists
            if not os.path.exists("./logs/"):
                os.makedirs("./logs/")
            # file handler con auto rotate 
            file_handler = TimedRotatingFileHandler(filename="./logs/app.log", encoding="utf-8", when="W0", backupCount=3)
            file_handler.setLevel(fileLevel)
            file_handler.setFormatter(fileformatter)
        
        console_handler.setFormatter(colorformatter)
        # add all
        self.logger.addHandler(console_handler)
        if save:
            self.logger.addHandler(file_handler)

    def _reconfigure(self, save: bool, consoleLevel: str, fileLevel: str):
        """
        Riconfigurazione del logger rimuovendo gli headers e ricreandoli
        """
        self.logger.handlers.clear()

        self._configure(save,consoleLevel,fileLevel)


    def getLogger(self):
        """
        Return the logger
        """
        return self.logger