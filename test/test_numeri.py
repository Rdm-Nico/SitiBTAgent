from utils.logger import Logger
from utils.util import process_tel_number
logger = Logger(save=False, consoleLevel="DEBUG").getLogger()



        

def Test():

    while(1):
        risposta = input("Scrivi un numero di telefono: ")
        if risposta == "/bye":
            break

        language, number = process_tel_number(risposta)

        if language == 'errore':
            logger.debug(f"errore nel nummero passato: {number}")
        else:
            logger.debug(f"il numero processato è : {number} con la lingua: {language}")
        



    
    

   
            
        
if __name__ == "__main__":
    Test()