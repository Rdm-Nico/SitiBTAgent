
from models.ModelProviderClient import clientRouter
from utils.logger import Logger
from pydantic import BaseModel
from schema.structured_ouput_model import Tecnico,FollowUp
from utils.util import clean_response
from utils.Tool import ExtractorTool, PushTool
from utils.configurator import Configurator
import argparse
import json
import sys
import os

logger = Logger(save=False, consoleLevel="DEBUG").getLogger()

# Aggiunge la directory root del progetto al path di Python per trovare i moduli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def extract_info(model,summary:str, structured_output:BaseModel=Tecnico):
    logger.debug(f"è stato chiamato il tool: extractor_expert con i seguenti parametri:(msg_to_send: {summary})")
    model_response = model.generate(prompt=summary, schema=structured_output)
    #logger.debug(f"extractor respose: {model_response}")
    clean_output = clean_response(model_response['content'])
    try:
        clean_json = json.loads(clean_output)
        predict = structured_output.model_validate_json(json.dumps(clean_json))

    except Exception as e:
        logger.error({"error": f"Errore: {str(e)}\n model_response: {model_response.choices[0].message.content},\n clean_output: {clean_output}"})
    return predict


def liveTest(conf, app:str):
    """
    Live demo dell'extractor agent con vLLM
    """
    # init client:
    client = clientRouter(conf, name="chat_model")
    client.add_tools(tools=[ExtractorTool(), PushTool()])
    

    # add the second agent 
    # init client:
    second_client = clientRouter(conf, name="extractor_model")

    if app == 'tecnico':
        while(1):
            print(f"\n**Arrivo messaggio su whatsapp**\n")
            print("""[AGENTE SITI]: Sollecito registrazione ore - Commessa 231447
                        Gentile collaboratore/collaboratrice,
                        dalla verifica delle registrazioni relative al giorno 05/10/2025 risulta che non siano state inserite le informazioni riguardanti la commessa n. 231447 del cliente Ferrari Spa.

                        Si richiede cortesemente di comunicare entro la giornata odierna i seguenti dati:
                        - Ore di lavoro ordinarie
                        - Ore straordinarie
                        - Ore di viaggio
                        - Eventuali inefficienze riscontrate

                        Le informazioni  devono essere inviate in un unico messaggio di risposta,  lo puoi inviare come messaggio di testo o anche messaggio audio.
                        Cordiali Saluti.
                  """)
            while(1):
                risposta = input("Scrivere per rispondere:")

                if risposta == '' or risposta == 'stop':
                    break
                
                
                response = client.chat(risposta, save_to_history=True)
                logger.debug(response)
                if response['tool_calls']:
                    funciton_name = response['tool_calls']['name']


                    if funciton_name == 'extractor_expert':
                        arguments = response['tool_calls']['arguments']
                        predict = extract_info(second_client, **arguments)

                        #logger.debug(f"l'esperto estrattore ha ritornato: {predict.model_dump_json()}")


                        # passiamo la risposta al primo modello

                        find_inefficienza = True if predict.durata_inefficienza != 0 or (predict.note_inefficienza != "" and predict.note_inefficienza != "null" and predict.note_inefficienza != "NULL") else False
                        if find_inefficienza:
                            output = f"Ho riconosciuto i seguenti campi:\nore ordinarie: *{predict.ore_ordinarie}*\nore straordinarie: *{predict.ore_straordinarie}*\nore di viaggio: *{predict.ore_viaggio}* \ninnefficienze: *{find_inefficienza}*\ndurata inefficienza: *{predict.durata_inefficienza}*\netichetta inefficienza: *ETICHETTA*\ndescrizione inefficienza: *{predict.note_inefficienza}*\nnote: *{predict.note}*\ncommessa: *{predict.commessa}*\n\nsono corretti?"
                        else:
                            output = f"Ho riconosciuto i seguenti campi:\nore ordinarie: *{predict.ore_ordinarie}*\nore straordinarie: *{predict.ore_straordinarie}*\nore di viaggio: *{predict.ore_viaggio}* \ninnefficienze: *{find_inefficienza}*\nnote: *{predict.note}*\ncommessa: *{predict.commessa}*\n\nsono corretti?"

                        client.save_2_conversation(output)
                    else:
                        # altra funzione per il push nel db
                        ris_push = response['tool_calls']['arguments']
                        tool_id = response['tool_calls']['id']
                        logger.debug(f"è stato chiamato il tool: push data con i seguenti parametri:(push: {ris_push})")
                        result = "La commessa non è stata salvata"
                        if bool(ris_push['push']) == True:
                            result = "La commessa è stata salvata"

                        logger.debug(f"la funzione push_data ha ritornato: {result}")
                        # passiamo la risposta al primo modello
                        response = client.chat(tool_name=funciton_name, tool_response=result, tool_id=tool_id, save_to_history=True)
                        output = response['content']
                else:
                    # non è stato chiamato il tool
                    output = response['content']

                print(output)


            ris =input(f"\n vuoi uscire?(N/Y):")
            # puliamo la memoria
            client.clear_history()
            if ris == 'Y':
                break
    
    if app == 'follow':
        while(1):
            print(f"\n**Arrivo messaggio su whatsapp**\n")
            print("""[AGENTE SITI]: Messaggio Follow-up - CodOfferta 231447
                        Gentile collaboratore/collaboratrice,
                        Ieri, 22/01/2025, era il 60esimo giorno per compilare l'update sulla offerta fatta dal cliente TalDeiTali s.r.l.
                        Il titolo della seguente offerta è Offerta per Levigatura + Squadratura

                        Si richiede cortesemente di comunicare entro la giornata odierna i seguenti dati:
                        - la probabilità di acquisizione da parte del cliente dell'offerta 
                        - la data di consegna prevista
                        - il prezzo di consegna previsto
                        - eventuali note AM da aggiungere
    
                        Le informazioni  possono essere inviate come messaggio di testo o anche messaggio audio.
                        Volendo si può compilare l'update direttamente cliccando il bottone sottostante 
                        Cordiali Saluti.
                  """)
            while(1):
                risposta = input("Scrivere per rispondere:")
    
                if risposta == '' or risposta == 'stop':
                    break
                
                
                response = client.chat(risposta, save_to_history=True)
                logger.debug(response)
                if response['tool_calls']:
                    funciton_name = response['tool_calls']['name']
                    
    
                    if funciton_name == 'extractor_expert':
                        arguments = response['tool_calls']['arguments']
                        if isinstance(arguments,str):
                            # dobbiamo provare a fare un loads dell'oggetto
                            arguments = json.loads(arguments)

                        predict = extract_info(second_client, **arguments, structured_output=FollowUp)
    
                        logger.debug(f"l'esperto estrattore ha ritornato: {predict.model_dump_json()}")
    
                        
                        # passiamo la risposta al primo modello
                        
                        
                        
                        output = f"Ho riconosciuto i seguenti campi:\n probabilità acquisizione: *{predict.prob_acquisizione}*\n data consegna: *{predict.data_consegna}*\n prezzo di vendita: *{predict.prezzo_vendita}*\n note: *{predict.note}*\n codice offerta: *{predict.codice_offerta}*\n\nsono corretti?"
                        
                        client.save_2_conversation(output)
                    else:
                        # altra funzione per il push nel db
                        ris_push = response['tool_calls']['arguments']
                        tool_id = response['tool_calls']['id']
                        logger.debug(f"è stato chiamato il tool: push data con i seguenti parametri:(push: {ris_push})")
                        result = "L offerta non è stata salvata"
                        if bool(ris_push['push']) == True:
                            result = "L offerta è stata salvata"
                        
                        logger.debug(f"la funzione push_data ha ritornato: {result}")
                        # passiamo la risposta al primo modello
                        response = client.chat(tool_name=funciton_name, tool_response=result, tool_id=tool_id, save_to_history=True)
                        output = response['content']
                else:
                    # non è stato chiamato il tool
                    logger.warning("non è stato chiamato nessun tool dal modello")
                    output = response['content']
                    
                print(output)
    
    
            ris =input(f"\n vuoi uscire?(N/Y):")
            # puliamo la memoria
            client.clear_history()
            if ris == 'Y':
                break

    



if __name__ == "__main__":
    ## add the config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file", help="Path to config file", default="config.yaml"
    )
    parser.add_argument(
        "-a", "--app", action="store", dest="app", help="use case", default="tecnico"
    )
    args = parser.parse_args()
    # non facciamo il check per non riccorrere all'errore di non avere settato le variabili d'ambiente 
    configurator = Configurator(file=args.conf_file, verify=False)

    conf = configurator.get_file()
    liveTest(conf,args.app)