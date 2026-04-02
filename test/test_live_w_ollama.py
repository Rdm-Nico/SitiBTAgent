
from models.OllamaHttpClient import OllamaClient
from utils.logger import Logger
from schema.structured_ouput_model import Tecnico
from utils.util import clean_response
from utils.Tool import ExtractorTool, PushTool
from utils.configurator import Configurator
import argparse
import json

logger = Logger(save=False, consoleLevel="DEBUG").getLogger()



def liveTest(conf):
    """
    Live demo dell'extractor agent
    """
    # init client:
    client = OllamaClient(conf)

    # add model
    client.add_model("chat-granite4:latest")
    client.add_tools(tools=[ExtractorTool(), PushTool()])
    

    # add the second agent 
    # init client:
    second_client = OllamaClient(conf)
    second_client.add_model("extractor-granite4:latest")
    # passa al modello datapizza
    #chat_agent = ChatAgent(client=OpenAILikeClient(api_key="", model="granite4:latest", base_url="http://172.18.31.178:11434/v1", temperature=0.7))
    
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
            
            if 'message' in response and 'tool_calls' in response['message']:
                funciton_name = response["message"]['tool_calls'][0]['function']['name']
                

                if funciton_name == 'extractor_expert':
                    msg_to_send = response["message"]['tool_calls'][0]['function']['arguments']['summary']
                    logger.debug(f"è stato chiamato il tool: {funciton_name} con i seguenti parametri:(msg_to_send: {msg_to_send})")
                    model_response = second_client.generate(prompt=msg_to_send, schema=Tecnico)
                    clean_output = clean_response(model_response['response'])

                    try:
                        clean_json = json.loads(clean_output)
                        predict = Tecnico.model_validate_json(json.dumps(clean_json))
                    except Exception as e:
                        logger.error({"error": f"Errore: {str(e)}\n model_response: {model_response['response']},\n clean_output: {clean_output}"})

                    logger.debug(f"l'esperto estrattore ha ritornato: {predict.model_dump_json()}")

                    
                    # passiamo la risposta al primo modello
                    
                    find_inefficienza = True if predict.durata_inefficienza != 0 or (predict.note_inefficienza != "" and predict.note_inefficienza != "null" and predict.note_inefficienza != "NULL") else False
                    if find_inefficienza:
                        output = f"Ho riconosciuto i seguenti campi:\nore ordinarie: *{predict.ore_ordinarie}*\nore straordinarie: *{predict.ore_straordinarie}*\nore di viaggio: *{predict.ore_viaggio}* \ninnefficienze: *{find_inefficienza}*\ndurata inefficienza: *{predict.durata_inefficienza}*\netichetta inefficienza: *ETICHETTA*\ndescrizione inefficienza: *{predict.note_inefficienza}*\nnote: *{predict.note}*\ncommessa: *{predict.commessa}*\n\nsono corretti?"
                    else:
                        output = f"Ho riconosciuto i seguenti campi:\nore ordinarie: *{predict.ore_ordinarie}*\nore straordinarie: *{predict.ore_straordinarie}*\nore di viaggio: *{predict.ore_viaggio}* \ninnefficienze: *{find_inefficienza}*\nnote: *{predict.note}*\ncommessa: *{predict.commessa}*\n\nsono corretti?"
                    
                    client.save_2_conversation(output)
                else:
                    # altra funzione per il push nel db
                    ris_push = response["message"]['tool_calls'][0]['function']['arguments']['push']
                    logger.debug(f"è stato chiamato il tool: {funciton_name} con i seguenti parametri:(push: {ris_push})")
                    result = "La commessa non è stata salvata"
                    if bool(ris_push) == True:
                        result = "La commessa è stata salvata"
                    
                    logger.debug(f"la funzione push_data ha ritornato: {result}")
                    # passiamo la risposta al primo modello
                    response = client.chat(tool_name=funciton_name, tool_response=result, save_to_history=True)
                    output = response['message']['content']
            else:
                # non è stato chiamato il tool
                output = response['message']['content']
                
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
    args = parser.parse_args()
    # non facciamo il check per non riccorrere all'errore di non avere settato le variabili d'ambiente 
    configurator = Configurator(file=args.conf_file, verify=False)

    conf = configurator.get_file()
    liveTest(conf)