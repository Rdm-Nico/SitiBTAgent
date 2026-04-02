import agentlightning as agl
from typing import Any, Dict, List, cast
import argparse
from utils.logger import Logger
import pyarrow.parquet as pq
from models.ModelProviderClient import vLLMClient
from utils.Tool import ExtractorTool, PushTool
from utils.util import clean_response
from schema.structured_ouput_model import Tecnico
import re
import json
import torch

logger = Logger(save=False, consoleLevel="INFO").getLogger()

CHAT_MODEL_SYSTEM_PROMPT= "Sei un assistente che estrae informazioni. Quando l'utente ti fornisce informazioni riguardanti le ore di lavoro fatte oppure le inefficienze trovate, DEVI chiamare la funzione extractor_expert per estrapolare le informazioni.\nDevi mandare alla funzione extractor_expert un riassunto di quello che l'utente ti ha detto riguardanti le ore di lavoro e le inefficienze trovate.\nDevi essere informale e non giudicare l'utente per le informazioni che lui ti da. Quando prendi in ingresso la risposta del extractor_expert non aggiungere informazioni sbagliate o pareri, devi ritornare quello che extractor_expert ritorna a te.\nDopo aver ricevuto la risposta dal extractor_expert chiedi conferma all'utente per salvare i dati. Se e solo se l'utente conferma le informazioni che sono state estratte te DEVI chiamare\nla funzione push_data per salvare le informazioni nel database, SOLO dopo l'utente ti a confermato che vanno bene, non prima  \nESEMPIO:\n[USER]: \"Ieri ho fatto 7 ore di lavoro\"\n[ASSISTENTE]: <chiama function extractor_expert con parametro: 'lavorato 7 ore '> \n[ASSISTENTE]: <extractor_expert ritorna -> {\n  \"ORE_ORDINARIE\": 7.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"INEFFICIENCY\": false,\n  \"NOTE\": null,\n  \"COMMESSA\": null,\n  \"risposta_singola\":\"\"\n} >\n[ASSISTENTE]: hai fatto 7 ore di lavoro senza inefficienze identificate, è corretto ?\n[USER]: si va bene \n[ASSISTENTE]: <chiama la function push_data>\n[ASSISTENTE]: <extractor_expert ritorna -> La commessa  è stata salvata >\n[ASSISTENTE]: la commessa è stata salvata sul database"
EXTRACTOR_MODEL_SYSTEM_PROMPT="Sei un assistente specializzato nell'estrazione di informazioni da messaggi per costruire una JSON.\n\nCOMPITO:\nAnalizza il messaggio dell'utente, identifica i dati rilevanti e costruisci un JSON con i campi specificati.\n\nREGOLE:\n- Estrai solo informazioni presenti esplicitamente nel testo\n- Se un campo richiesto non è presente nel messaggio, inserisci NULL\n- Non inventare o inferire dati non esplicitamente menzionati\n- Usa i valori esatti trovati nel testo\n- Costruisci il JSON nel formato richiesto\n\nCAMPI DA CERCARE:\ndevi cercare nel testo se si fa riferimento ai seguenti campi e salvali così come sono:\n- ORE_ORDINARIE: ore di lavoro ordinarie effettuate\n- ORE_STRAORDINARIE: ore di lavoro straorinario effettuate\n- ORE_VIAGGIO: ore di viaggio effettuate\n- DURATA_INEFFICIENCY: è un valore numerico che identifica quante ore a perso il tecnico per una inefficienza riscontrata a lavoro. se non trovi nessuna inefficienza settare questo campo a 0\n- NOTE_INEFFICIENCY: usalo per inserire la descrizione dell'inefficienza che il tecnico ha riscontrato. Se non trovi nessuna inefficienza deve essere NULL. Deve essere breve e conciso.\n- NOTE: usalo per inserire le cause per cui non ci sono state ore di lavoro oppure altre informazioni sulla giornata. Altrimenti deve essere NULL, deve essere breve e conciso senza l'utilizzo di tempi verbali. IMPORTANTE:la stringa deve essere sempre dentro a virgolette.\n- COMMESSA: id della commessa di lavoro, è una stringa numerica in questo formato 250376, può essere non sempre presente, se non la si trova settare a NULL il campo\n\nOUTPUT:\nRestituisci il JSON costruito con i dati estratti e i campi esattamente come ti ho passato, ritorna unicamente il JSON.\nOUTPUT ESEMPIO:\n{\n  \"ORE_ORDINARIE\": 0.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 0.0,\n  \"NOTE_INEFFICIENCY\":null,\n  \"NOTE\": null,\n  \"COMMESSA\": null\n}\nESEMPIO1:\n Ieri è stata una giornata di riposo\nOUTPUT:\n{\n  \"ORE_ORDINARIE\": 0.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 0.0,\n  \"NOTE_INEFFICIENCY\":null,\n  \"NOTE\": riposo,\n  \"COMMESSA\": null\n}\nESEMPIO2:\n  Ieri per la commessa 290120 ho fatto 7 ore di lavoro\nOUTPUT:\n{\n  \"ORE_ORDINARIE\": 7.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 0.0,\n  \"NOTE_INEFFICIENCY\":null,\n  \"NOTE\": null,\n  \"COMMESSA\": 290120\n}\nESEMPIO3:\n  Ieri ho lavorato 5 ore ma ho avuto un problema che mi è costato un ora del mio tempo: il cliente non mi forniva un muletto per svolgere un operazione \nOUTPUT:\n{\n  \"ORE_ORDINARIE\": 5.0,\n  \"ORE_STRAORDINARIE\": 0.0,\n  \"ORE_VIAGGIO\": 0.0,\n  \"DURATA_INEFFICIENCY\": 1.0,\n  \"NOTE_INEFFICIENCY\":\"cliente non mi forniva un muletto per svolgere un operazione\",\n  \"NOTE\": null,\n  \"COMMESSA\": null\n}"
class SitiAgent:
    """
    Classe con all'interno la logica dell'agente di Siti 
    """
    def __init__(self, llm_resources:agl.NamedResources):
        # creiamo i due client 
        self.chat_model: vLLMClient = vLLMClient(base_url=llm_resources.endpoint)
        self.extractor_model: vLLMClient = vLLMClient(base_url=llm_resources.endpoint)
        
        self.embedding_model: vLLMClient = vLLMClient(base_url="http://172.18.31.178:8001")

        # aggiungiamo gli hyperparameters
        self.chat_model.add_options(llm_resources.sampling_parameters)

        # aggiungiamo i system prompt 
        self.chat_model.add_system_prompt(CHAT_MODEL_SYSTEM_PROMPT)
        self.extractor_model.add_system_prompt(EXTRACTOR_MODEL_SYSTEM_PROMPT)

        # aggiungiamo i tools
        self.chat_model.add_tools([ExtractorTool(),PushTool()])

    
    def chat(self, msg:str):
        """Funzione per scrivere un messaggio al chat model"""
        return self.chat_model.chat(message=msg, save_to_history=True, user="chat")

    def extract(self, msg:str)-> Tecnico|None:
        """Funzione per scrivere un messaggio al extractor model"""
        model_response = self.extractor_model.chat(message=msg, schema=Tecnico, user="extractor")
        # controlliamo che non sia avvenuto un errore:
        if "error" in model_response['status']:
            logger.error(f"errore nella estrazione dei dati: {model_response["status"]["error"]}")
            # facciamo un alto tentativo
            model_response = self.extractor_model.chat(message=msg, schema=Tecnico, user="extractor")
            if "error" in model_response['status']:
                logger.error(f"errore nuovamente, passiamo avanti: {model_response["status"]["error"]}")
                return None
        # puliamo da eventuali json tag non desiderati
        model_response = clean_response(model_response['content'])
        # castiamo e controlliamo i campi
        try: 
            json_response = json.loads(model_response)
            model_predict = Tecnico.model_validate_json(json.dumps(json_response))
        except Exception as e:
            # proviamo a fare un altro tentativo
            model_response = self.extractor_model.chat(message=msg, schema=Tecnico, user="extractor")
            if "error" in model_response['status']:
                logger.error(f"errore nuovamente, passiamo avanti: {model_response["status"]["error"]}")
                return None
            # puliamo da eventuali json tag non desiderati
            model_response = clean_response(model_response['content'])
            # castiamo e controlliamo i campi
            try: 
                json_response = json.loads(model_response)
                model_predict = Tecnico.model_validate_json(json.dumps(json_response))
            except Exception as e:
                logger.error({"error": f"Errore: {str(e)}\n model_response: {model_response}, ritorniamo None e continuiamo"})
                return None
        
        return model_predict

    def embed(self, prompt:str) -> list[float]:
        """Funzione per convertire un testo in embeddings"""
        return self.embedding_model.embed(prompt=prompt, dim=256)
    
    def save_tool_response(self,extractor_response:Any, tool_id:str, tool_name:str):
        """Funzione per salvare la risposta del tool nella history del chat model"""
        self.chat_model.save_2_conversation(text=extractor_response, role="tool", tool_name=tool_name, tool_call_id=tool_id)


        

class LitSitiAgent(agl.LitAgent[Dict[str,Any]]):
    def __init__(self, *, trained_agents = None):
        super().__init__(trained_agents=trained_agents)


    def compute_push_reward(self, response:Dict[str,Any], gt:Dict[str,Any])->float:
        """Funzione per calcolare il final reward dell'episodio"""
        # estraiamo le ground truth
        clean_gt_str =  re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", gt['content'])
        # e convertiamo in un json object
        clean_gt = json.loads(clean_gt_str)
        
        if clean_gt['name'] != 'push_data':
            #TODO: in teoria questo non dovrebbe accadere. se lo nota allora gestiro in un certo modo
            logger.error(f"la ground truth è non simmetrica con i dati: {gt}")
            return 0.0
        if clean_gt['arguments']['push'] == bool(response['push']):
            # abbassiamo il premio della push per evitare che il modello chiami solo questo
            return 2
        else:
            return 0

    def return_gt_extractor(self, gt:Dict[str, Any])-> Tecnico|None:
        # estraiamo le ground truth
        clean_gt_str =  re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", gt['content'])
        # e convertiamo in un json object
        clean_gt = json.loads(clean_gt_str)
        # convertiamo nel modello
        try: 
                gt_model = Tecnico.model_validate_json(clean_gt)
        except Exception as e:
            logger.error({"error": f"Errore: {str(e)}\n gt: {gt}"})
            return None
        return gt_model

    def compute_extractor_reward(self, response:Tecnico,gt:Dict[str, Any], metric='all')-> float:
        """Funzione per calcolare i verifiable reward dell extractor model"""
        # estraiamo le ground truth
        clean_gt_str =  re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", gt['content'])
        # e convertiamo in un json object
        clean_gt = json.loads(clean_gt_str)

        # facciamo il confronto:
        match metric:
            case 'all': 
                # 1 se tutti i valori sono uguali, 0 altrimenti
                if response.ore_ordinarie == clean_gt['arguments']['ore_ordinarie'] and response.ore_straordinarie == clean_gt['arguments']['ore_straordinarie'] \
                    and response.ore_viaggio == clean_gt['arguments']['ore_viaggio'] and response.durata_inefficienza == clean_gt['arguments']['durata_inefficienza']:
                    return 1
                else:
                    return 0
                
    def compute_summary_reward(self, predict:str, gt:Dict[str,Any])-> torch.Tensor:
        """Funzione per calcolare il reward tramite gli embedding"""
        # estraiamo gli embedding
        gt_embed =  torch.Tensor(gt['embedding'])
        """ # e convertiamo in un json object
        clean_gt = json.loads(clean_gt_str)
        try:
            result = self.agent.embed(clean_gt['arguments']['summary'])
            if "error" in result['status']:
                logger.warning(f"errore nella request per embedding: {result["status"]["error"]}")
                return None
            gt_embed =  torch.Tensor(self.agent.embed(clean_gt['arguments']['summary'])['embeddings'][0]['embedding'])
        except Exception as e:
            logger.error(f"la gt in verità è {clean_gt['arguments']}") """
        
        try:
            predict_embed = torch.Tensor(self.agent.embed(predict)['embeddings'][0]['embedding'])
        except Exception as e:
            logger.error(f"errore nel embed del tensore del predict")
        #logger.debug(gt_embed)
        # compute cosine similarity
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        result = cos(predict_embed, gt_embed)

        return result

    def rollout(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> float|None:
        """
        Rollout function che fa arrivarre una task dallo scheduler di agent-lightning (nel nostro caso un elemnto proveniente dal dataset di training),
        una resorsa su cui fare il training (nel nostro caso llm endpoint di vLLM per modificare i pesi del modello) e dei metadati sulla rollout.
        Quello il nostro LitSitiAgent fa sará:
        1. prendere una task che contiene una conversazione completa dell'utente 
        2. per ogni user prompt si utilizza llm con le chiamate ai tools 
        3. si calcola il valore intermedio dei reward ad ogni step della conversazione 
        4. si emette un unico reward globale
        """
        llm: agl.LLM = cast(agl.LLM, resources['main_llm'])

        # init agent
        self.agent:SitiAgent = SitiAgent(llm)

        final_turn_reward = 0
        #summary_rewards = torch.Tensor([]) 
        push_reward = 0
        #turn_reward = 0
        
        user_prompts = task["prompt"]
        max_turns = task['extra_info']['user_msg_length']
        gt = task['reward_model']['ground_truth']
        # separiamo assistant gt rispetto a extractor
        gt_assistant = [assistant for assistant in gt if assistant['role'] == 'assistant']
        gt_extractor = [assistant for assistant in gt if assistant['role'] == 'extractor']

        
        

        
        for idx,user_msg in enumerate(user_prompts):
            # skippiamo il primo messaggio già inserito
            model_response = self.agent.chat(user_msg['content'])
            #logger.debug(f"model response of turn {idx}: {model_response}")
            # controlliamo se ha chiamato un tool
            if model_response['tool_calls']:
                function_name = model_response['tool_calls']['name']
                tool_call_id = model_response['tool_calls']['id']

                if function_name == 'extractor_expert':
                    if idx == max_turns -1:
                        logger.info(f"modello chiama extractor quando dovrebbe chiamare push_data: reward nullo e blocco")
                        break
                    # gli forniamo un reward di più uno perché a chiamato il tool nel momento adatto
                    #turn_reward += 1
                    
                    # calcolare il reward prima di chiamara l'agent (con gli embedding)
                    try:
                        summary = model_response['tool_calls']['arguments']['summary']
                    except Exception as e:
                        logger.error(f"il modello ha ritonato un errore: {str(e)} per la response:\n {model_response}\n blocchiamo l'iterazione")
                        break
                    try:
                        #summary_rewards = torch.cat((summary_rewards,self.compute_summary_reward(summary, gt_assistant[idx]).reshape(1)))
                    
                        # aggiungiamo il reward del tool e della summary
                        intermediate_reward = 1.0 + self.compute_summary_reward(summary, gt_assistant[idx]).item()
                        agl.emit_reward(intermediate_reward)
                    except Exception as e:
                        logger.warning(f"il modello ha ritonato un errore: {str(e)} per la response:\n {model_response}\n blocchiamo l'iterazione")
                        break
                    
                    extractor_response = self.agent.extract(summary)
                    if extractor_response == None:
                        logger.warning(f"il modello estrattore è andato in errore, in questi casi limitati utilizziamo la gt")
                        extractor_response = self.return_gt_extractor(gt_extractor[idx])
                        if extractor_response == None:
                            logger.error(f"la gt non è stata riuscita ad inserire nel modello Pydantic, continuiamo come nulla fosse")
                            continue
                    """ except Exception as e:
                        logger.warning(f"il modello ha ritonato un errore: {str(e)} per la response:\n {model_response}\n blocchiamo l'iterazione")
                        break """

                    
                    # inseriamo nella history la risposta del extractor come un tool
                    self.agent.save_tool_response(extractor_response.model_dump_json(),tool_call_id,function_name) 

                elif function_name == 'push_data':
                    if idx != max_turns - 1:
                        logger.warning(f"il push tool è stato chiamato prima del previsto. blocchiamo l'iterazione")
                        break
                    # gli forniamo un reward di più uno perché a chiamato il tool nel momento adatto
                    final_turn_reward = 1
                    ris_push = model_response['tool_calls']['arguments']
                    # controlliamo se corrisponde alla gt
                    try:
                        push_reward += self.compute_push_reward(ris_push, gt_assistant[idx])
                    except TypeError as e:
                        logger.warning(f"rollout: {rollout.rollout_id} ha ritornato un errore di tipo: {str(e)} con la response del modello:\n{model_response}")
                        push_reward = 0
                    #logger.debug(f"reward del push è {push_reward}")
                else:
                    logger.info(f"chiamato una funzione che non esiste: {function_name}: blocchiamo iterazione")
                    break    
            else:
                # reward negativo per non aver chiamato nessun tool
                logger.info(f"non chiamato nessun tool: blocchiamo iterazione e diamo i reward fino ad ora")
                break
    
        
        # compute final reward
        final_reward = push_reward + final_turn_reward
        
        """ if fail_tool_reward != 0:
            # forniamo come final reward solo quello 
            final_reward = fail_tool_reward
        # check first if summary_rewards ha dei valori se no il final_reward sarà nan e questo bloccherà il training
        elif summary_rewards.numel() != 0:
            
            final_reward = push_reward + fail_reward + summary_rewards.mean(0).item()  + turn_reward
        else:
            # se non abbiamo salvato nessuna dato allora il final reward deve essere nullo 
            final_reward = 0 """
        
        logger.info("+" * 30)
        logger.info(f"rollout id: {rollout.rollout_id}")
        logger.info(f"msg length: {max_turns}")
        #logger.info(f"turn reward: {turn_reward}")                    
        #logger.info(f"summary_rewards: {summary_rewards}")
        logger.info(f"push_reward: {push_reward}")
        logger.info("+" * 30)
        logger.info(f"final reward of rollout: {rollout.rollout_id} is: {final_reward}")
        return final_reward





def debug_siti_agent(args:Any):
    """
    funzione per fare il run dell'agente attraverso un llm locale prima di utilizzare VERL per controllare se il workflow avviene come atteso prima del RL
    """
    if args.train_file == None:
        logger.error("é necessario specificare un dataset di training per realizzare il debug")
        return
    table = pq.read_table(args.train_file)

    df = cast(List[Dict[str, Any]], table.to_pylist())
    logger.debug(f"Debug data: {df}")
    

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=args.vllm_endpoint,
                model="Qwen/Qwen3-4B-Instruct-2507",
                sampling_parameters={"temperature": 0.7}    
            )
        },
    )

    trainer.dev(LitSitiAgent(), df)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", action="store", dest="train_file", help="file path for parquet training data" 
    )
    parser.add_argument(
        "-e", "--endpoint", action="store", dest="vllm_endpoint", help="endpoint of vllm model" 
    )
    args = parser.parse_args()
    debug_siti_agent(args)