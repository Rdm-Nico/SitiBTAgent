import jsonlines
INPUT_DATA_LOC="train_generated.jsonl"
DATASET_TYPE="extractor"
import random
import pandas as pd
import re
import pyarrow.parquet as pq
import json
from typing import Any, Dict, Optional, cast, List
from transformers import AutoTokenizer
from datasets import Dataset

EXTRACTOR_MODEL_SYSTEM_PROMPT="Sei un assistente specializzato nell'estrazione di informazioni da messaggi per costruire una JSON.\n\nCOMPITO:\nAnalizza il messaggio dell'utente, identifica i dati rilevanti e costruisci un JSON con i campi specificati.\n\nREGOLE:\n- Estrai solo informazioni presenti esplicitamente nel testo\n- Se un campo richiesto non è presente nel messaggio, inserisci NULL\n- Non inventare o inferire dati non esplicitamente menzionati\n- Usa i valori esatti trovati nel testo\n- Costruisci il JSON nel formato richiesto\n\nCAMPI DA CERCARE:\ndevi cercare nel testo se si fa riferimento ai seguenti campi e salvali così come sono:\n- ore_ordinarie: ore di lavoro ordinarie effettuate\n- ore_straordinarie: ore di lavoro straorinario effettuate\n- ore_viaggio: ore di viaggio effettuate\n- durata_inefficienza: è un valore numerico che identifica quante ore a perso il tecnico per una inefficienza riscontrata a lavoro. se non trovi nessuna inefficienza settare questo campo a 0\n- note_inefficienza: usalo per inserire la descrizione dell'inefficienza che il tecnico ha riscontrato. Se non trovi nessuna inefficienza deve essere NULL. Deve essere breve e conciso.\n- note: usalo per inserire le cause per cui non ci sono state ore di lavoro oppure altre informazioni sulla giornata. Altrimenti deve essere NULL, deve essere breve e conciso senza l'utilizzo di tempi verbali. IMPORTANTE:la stringa deve essere sempre dentro a virgolette.\n- commessa: id della commessa di lavoro, è una stringa numerica in questo formato 250376, può essere non sempre presente, se non la si trova settare a NULL il campo\n\nOUTPUT:\nRestituisci il JSON costruito con i dati estratti e i campi esattamente come ti ho passato, ritorna unicamente il JSON.\nOUTPUT ESEMPIO:\n{\n  \"ore_ordinarie\": 0.0,\n  \"ore_straordinarie\": 0.0,\n  \"ore_viaggio\": 0.0,\n  \"DURATA_INEFFICIENCY\": 0.0,\n  \"note_inefficienza\":null,\n  \"note\": null,\n  \"commessa\": null\n}\nESEMPIO1:\n Ieri è stata una giornata di riposo\nOUTPUT:\n{\n  \"ore_ordinarie\": 0.0,\n  \"ore_straordinarie\": 0.0,\n  \"ore_viaggio\": 0.0,\n  \"durata_inefficienza\": 0.0,\n  \"note_inefficienza\":null,\n  \"note\": \"riposo\",\n  \"commessa\": null\n}\nESEMPIO2:\n  Ieri per la commessa 290120 ho fatto 7 ore di lavoro\nOUTPUT:\n{\n  \"ore_ordinarie\": 7.0,\n  \"ore_straordinarie\": 0.0,\n  \"ore_viaggio\": 0.0,\n  \"durata_inefficienza\": 0.0,\n  \"note_inefficienza\":null,\n  \"note\": null,\n  \"commessa\": 290120\n}\nESEMPIO3:\n  Ieri ho lavorato 5 ore ma ho avuto un problema che mi è costato un ora del mio tempo: il cliente non mi forniva un muletto per svolgere un operazione \nOUTPUT:\n{\n  \"ore_ordinarie\": 5.0,\n  \"ore_straordinarie\": 0.0,\n  \"ore_viaggio\": 0.0,\n  \"durata_inefficienza\": 1.0,\n  \"note_inefficienza\":\"cliente non mi forniva un muletto per svolgere un operazione\",\n  \"note\": null,\n  \"commessa\": null\n}"

def post_process_agent_dataset(data):
    """
    function for post-process the data in a correct format for verl framework for agent conversational task.
    
    We convert in a format like the following(this is from verl Data preparation page):
    data = {
    "data_source": data_source,     # String: Name/identifier of the data source
    "prompt": [                     # List: Conversation format
            {
                "role": "user",            
                "content": question,       
            }
        ],
        "abitity": "math",         # String: tag
        "reward_model": {
            "style": "rule",           # String: Either "rule" or "reward_model"
            "ground_truth": solution,   # Expected solution
        },
        "extra_info": {                 # Dict: Optional additional metadata
            # ... add your own fields here
        },
    }

    so in the prompt key we insert only the role flag with 'user', meanwhile in the ground_truth we can try to put another obj that will be the ground_truth solutions
    of the exporter and the chat model
    
    :param data: the data to convert
    """
    data_post_process = []
    for sample in data:
        # filter only user content:
        user_only_msg = [msg['content'] for msg in sample['messages'] if msg['role'] == 'user']
        # filter all the other messages
        gt_assistant_msg = [msg['content'] for msg in sample['messages'] if msg['role'] == 'assistant']
        gt_extractor_msg = [msg['content'] for msg in sample['messages'] if msg['role'] == 'extractor']
        
        obj = {
            "data_source": "conversation_tecnico",
            "prompt": [{
               "role": "user",
               "content": question 
            } for question in user_only_msg],
            "ability": "tool_calling",
            "reward_model": {
                "style": "rule",
                "ground_truth": [{
                    "role": "assistant", "content": result} for result in gt_assistant_msg
                    ] + 
                    [{
                    "role": "extractor", "content": result} for result in gt_extractor_msg
                    ]
            },
            "extra_info":{
                "language": sample['language'],
                "type": sample['type'],
                "id": sample['id'],
                "reference_id": sample['reference_id'],
                "agent_tools": sample['agent_tools'],
                "user_msg_length": len(user_only_msg)
            }
        }
        data_post_process.append(obj)

    return  data_post_process

def post_process_extractor_dataset(data):
    """
    function for post-process the data in a correct format for verl framework for LLM extractor task.
    
    We convert in a format like the following(this is from verl Data preparation page):
    data = {
    "data_source": data_source,     # String: Name/identifier of the data source
    "prompt": [                     # List: Conversation format
            {
                "role": "user",            
                "content": question,       
            }
        ],
        "abitity": "math",         # String: tag
        "reward_model": {
            "style": "rule",           # String: Either "rule" or "reward_model"
            "ground_truth": solution,   # Expected solution
        },
        "extra_info": {                 # Dict: Optional additional metadata
            # ... add your own fields here
        },
    }

    so in the prompt key we insert only the role flag with 'user', meanwhile in the ground_truth we put the ground_truth solution
    of the exporter
    
    :param data: the data to convert
    """
    data_post_process = []
    for sample in data:
        # filter only user content:
        user_only_msg = [msg['content'] for msg in sample['messages'] if msg['role'] == 'user']
        # filter all the other messages
        gt_assistant_msg = [msg['content'] for msg in sample['messages'] if msg['role'] == 'assistant']
        gt_extractor_msg = [msg['content'] for msg in sample['messages'] if msg['role'] == 'extractor']
        
        # we create a single rollout for every gt assistant response + gt_extractor response
        for index, (prompt_msg, response_msg) in enumerate(zip(gt_assistant_msg[:-1],gt_extractor_msg)):
            # facciamo già il clean del tag
            prompt_clean = re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", prompt_msg)
            clean_obj = json.loads(prompt_clean)
            question = clean_obj['arguments']['summary']

            gt_clean = re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", response_msg) 
            clean_gt_obj = json.loads(gt_clean)
            gt_response = clean_gt_obj["arguments"]
            obj = {
                "data_source": "conversation_tecnico",
                "prompt": [{
                   "role": "user",
                   "content": question 
                }],
                "ability": "structured output",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt_response
                },
                "extra_info":{
                    "language": sample['language'],
                    "type": sample['type'],
                    "id": sample['id'],
                    "progression_number_in_conversation": index,
                    "reference_id": sample['reference_id'],
                    "agent_tools": sample['agent_tools'],
                    "user_msg_length": len(user_only_msg)
                }
            }
            data_post_process.append(obj)

    return  data_post_process

def main():
    """Preprocessing data in a format that is acceptable for the training and save it in parquet file format"""
    # load data
    data = []
    with jsonlines.open(INPUT_DATA_LOC) as r:
        for obj in r:
            data.append(obj)

    print(f'load {len(data)} rows of data')

    # split in 1000 sample for the train and 200 sample for the test. For the test make scure the same reference_id is not present in both train and test
    test_set = []
    already_tested = []
    remove_idx = []
    while 1:
        idx = random.randint(0,len(data))
        if idx in already_tested:
            continue

        search_ids = [obj['reference_id'] for id,obj in enumerate(data) if id != idx]
        if data[idx]['reference_id'] not in search_ids:
            test_set.append(data[idx])
            remove_idx.append(idx)
        if len(test_set) == 200:
            break
        already_tested.append(idx)
    
    print(f"trovati 200 sample per il testing. la distribuzione della lingua é la seguente")
    print(f"italiano: {len([obj for obj in test_set if obj['language'] == 'italian'])}")
    print(f"ingelse: {len([obj for obj in test_set if obj['language'] == 'english'])}")
    print(f"spagnolo: {len([obj for obj in test_set if obj['language'] == 'spanish'])}")
    print(f"mostriamo anche la distribuzione delle task")
    tasks = ['normal','normal_w_correction','edge_no_explain']
    for task in tasks:
        print(f" {task}: {len([obj for obj in test_set if obj['type'] == task])}")

    
    # remove the idx from the main data
    train_set =  [sample for idx,sample in enumerate(data) if idx not in remove_idx]

    
    # 1 STEP: convert each obj in the correct format
    if DATASET_TYPE == "extractor":
        train_converted = post_process_extractor_dataset(train_set) 
        test_converted = post_process_extractor_dataset(test_set)
    
    print(f"train samples: {len(train_converted)}\t test samples: {len(test_converted)}")
    print(f"data correctly post process. Ex: \n{train_converted[80]}")
    # 2 STEP: save as a df in pandas 
    train_df = pd.DataFrame(train_converted)
    test_df = pd.DataFrame(test_converted)
        

    # 3 STEP: convert to parquet binary file
    train_df.to_parquet("train_extractor.parquet")
    test_df.to_parquet("test_extractor.parquet")



def sft_process_agent(dt:List[Dict[str, Any]])->List[Dict[str, Any]]:
    """
    Funzione che crea i dataset nel formato language modeling in modo conversational + toll calling.
    La struttura dovrebbe essere cosí:
    
    messages = [
    {"role": "user", "content": "Ieri ho lavorato 8 ore"},
    {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {
            "name": "extractor_expert",
            "arguments": {"summary": "Ieri ho lavorato otto ore"}
        }}]
    },
        {"role": "tool", "name": "extractor_expert", "content": '{/"ore_ordinarie/": 8.0, /"ore_straordinarie/": 0, /"ore_viaggio/":
 0, /"durata_inefficienza/": 0, /"note/": /"/", /"commessa/": /"/"}'},
        {"role": "user", "content": "ho fatto anche un ora di viaggio"},
        ...
    ]

    In oltre ci dovrebbe essere un altra colonna chiamata tools con gli schemi dei tools.
    Il risultato finale:

     {"messages": messages, "tools": json_schema}

     con json_schema = [{tool_schema},{tool_schema}]

    """

    output_list = []

    for row in dt:
        user_prompt = row["prompt"]
        gt = row['reward_model']['ground_truth']
        # separiamo assistant gt rispetto a extractor
        gt_assistant = [assistant for assistant in gt if assistant['role'] == 'assistant']
        
        gt_extractor = [assistant for assistant in gt if assistant['role'] == 'extractor']
        info = row["extra_info"]
        len_msg = info['user_msg_length']
        messages = []

        # build dei messaggi
        for idx,user_msg in enumerate(user_prompt):
            messages.append(user_msg)
            
            if idx != len_msg -1:
                # we add the gt_assistant
                assistant_response = gt_assistant[idx]['content']
                # convert <TOOLCALL> tags 
                assistant_response = re.sub("<TOOLCALL>","<tool_call>",assistant_response)
                assistant_response = re.sub("</TOOLCALL>","</tool_call>",assistant_response)
                # remove []
                assistant_response = re.sub(r"[\[\]]","", assistant_response)
                messages.append({'role': 'assistant', 'content': assistant_response})

                # add the gt_extractor
                tool_response = gt_extractor[idx]['content']
                # remove <TOOLCALL> tags
                tool_response = re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", tool_response)
                # convert in json and extract only the arguments 
                tool_obj = json.loads(tool_response)

                tool_result = tool_obj['arguments']

                messages.append({"role": "tool", "name": "extractor_expert", "content": json.dumps(tool_result)})
            else:
                # siamo all'ultimo messaggio in cui si chiama il push
                assistant_response = gt_assistant[idx]['content']
                # prendiamo il valore dell'arguments:
                tool_call = re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", assistant_response)
                clean_gt = json.loads(tool_call)
                try:
                    ris = clean_gt['arguments']['push']
                except Exception as e:
                    print(f"errore qui: {gt_assistant}")
                    raise Exception
                # convert <TOOLCALL> tags 
                assistant_response = re.sub("<TOOLCALL>","<tool_call>",assistant_response)
                assistant_response = re.sub("</TOOLCALL>","</tool_call>",assistant_response)
                # remove []
                assistant_response = re.sub(r"[\[\]]","", assistant_response)
                messages.append({'role': 'assistant', 'content': assistant_response})
                # aggiungiamo la tool response 
                tool_response = 'data updated' if ris == True else 'data not updated'
                model_response = 'Aggiornamento corretto' if ris == True else 'Aggiornamento non andato a buon fine'
                messages.append({"role": "tool", "name": "extractor_expert", "content": tool_response })
                # e l'assistant response 
                messages.append({'role': 'assistant', 'content': model_response})


        # add tools: rimuoviamo tutti i campi vuoti in modo ricorsivo
        tool_1 = clean_dict(info['agent_tools'][0])
        tool_2 = clean_dict(info['agent_tools'][1])
        
        # add all toghether 
        output_list.append({
            "messages": messages,
            "tools": [tool_1,tool_2],
            "metadata": {
                    "id": info["id"],
                    "reference_id": info['reference_id'],
                    "language": info['language'],
                    "user_msg_length": info['user_msg_length']
            }
        })
        
        

    return output_list

def sft_process_extractor(dt:List[Dict[str, Any]])->List[Dict[str, Any]]:
    """
    Funzione per creare un dataset di tipo language modelling conversazionale.
    lo schema dovrebbe essere:
    sample = {
    "messages": [
        {
            "role": "system",
            "content": (
                "Sei un assistente che estrae informazioni strutturate da report di lavoro giornalieri. "
                "Rispondi SEMPRE e SOLO con un oggetto JSON valido che rispetta esattamente questo schema, "
                "senza testo aggiuntivo, senza markdown, senza backtick:\n\n"
                "{\n"
                '  "ore_ordinarie": <float>,\n'
                '  "ore_straordinarie": <float>,\n'
                '  "ore_viaggio": <float>,\n'
                '  "durata_inefficienza": <float>,\n'
                '  "note": <string>,\n'
                '  "commessa": <string>\n'
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                "fatto 8 ore di lavoro: preparazione fruste per alimentare i traini dei due essicatoi, "
                "collegamento dei traini, posa sensori di movimento e raccordi. Ridimensionata la squadra "
                "da 6/8 a 2–3 persone perché non c'era canale elettrico, un inefficiente che ha rallentato "
                "il lavoro. Riorganizzato il lavoro per evitare fermo e mantenere la puntualità all'arrivo "
                "del canale. Prevediamo comunque ritardi nella consegna dei macchinari."
            )
        },
        {
            "role": "assistant",
            "content": (
                '{"ore_ordinarie": 8.0, "ore_straordinarie": 0, "ore_viaggio": 0, '
                '"durata_inefficienza": 0, '
                '"note": "Inizio Preparazione fruste per alimentazione traini dei due essicatoi ed '
                'collegamento dei traini, inizio posa sensori movimento traini, posa di raccordi vari.\\r\\n\\r\\n'
                'Nota ho dovuto ridimensionare la squadra in questi primi giorni, da 6/8 persone a 2 massimo 3... '
                'perché essendo assente di canale elettrico, ho dovuto riorganizzare tutto il lavoro, in modo che '
                'nessuno sia fermo, così si arriva giusti all\'arrivo del canale elettrico, da quel momento in poi '
                'all\'ora riavrò la squadra al completo, naturalmente ci saranno comunque ritardi sulle consegne '
                'dei macchinari per via del canale che arriverà tardi, naturalmente cercando di recuperare il tempo", '
                '"commessa": ""}'
            )
        }
    ]
}

c'é da tenere conto anche della distribuzione delle lingue:
per 200 righe di train:
- 150 samples in italiano
- 25 samples in inglese
- 25 samples in spagnolo 
stessa proporzione per i 30 samples di test
    """
    train_data = test_data = []
    italian_occ =  english_occ = spanish_occ = 0
    selected_ids = []

    while(len(train_data) < 200):
        # selezioniamo a caso un sample 
        index = random.randint(0,len(dt))
        if index in selected_ids:
            continue
        # controlliamo la lingua
        user_msg = dt[index]['prompt'][0]["content"]
        language = dt[index]['extra_info']['language']
        gt = dt[index]['reward_model']['ground_truth']
        if language == 'italian' and italian_occ > 150:
            selected_ids.append(index)
            continue
        if language == 'english' and english_occ > 25:
            selected_ids.append(index)
            continue
        if language == 'spanish' and spanish_occ > 25:
            selected_ids.append(index)
            continue
        #  controlliamo che il json della gt sia corretto

        try:
            json.loads(json.dumps(gt))
        except json.JSONDecodeError as e:
            print(f"errore nel json malformato della gt, andiamo avanti") 
            continue

        sample = {
            "messages": [
                {
                    "role": "system", 
                    "content": EXTRACTOR_MODEL_SYSTEM_PROMPT 
                },
                {
                    "role": "user",
                    "content": user_msg
                },
                {
                    "role": "assistant",
                    "content": json.dumps(gt)
                }

            ]
        }

        train_data.append(sample)
        selected_ids.append(index)
    
    print(f"train df: {len(train_data)}")

    # creo anche il test data 
    italian_occ =  english_occ = spanish_occ = 0
    while(len(test_data) < 30):
        index = random.randint(0,len(dt))
        if index in selected_ids:
            continue
        # controlliamo la lingua
        user_msg = dt[index]['prompt'][0]["content"]
        language = dt[index]['extra_info']['language']
        gt = dt[index]['reward_model']['ground_truth']
        if language == 'italian' and italian_occ > 20:
            selected_ids.append(index)
            continue
        if language == 'english' and english_occ > 5:
            selected_ids.append(index)
            continue
        if language == 'spanish' and spanish_occ > 5:
            selected_ids.append(index)
            continue
        #  controlliamo che il json della gt sia corretto

        try:
            json.loads(json.dumps(gt))
        except json.JSONDecodeError as e:
            print(f"errore nel json malformato della gt, andiamo avanti") 
            continue

        sample = {
            "messages": [
                {
                    "role": "system", 
                    "content": EXTRACTOR_MODEL_SYSTEM_PROMPT 
                },
                {
                    "role": "user",
                    "content": user_msg
                },
                {
                    "role": "assistant",
                    "content": json.dumps(gt)
                }

            ]
        }

        test_data.append(sample)
        selected_ids.append(index)

    print(f"test df: {len(test_data)}")
    
    return train_data, test_data




def clean_dict(df:dict)->dict:
    """Funzione ricorsiva per pulire il dict da valori nulli"""
    output = {}
    for k,v in df.items():
        if type(v) == dict:
            ris = clean_dict(v)
            output[k] = ris
        elif v != None:
            output[k] = v

    return output


def main_SFT():
    """
     Funzione per processare i dataset in formato adatto per trl SFT
    """
    # upload il train e il test
    train_table = pq.read_table("./train_extractor.parquet")
    train_dataset = cast(List[Dict[str, Any]], train_table.to_pylist()) 

    print(f"len del train prima del processo: {len(train_dataset)}")
    train_post, test_post = sft_process_extractor(train_dataset)
    print(train_post[0])
    print('-'*90)
    print(test_post[0])
    
    # save as json
    with open('train_sft_extractor.json','w') as f:
        json.dump(train_post,f)

    with open('../test/test_sft_extractor.json','w') as f:
        json.dump(test_post,f)

    
    


if __name__ == "__main__":
    #main()
    main_SFT()