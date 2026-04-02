import pandas as pd
import pyarrow.parquet as pq
from typing import cast, List, Dict, Any
from omegaconf import OmegaConf
from utils.util import clean_response
from utils.logger import Logger
from utils.Tool import PushTool, ExtractorTool
from datetime import datetime
from models.ModelProviderClient import vLLMClient
from schema.structured_ouput_model import Tecnico
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import re
import json
import argparse
"""File per fare il test per l'accuracy dopo aver fatto il training"""

VAL_SET="/home/tecnico/SitiBT-Agent-AI/data/training_dataset/test/test.parquet"
OUTPUT_PATH="/home/tecnico/SitiBT-Agent-AI/data/training_dataset/results/"
TARGET_MODEL='qwen-chat-agent-9'
RUN_ID= 3
COLUMNS = [
            'model','language','messaggi', 'ore_ordinarie_true', 'ore_straordinarie_true', 'ore_viaggio_true', 'note_true', 'inefficiency_true','inefficiency_note_true',
           'chat_context', 'function_calls','push_data_response','ore_ordinarie_acc', 'ore_straordinarie_acc', 'ore_viaggio_acc',
            'inefficiency_acc','tot_acc','ore_ordinarie_predict', 'ore_straordinarie_predict', 'ore_viaggio_predict', 'note_predict', 'inefficiency_predict','inefficiency_note_predict',
            'REF_INDICE','L_MESSAGGI','TYPE_EVAL'
        ]
SAVE_CHECKPOINTS = 50
NUM_WORKERS = 3

conf = OmegaConf.load("config.yaml")
logger = Logger(save=False, consoleLevel="INFO").getLogger()

def compute_accuracy(result):
    """
    Calcola l'accuracy per i 4 campi numerici.
    - Se true != 0: confronto diretto (campo rilevante)
    - Se true == 0 e predict != 0: penalità (falso positivo)
    - Se true == 0 e predict == 0: ignorato (non rilevante, nessun errore)
    """
    fields = [
        ('ore_ordinarie',      'ore_ordinarie',        result['ore_ordinarie_predict']),
        ('ore_straordinarie',  'ore_straordinarie',    result['ore_straordinarie_predict']),
        ('ore_viaggio',        'ore_viaggio',          result['ore_viaggio_predict']),
        ('inefficiency',       'durata_inefficienza',  result['inefficiency_predict']),
    ]

    denominator = 0
    accumulator = 0

    for acc_key, true_key, pred_value in fields:
        true_value = result[f'{acc_key}_true']

        if true_value != 0:
            # campo rilevante: confronto diretto
            match = 1 if pred_value == true_value else 0
            result[f'{acc_key}_acc'] = match
            accumulator += match
            denominator += 1

        elif pred_value != 0:
            # falso positivo: true è 0 ma il modello ha predetto qualcosa
            result[f'{acc_key}_acc'] = 0
            denominator += 1
            # accumulator non incrementa -> penalità

        # else: true == 0 e predict == 0 -> non rilevante, skip

    if denominator == 0:
        # tutti i campi sono 0 sia in true che in predict (es. giornata di riposo)
        result['ore_ordinarie_acc'] = 1
        result['ore_straordinarie_acc'] = 1
        result['ore_viaggio_acc'] = 1
        result['inefficiency_acc'] = 1
        result['tot_acc'] = 1.0
    else:
        result['tot_acc'] = accumulator / denominator

    return result

def get_original(gt):
    clean_gt_str =  re.sub(r"<TOOLCALL>\[(.*?)\]</TOOLCALL>\n?",r"\1", gt['content'])
    clean_gt = json.loads(clean_gt_str)
    return clean_gt['arguments']

def process_sample(row, model_name):
    # load client
    chat_model = vLLMClient(conf, model_name="chat_model")
    extractor_model = vLLMClient(conf, model_name="extractor_model")
    chat_model.add_tools(tools=[ExtractorTool(), PushTool()])

    result = {}
    result['model'] = model_name
    
    # indice di riferimento dai dati: ./data/second_test/raw_data_combinated_with_less_inf.csv
    result['REF_INDICE'] = row['extra_info']['reference_id']
    # prendiamo i messaggi 
    user_prompts = row["prompt"]
    # prendiamo le gt
    gt = row['reward_model']['ground_truth']
    gt_assistant = [assistant for assistant in gt if assistant['role'] == 'assistant']
    gt_extractor = [assistant for assistant in gt if assistant['role'] == 'extractor']
    # la ground truth  giusta è l'ultima riga del extractor
    gt_info = get_original(gt_extractor[-1])
    
    # input da tenere
    result['language'] = row['extra_info']['language']
    
    result['ore_ordinarie_true'] = gt_info['ore_ordinarie']
    result['ore_straordinarie_true'] = gt_info['ore_straordinarie']
    result['ore_viaggio_true'] = gt_info['ore_viaggio']
    
    result['inefficiency_true'] = gt_info['durata_inefficienza']
    if 'note_inefficienza' in gt_info:
        result['inefficiency_note_true'] = gt_info['note_inefficienza'] 
    else:
        result['inefficiency_note_true'] = None

    result['note_true'] = gt_info['note']
    result['messaggi'] = user_prompts
    
    result['L_MESSAGGI'] = row['extra_info']['user_msg_length']
    result['TYPE_EVAL'] = row['extra_info']['type']
    
    result['function_calls'] = []

    
    
    for idx,user_msg in enumerate(user_prompts):
        chat_response = chat_model.chat(message=user_msg['content'], save_to_history=True)
        if chat_response['tool_calls']:
            function_name = chat_response['tool_calls']['name']
            result['function_calls'].append(function_name)
            tool_call_id = chat_response['tool_calls']['id']
            if function_name == 'extractor_expert':
                summary = chat_response['tool_calls']['arguments']['summary']
                # chiamiamo il modello 
                extractor_response = extractor_model.chat(message=summary, schema=Tecnico)
                extractor_response = clean_response(extractor_response['content'])
                try: 
                    json_response = json.loads(extractor_response)
                    predict = Tecnico.model_validate_json(json.dumps(json_response))
                except Exception as e:
                    # non siamo stati in grado di guardare il risultato, è un errore
                    result['ore_ordinarie_acc'] = 0
                    result['ore_straordinarie_acc'] =  0
                    result['ore_viaggio_acc'] =  0
                    result['inefficiency_acc'] = 0
                    result['tot_acc'] = 0
                    break
                      
                    
                # salviamo solo le predict e calcoliamo alla fine del push le accuracy
                result['note_predict'] = predict.note
                result['ore_ordinarie_predict'] = predict.ore_ordinarie
                result['ore_straordinarie_predict'] = predict.ore_straordinarie
                result['ore_viaggio_predict'] = predict.ore_viaggio
                result['inefficiency_note_predict'] = predict.note_inefficienza
                result['inefficiency_predict'] = predict.durata_inefficienza
                # inseriamo nella history del modello chat la tool response
                chat_model.save_2_conversation(text=predict.model_dump_json(), role="tool", tool_name=function_name, tool_call_id=tool_call_id)
            elif function_name == "push_data":
                try:
                    ris_push = chat_response['tool_calls']['arguments']['push']
                    result['push_data_response'] = ris_push
                except KeyError as e:
                    # avvenuto un keyError lo settiamo a false
                    result['push_data_response'] = False
                # controlliamo se siamo all'ultimo messaggio
                if idx != len(user_prompts) -1:
                    # non siamo all'ultimo messaggio è questo è un errore
                    result['ore_ordinarie_acc'] = 0
                    result['ore_straordinarie_acc'] =  0
                    result['ore_viaggio_acc'] =  0
                    result['inefficiency_acc'] = 0
                    result['tot_acc'] = 0
                    break
                # calcoliamo l'acc dell'extractor
                result = compute_accuracy(result)
                
        
        else:
            # se non viene chiamato nessun tool dovrebbe essere un errore     
            result['ore_ordinarie_acc'] = 0
            result['ore_straordinarie_acc'] =  0
            result['ore_viaggio_acc'] =  0
            result['inefficiency_acc'] = 0
            result['tot_acc'] = 0
            result['push_data_response'] = False
            break
    
    
    # save all the context 
    result['chat_context'] = chat_model.export_conversation()
    chat_model.clear_history()

    return result


def Test(test_set, output_path, repeat, model_name) -> pd.DataFrame:
    
    results = pd.DataFrame(columns=COLUMNS)
    if os.path.exists(output_path):
        results = pd.read_csv(output_path)
        logger.info(f"riprendiamo dal file {output_path}")
        for multi in range(1,repeat+1):
            ris = len(results) - multi*len(test_set)
            if ris < 0:
                # trovato
                from_run = multi - 1
                start_at = len(results) - from_run * len(test_set)
                break
            elif ris == 0:
                from_run = multi 
                start_at = 0
               
            logger.info(f"riprendiamo da run {from_run} su {repeat}")
    else:
        results = pd.DataFrame(columns=COLUMNS)
        start_at = 0
        from_run = 0

    logger.info('-' * 100)
    logger.info(f'modello da testare versione: {output_path}\n')
    logger.info('-' * 100)
    for _ in range(from_run,repeat):
        length= len(test_set)
        remaining = list(range(start_at, length))
        # lista temporanea per le nuove righe
        tmp_rows = []
        with tqdm(total=length, initial=start_at) as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(process_sample, test_set[j],model_name): j
                    for j in remaining
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(f"errore nel sample {futures[future]}: {e}")

                    tmp_rows.append(result)
                    # incrementiamo la progress bar
                    pbar.update(1)
                    if len(tmp_rows) % SAVE_CHECKPOINTS == 0:
                        # salva il df corrente ad ogni iterazioni
                        if tmp_rows:
                            new_df = pd.DataFrame(tmp_rows)
                            results = pd.concat([results, new_df], ignore_index=True)
                            # svuotiamo la lista
                            tmp_rows = []
                        results.to_csv(output_path, index=False, index_label=False)
                        logger.info(f"risultati salvati al checkpoint  in '{output_path}'\n")




        # salva il df corrente ad ogni iterazioni
        if tmp_rows:
            new_df = pd.DataFrame(tmp_rows)
            results = pd.concat([results, new_df], ignore_index=True)

        results.to_csv(output_path, index=False, index_label=False)
        logger.info(f"risultati salvati in '{output_path}'\n")
        logger.info(f'==================================================\n')


def main():
    # laod dataset
    table = pq.read_table(VAL_SET)
    df = cast(List[Dict[str, Any]], table.to_pylist())
    logger.info(f"load dataset: {len(df)} sample to test")

    # load output_name & target model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",action="store", dest="model_name", help="output model name", required=True
    )
    args = parser.parse_args()

    
    """ output_file = OUTPUT_PATH + f"test_{TARGET_MODEL}_{id}_{datetime.now().strftime("%Y_%m_%d")}.csv"
    logger.info(f"risultati da salvare in: {output_file}") """
    output_file = OUTPUT_PATH + f"test_{args.model_name}_{datetime.now().strftime("%Y_%m_%d")}.csv"
    Test(df, output_file, repeat=RUN_ID, model_name=args.model_name)


if __name__ == "__main__":
    main()