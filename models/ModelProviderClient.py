import requests
from omegaconf import DictConfig
import os
import json
from utils.logger import Logger
from datetime import datetime
from schema.structured_ouput_model import Tecnico
from typing import Dict, Any, Iterator
from utils.Tool import Tool
from openai import OpenAI
from abc import ABC,abstractmethod
logger = Logger(save=False).getLogger()



class InferenceClient:
    """Classe astratta per i client di inferenza dei modelli AI"""
    @abstractmethod
    def test_connection(self):
        pass
    @abstractmethod
    def add_model(self, model:str):
        pass
    @abstractmethod
    def get_model(self):
        pass
    @abstractmethod
    def add_system_prompt(self, system_message:str):
        pass
    @abstractmethod
    def add_options(self, options:dict):
        pass
    @abstractmethod
    def get_system_prompt(self):
        pass
    @abstractmethod
    def add_tools(self, tools:list[Tool]):
        pass
    
    @abstractmethod
    def tokenize(self,text:str)-> str:
        pass

    @abstractmethod
    def embed(self, prompt: str, dim:int = None,verbose:bool=False) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, prompt: str, stream: bool = False, save_to_history:bool = False, schema:Tecnico = None, verbose:bool=False) -> Dict[str, Any]:
        pass

    @abstractmethod
    def chat(self, message: str=None, useDB:bool=False,save_to_history: bool = False, tool_name:str=None, tool_response:str=None, tool_id:str=None, schema:Tecnico = None, verbose:bool=False) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def save_2_conversation(self, text, role="assistant"):
        pass
    @abstractmethod
    def translate_text(self,text: str, target_language: str) -> dict:
        pass
    @abstractmethod
    def clear_history(self):
        pass
    @abstractmethod
    def export_conversation(self):
        pass
    @abstractmethod
    def save_conversation(self):
        pass
    @abstractmethod
    def translate_text(self,text: str, target_language: str) -> dict:
        pass
    def generate_output(self, status:Any=None, content:str=None, tool_calls:dict=None, embeddings:Any=None)->dict[str,Any]:
        return {
            'status': status,
            'content': content,
            'tool_calls': tool_calls,
            'embeddings': embeddings
        }

def clientRouter(config:DictConfig, base_url:str=None, name:str=None)->InferenceClient:
    """
    Metodo per controllare quale client utilizzare per la seguente run
    """
    if config.provider.use == 'vllm':
        return vLLMClient(config=config, base_url=base_url, model_name=name)
    else:   
        return OllamaClient(config=config, base_url=base_url, model_name=name)

        
    

class OllamaClient(InferenceClient):
    def __init__(self, config:DictConfig, base_url: str = None, model_name:str=None):
        """Init e test connessione"""
        self.logger = logger
        self.config = config
        if base_url is None:
            base_url = config.ollama.url
        self.base_url = base_url.rstrip('/')

        # add conversation history , system prompt e model init 
        self.conversation_history = {}
        self.system_prompt = {}
        self.options = {}
        self.avaible_models = [config.ollama[model_name].source]
        self.tools_list = []

        if config:
            # add system prompts and other config from json file
            if 'path' in self.config.ollama[model_name]:
                modelfile_path = self.config.ollama[model_name].path
                with open(modelfile_path) as f:
                    d = json.load(f)
                if 'system_prompt' in d:
                    self.add_system_prompt(d['system_prompt'])
                if 'temperature' in d:
                    self.add_options({'temperature': d['temperature']})
                if 'top_p' in d:
                    self.add_options({'top_p': d['top_p']})

        # init cartella cronologia
        self.history_folder = 'history/'

        self.logger.info(f"   model name: {model_name}")
        # test connessione
        self.test_connection()
        # aggiungiamo il modello
        self.add_model(config.ollama[model_name].source)

    def test_connection(self)-> bool:
        """
        test della connesione a Ollama
        
        :return: risultato 
        :rtype: bool
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                self.logger.info(f"✅ Server Ollama raggiungibile - {len(models['models'])} modelli disponibili")
                self.logger.info(f"   URL: {self.base_url}")
            
                # Mostra i modelli
                for model in models['models']:
                    size_mb = model['size'] / (1024 * 1024)
                    self.logger.info(f"   📦 {model['name']} ({size_mb:.1f} MB)")
                    # save model name
                    self.avaible_models.append(model['name'])
            
                return True
            else:
                self.logger.error(f"❌ Server Ollama risponde con codice: {response.status_code}")
        except requests.ConnectionError:
            self.logger.error(f"❌ Impossibile connettersi al server Ollama su {self.base_url}")
            self.logger.error("   Assicurati che Ollama sia in esecuzione:")
            self.logger.error("   ollama serve")
        return False
    

    def add_hostname(self, hostname:str)->bool:
        """
        Modifica l'hostname attuale con cui si accede all'istanza Ollama
        
        :param hostname: hostname da utilizzare
        :type hostname: str
        :return: risultato della connessione
        """
        self.base_url = hostname
        # facciamo il test della connessione
        ris = self.test_connection()
        return ris


    def add_model(self, model: str):
        """Aggiungi il modello da utilizzare se è presente"""
        assert model in self.avaible_models
        self.model = model
        self.conversation_history[self.model] = []

    def get_model(self) -> str:
        """ritorna il modello utilizzato"""
        return self.model
    
    def unload_model(self) -> Dict[str, Any]:
        """Rimuove il modello corrente dalla memoria per liberare spazio nella VRAM"""
        payload = {
            "model": self.model,
            "keep_alive": 0
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )

            response.raise_for_status()
            
            return {"output": f"Il modello '{self.model}' è stato rimosso dalla VRAM"}
        except requests.RequestException as e:
            return {"error": str(e)}

    def add_system_prompt(self, system_message: str):
        """ aggiungere un system prompt al modello specifico per definire il comportamento del modello. Se è già presente lo sostutisce"""
        try:
            self.system_prompt[self.model] = {
                'role': 'system',
                'content': system_message
            }
            self.conversation_history[self.model].insert(0, self.system_prompt[self.model])
        except NameError as e :
            return {"error": str(e)}
        except TypeError as e: 
            return {"error": str(e)}
    
    def add_options(self, options:dict):
        """
        Funzione che aggiunge opzioni aggiuntive al modello
        """
        try:
            for k,v in options.items():
                self.options[k] = v

                
        except NameError as e :
            return {"error": str(e)}
        except TypeError as e: 
            return {"error": str(e)}

    def get_system_prompt(self) -> str:
        """Ritorna il system prompt"""
        return self.system_prompt
    
    def add_tools(self, tools:list[Tool]):
        """Funzione per caricare dei tools dentro alle chiamate /chat"""
        self.tools_list = [tool.to_dict() for tool in tools]
        
    def list_models(self) -> Dict[str, Any]:
        """Lista tutti i modelli disponibili"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def tokenize(self,text:str)-> str:
        """Genera i token dal testo fornito e dal modello specifico."""
        payload = {
            'model': self.model,
            'text': text
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/tokenize",
                json=payload
            )
            response.raise_for_status()

            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def embed(self, prompt: str, dim:int = None, verbose:bool=False) -> Dict[str, Any]:
        """
        Genera gli embedding usando l'API /api/embed
        """
        payload= {
            "model": self.model,
            "input": prompt,
            "dimensions" : dim
        }
        #self.logger.debug(f"payload dell embd: {payload}")
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json=payload
            )
            self.logger.debug(f"risposta del modello embedding: {response.json()}")
            response.raise_for_status()
            if verbose:
                return response.json()
            
            embeddings = []
            for emb in response.json()['embeddings']:
                embeddings.append({"embedding": emb})

            return self.generate_output(status="success", embeddings=embeddings)
        except requests.RequestException as e:
            if verbose:
                return {"error" : str(e)}
            return self.generate_output(status={"error" : str(e)})
    
    def generate(self, prompt: str, save_to_history:bool = False, schema:Tecnico = None, verbose:bool=False) -> Dict[str, Any]:
        """
        Genera una risposta usando l'API /api/generate
        
        :param prompt: promprt da passare al modello
        :type prompt: str
        :param save_to_history: salvare nella memoria della calsse
        :type save_to_history: bool
        :param schema: utilizzare uno schema per la structured output
        :type schema: Tecnico
        :param verbose: ritornare la risposta completa oppure una versione constrained 
        :type verbose: bool
        :return: risposta del modello
        :rtype: Dict[str, Any]
        """

        # controlla se è presente un system prompt e se si lo ritorna
        system_prompt = self.system_prompt.get(self.model) 
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt['content'] if system_prompt is not None else None,
            "format": schema.model_json_schema() if schema is not None else None,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            
            if save_to_history:
                self.conversation_history[self.model].append({
                    'role': 'user',
                    'content': prompt
                })
                self.conversation_history[self.model].append({
                    'role': 'assistant',
                    'content': response.json()['response']
                })
            if verbose:
                return response.json()
            
            return self.generate_output(status="success", content=response.json()['response'])
                
        except requests.RequestException as e:
            if verbose:
                return {"error": str(e)}    
            return self.generate_output(status={"error": str(e)})
    
    def chat(self, message: str=None, useDB:bool=False,save_to_history: bool = False, tool_name:str=None, tool_response:str=None,tool_id:str=None, schema:Tecnico = None, verbose:bool=False) -> Dict[str, Any]:
        """Chat usando l'API /api/chat"""

        
        if useDB:
            # stiamo usando il db per salvare i messaggi, quindi quello che ci viene passato è gia nel formato che ci interessa
            # add system prompt
            current_messages = self.conversation_history[self.model] +  message
        else:
            if tool_name and tool_response:
                current_messages = self.conversation_history[self.model] + [{
                    'role': 'tool',
                    'tool_name': tool_name,
                    'content': tool_response
                }]
            else:
                current_messages = self.conversation_history[self.model] + [{
                    'role': 'user',
                    'content': message
                }]
        
        #logger.debug(f"\nil contesto attuale è: {current_messages}\n")
        options = None
        if self.options:
            # aggiungiamo i parametri del modello 
            options = self.options

        
        payload = {
            "model": self.model,
            "messages": current_messages,
            "options": options,
            "tools": self.tools_list if len(self.tools_list) != 0 else None,
            "stream": False
        }
        logger.debug(f"payload: {payload}")
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            logger.debug(f"risposta modello: {response.json()}")
            response.raise_for_status()
            json_reponse = response.json()

            if save_to_history:
                self.conversation_history[self.model].append({
                    'role': 'user',
                    'content': message
                })
                

                if 'message' in json_reponse and 'tool_calls' in json_reponse['message']:
                    # salviamo anche la chiamata al tool
                    assistant_response = json_reponse["message"]['tool_calls'][0]['function']
                    self.conversation_history[self.model].append({
                    'role': 'assistant',
                    'tool_calls': [{
                        'type': 'function',
                        'function': assistant_response 
                    }] 
                    })
                else:
                    # salviamo solo il messaggio
                    assistant_response = json_reponse['message']['content']

                    self.conversation_history[self.model].append({
                        'role': 'assistant',
                        'content': assistant_response
                    })
            
            if verbose:
                return json_reponse
            
            # controlliamo se si stratta di una chiamata ad un tool
            if 'message' in json_reponse and 'tool_calls' in json_reponse['message']:
                tool_call = {
                    'name': json_reponse["message"]['tool_calls'][0]['function']['name'],
                    'type': 'function',
                    'id': json_reponse["message"]['tool_calls'][0]['id'],
                    'arguments': json_reponse["message"]['tool_calls'][0]['function']['arguments']
                }
                return self.generate_output(status="success", tool_calls=tool_call)
            # se no è un messaggio normale
            return self.generate_output(status="success", content=json_reponse['message']['content'])
                
                
        except requests.RequestException as e:
            if verbose:
                return {"error": str(e)}
            return self.generate_output(status={"error": str(e)})
        
    def save_2_conversation(self, text, role="assistant"):
        """
        Funzione che salva nella cronologia della conversazione del testo esterno
        """
        self.conversation_history[self.model].append({
                        'role': role,
                        'content': text
        })
        
    def translate_text(self,text: str, target_language: str) -> dict:
        """Traduce del testo usando Ollama"""
        prompt = f"Traduci il seguente testo in {target_language}:\n\n{text}"
        
        return self.generate(prompt=prompt) 

    
    def clear_history(self):
        """Pulisce la cronologia della conversazione"""
        # Mantieni solo i messaggi di sistema
        self.conversation_history[self.model] = [
            msg for msg in self.conversation_history[self.model]
            if msg['role'] == 'system'
        ]
    
    def export_conversation(self):
        """
        ritorna la conversazione come stringa json 
        """
        return json.dumps(self.conversation_history, indent=2)
    

    def save_conversation(self):
        """Salva la conversazione in un file JSON nella cartella 'history/' """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"programming_session_{timestamp}.json"
        filepath = os.path.join(self.history_folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print('sessione salvata in: ',filepath)
    

class vLLMClient(InferenceClient):
    """Classe Client vLLM"""
    def __init__(self, config:DictConfig=None, base_url: str = None, model_name:str=None):
        """Init e test connessione"""
        self.logger = logger

        if config is None:
            # si utilizza il base_url
            if base_url is None:
                logger.error(f"ERROR: uno tra il config dict e il base url deve essere utilizzato")
                return
            self.base_url = base_url.rstrip('/')
            # add OpenAI client 
            self.client = OpenAI(api_key="EMPTY", base_url=self.base_url + '/v1')
            self.model = self.client.models.list().data[0].id 
        else:    
            self.config = config

            # check model_name
            if model_name not in self.config.vllm:
                logger.error(f"ERROR: nome del modello non compare dentro al file yaml: {model_name}")
            if base_url is None:
                base_url = config.vllm[model_name].url

            self.base_url = base_url.rstrip('/')
            # add OpenAI client 
            self.client = OpenAI(api_key="EMPTY", base_url=self.base_url + '/v1')
            self.model =  None
            # search for the model that corrispond to source in the config
            for model in self.client.models.list():
                if model.id == config.vllm[model_name].source:
                    self.model = model.id
            if self.model == None:
                logger.error(f"ERROR: nessun modello presente in vllm corrisponde a quello presente nel campo source del config per il modello:{model_name}")
            logger.debug(f"modello settato a {self.model}")
        # add conversation history , system prompt e model init 
        self.conversation_history = {}
        self.system_prompt = {}
        self.options = {}
        
        self.avaible_models = []
        self.tools_list = []
        
        if config:
            # add system prompts and other config from json file
            if 'path' in self.config.vllm[model_name]:
                modelfile_path = self.config.vllm[model_name].path
                with open(modelfile_path) as f:
                    d = json.load(f)
                if 'system_prompt' in d:
                    self.add_system_prompt(d['system_prompt'])
                if 'temperature' in d:
                    self.add_options({'temperature': d['temperature']})
                if 'top_p' in d:
                    self.add_options({'top_p': d['top_p']})
        
        # init cartella cronologia
        self.history_folder = 'history/'

        self.logger.debug(f"   model name: {model_name}")
        # test connessione
        self.test_connection()
        


    def test_connection(self)-> bool:
        """
        test della connesione a vLLM con endpoint
        
        :return: risultato 
        :rtype: bool
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                model_response = response.json()
                self.logger.debug(f"✅ Server vLLM raggiungibile - {len(model_response['data'])} modelli disponibili")
                self.logger.debug(f"   URL: {self.base_url}")
            
                # Mostra i modelli
                for model in model_response['data']:
                    self.logger.debug(f"   📦 {model['id']}\t Max context length: {model['max_model_len']}")
                    self.logger.debug(f"    Model settings:  {self.options}")
                    # save model name
                    self.avaible_models.append(model['id'])
            
                return True
            else:
                self.logger.error(f"❌ Server vLLM risponde con codice: {response.status_code}")
        except requests.ConnectionError:
            self.logger.error(f"❌ Impossibile connettersi al server vLLM su {self.base_url}")
            self.logger.error("   Assicurati che vLLM sia in esecuzione:")
        return False
    
    def get_model(self) -> str:
        """ritorna il modello utilizzato"""
        return self.model
    def add_model(self, model):
        """Aggiungi il modello da utilizzare se è presente"""
        assert model in self.avaible_models
        self.model = model
        self.conversation_history[self.model] = []

    def add_system_prompt(self, system_message: str):
        """ aggiungere un system prompt al modello specifico per definire il comportamento del modello. Se è già presente lo sostutisce"""
        try:
            self.system_prompt[self.model] = {
                'role': 'system',
                'content': system_message
            }
            if  self.model in self.conversation_history:
                self.conversation_history[self.model].insert(0, self.system_prompt[self.model])
            else:
                self.conversation_history[self.model] = [self.system_prompt[self.model]]
        except NameError as e :
            return {"error": str(e)}
        except TypeError as e: 
            return {"error": str(e)}
    
    def add_options(self, options:dict):
        """
        Funzione che aggiunge opzioni aggiuntive al modello
        """
        try:
            for k,v in options.items():
                self.options[k] = v


        except NameError as e :
            return {"error": str(e)}
        except TypeError as e: 
            return {"error": str(e)}
    def get_system_prompt(self) -> str:
        """Ritorna il system prompt"""
        return self.system_prompt
    
    def add_tools(self, tools:list[Tool]):
        """Funzione per caricare dei tools dentro alle chiamate"""
        self.tools_list = [tool.to_dict() for tool in tools]
        
    def generate(self, prompt: str, save_to_history:bool = False, schema:Tecnico = None, verbose:bool=False) -> Dict[str, Any]:
        """
        Funzione simile a api/generate di ollama ma in questo caso si utilizza v1/chat/completions di vLLM 
        
        :param prompt: messaggio da passare al modello
        :param save_to_history: salvare nella conversazione della classe
        :param schema: che tipo di schema utilizzare per lo structured output
        :param verbose: fornire in output un constraint output oppure no
        :return: un dizionario con la risposta del modello
        :rtype: ChatCompletion | dict[str, str]
        """
        return self.chat(prompt, schema=schema, verbose=verbose)
    
    def tokenize(self,text:str)-> str:
        """Genera i token dal testo fornito e dal modello specifico."""
        payload = {
            'model': self.model,
            'text': text
        }
        try:
            response = requests.post(
                f"{self.base_url}/tokenize",
                json=payload
            )
            response.raise_for_status()

            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def embed(self, prompt: str, dim:int = None, verbose:bool = False) -> Dict[str, Any]:
        """
        Genera gli embedding usando l'API /v1/embedding
        """
        try:
            respose = self.client.embeddings.create(
                input=prompt, dimensions=dim, model=self.model
            )
            
            if verbose:
                return respose
            embeddings = []
            for emb in respose.data:
                embeddings.append({"embedding": emb.embedding})
            return self.generate_output(status="success", embeddings=embeddings)
        except Exception as e:
            if verbose:
                return {"error" : str(e)}
            return self.generate_output(status={"error" : str(e)})
    
    
    def chat(self, message: str=None, useDB:bool=False,save_to_history: bool = False, tool_name:str=None, tool_response:str=None, tool_id:str=None, schema:Tecnico = None, verbose:bool=False, user:str=None) -> Dict[str, Any]:
        """Chat usando l'API /chat/completions"""
        if useDB:
            # stiamo usando il db per salvare i messaggi, quindi quello che ci viene passato è gia nel formato che ci interessa
            # add system prompt
            current_messages = self.conversation_history[self.model] +  message
             
        else:
            if tool_name and tool_response:
                current_messages = self.conversation_history[self.model] + [{
                    'role': 'tool',
                    'name': tool_name,
                    'content': tool_response,
                    'tool_call_id': tool_id
                }]
            else:
                current_messages = self.conversation_history[self.model] + [{
                    'role': 'user',
                    'content': message
                }]
        
        #logger.debug(f"\nil contesto attuale è: {current_messages}\n")

        # add tool if are presents
        tools = self.tools_list if len(self.tools_list) != 0 else None
        # aggiungiamo lo schema se necessario
        response_format = None
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "tecnico-description",
                    "schema": schema.model_json_schema()
                }
            }
        # controlliamo se ci sono altri parametri da mettere
        temp = None
        top_p = None
        if self.options:
            if 'temperature' in self.options:
                temp = self.options['temperature']
            if 'top_p' in self.options:
                top_p = self.options['top_p']
        try:
            self.logger.debug(f"chiamata all'endpoint chat con il seguente payload: {current_messages}\ntools: {tools}\nresponse_format: {response_format}")
            chat_response = self.client.chat.completions.create(
                messages=current_messages, model=self.model, tools=tools, temperature=temp, top_p=top_p, response_format=response_format, extra_body={"return_token_ids": True}, user=user
            )
            self.logger.debug(f"chat response: {chat_response}")
            msg_response = chat_response.choices[0].message
            if save_to_history:
                self.conversation_history[self.model].append({
                    'role': 'user',
                    'content': message
                })
                if len(msg_response.tool_calls) > 0:
                    tool_call = msg_response.tool_calls[0]
                    # salviamo anche la chiamata al tool
                    self.conversation_history[self.model].append({
                        'role': 'assistant',
                        'tool_calls': [{
                            'id': tool_call.id,
                            'type': tool_call.type,
                            'function': {
                                'name': tool_call.function.name,
                                'arguments': tool_call.function.arguments
                            }
                        }]
                    })
                else:
                    # salviamo solo il messaggio
                    self.conversation_history[self.model].append({
                        'role': 'assistant',
                        'content': msg_response.content
                    })
            
            if verbose:
                return chat_response
            if len(msg_response.tool_calls) > 0:
                tool_call = {
                    'name': msg_response.tool_calls[0].function.name,
                    'type': 'function',
                    'id': msg_response.tool_calls[0].id,
                    'arguments': json.loads(msg_response.tool_calls[0].function.arguments)
                }
                return self.generate_output(status="success", tool_calls=tool_call)
            return self.generate_output(status="success", content=msg_response.content)
        except Exception as e:
            if verbose:
                return {"error": str(e)}
            return self.generate_output(status={"error": str(e)})
        
    
    def translate_text(self,text: str, target_language: str, verbose:bool=False) -> dict:
        """Traduce del testo usando l'api chat """
        prompt = f"Traduci il seguente testo in {target_language}:\n\n{text}"
        return self.chat(message=prompt, verbose=verbose)
        
    def save_2_conversation(self, text, role="assistant", tool_name=None, tool_call_id=None):
        """
        Funzione che salva nella cronologia della conversazione del testo esterno
        """
        if role != 'tool':
            self.conversation_history[self.model].append({
                            'role': role,
                            'content': text
            })
        else:
            assert tool_name and tool_call_id
            self.conversation_history[self.model].append({
                            'role': role,
                            'name': tool_name,
                            'content': text,
                            'tool_call_id': tool_call_id
            })
    
    def clear_history(self):
        """Pulisce la cronologia della conversazione"""
        # Mantieni solo i messaggi di sistema
        self.conversation_history[self.model] = [
            msg for msg in self.conversation_history[self.model]
            if msg['role'] == 'system'
        ]
    
    def export_conversation(self):
        """
        ritorna la conversazione come stringa json 
        """
        try:
            value = json.dumps(self.conversation_history, indent=2)
        except TypeError as e:
            logger.error(f"errore nel salvataggio della storia, per la storia con conversazione {self.conversation_history}")
            return None
        return value
    

    def save_conversation(self):
        """Salva la conversazione in un file JSON nella cartella 'history/' """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"programming_session_{timestamp}.json"
        filepath = os.path.join(self.history_folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print('sessione salvata in: ',filepath)
    
    def _handle_streaming_response(self, response) -> Iterator[Dict[str, Any]]:
        """Gestisce le risposte in streaming"""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue