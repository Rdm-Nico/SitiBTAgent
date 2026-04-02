from pydantic import Field, AwareDatetime
from enum import Enum
from typing import Annotated
from psycopg.types.enum import EnumInfo, register_enum
import psycopg
from psycopg import Connection
from pydantic import BaseModel
from typing import ClassVar


class message_role(str, Enum):
    """
    Enumerate del ruolo nella conversazione 
    """ 
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class SQLModel(BaseModel):
    """
    Superclass per avere una classe BaseModel 
    """
    MAPPING:ClassVar[dict] = {
    'string': 'TEXT',
    'integer': 'INTEGER',
    'number': 'NUMERIC(4,2)',
    'date': 'TIMESTAMP',
    'array': 'vector',
    'boolean': 'BOOLEAN'
    }


    def toQuery(self, primarykey:list[str], vec_dim:int=None, isnotnull:list[str]=None, dt:str="last_modify", double_precision:list[str]=[]) -> str:
        """
        Metodo che ritona lo schema da passare per creare la tabella:
        - msg(str): per passare lo schema
        - primarykey: per inserire la primary key
        - vec_dim: per inserire la dimensione del vettore 
        - isnotnull: per inserire vincoli non null,
        - dt: per specificare quali campi sono TIMESTAMP
        - double_precision: lista per tipi double precision
        """
        obj = self.model_json_schema()
        properties = obj['properties']
        output = ""

        for name in  properties.keys():
            keys = properties[name].keys()
            if "$ref" in keys:
                output += "{} {},\n".format(name,properties[name]['$ref'].split("/")[2] )
            else:
                py_type = properties[name]['type']
                sql_type = self.MAPPING[( py_type if name != dt else 'date')]

                if name in isnotnull:
                    if sql_type == 'vector':
                        output += "{} {}({}) {},\n".format(name, sql_type, vec_dim, 'NOT NULL') 
                    else:
                        output += "{} {} {},\n".format(name, sql_type, 'NOT NULL') 
                elif sql_type == 'vector':
                    if name in primarykey:
                        output += "{} {}({}) {},\n".format(name, sql_type, vec_dim, 'PRIMARY KEY')
                    else:
                        output += "{} {}({}),\n".format(name, sql_type, vec_dim) 
                elif name in primarykey:
                    output += "{} {} {},\n".format(name, sql_type, 'PRIMARY KEY') 
                elif name in double_precision:
                    # siccome in python ci sono solo i float, ma magari si vuole certe volte dei numeri più grandi e dei numeri più piccoli
                    # si introduce questa lista per aggiungere le colonne che devono essere double precision
                    if name in primarykey:
                        output += "{} {} {},\n".format(name, 'DOUBLE PRECISION','PRIMARY KEY')
                    else:
                        output += "{} {},\n".format(name,'DOUBLE PRECISION')
                else:
                    output += "{} {},\n".format(name, sql_type) 

        # togliamo ,\n
        return output[:-2]
    
    def columns(self) ->list[str]:
        """
        Metodo per tornare i nomi delle colonne di una tabella
        """
        return list(self.model_dump().keys())


class Etichetta(SQLModel):
    """
    Classe per crare un etichetta da inserire nella tabella Postgres
    """
    etichetta:str = Field(description="l'etichetta della inefficienza", frozen=True)
    testo:str = Field(description="il testo dell'etichetta", frozen=True)
    embedding:list[float] = Field(description="embedding del testo (parziale)", frozen=True)


class MessaggioTecnici(SQLModel):
    """
    Classe per creare un messaggio della applicazione dei tecnici da inserire nella tabella Postgres
    """
    ruolo:message_role = Field(description="il ruolo dell'utente nella conversazione", default=None)
    commessa:str = Field(description="l'id della commessa del tecnico", frozen=True)
    cel_mittente:str = Field(description="numero di cellulare del mittente", default=None) 
    cel_destinatario:str = Field(description="numero di cellulare del destinatario", default=None)
    msg_id:str = Field(description="id del messagio inviato o ricevuto", default=None)
    matricola_tecnico:str = Field(description="numero di matricola del tecnico", frozen=True)
    data_intervento: Annotated[AwareDatetime, Field(description="la data dell'intevento da segnlare con timezone", frozen=True),]
    codice_intervento:str = Field(description="il codice dell'intervento", frozen=True)
    testo:str = Field(description="Testo del messaggio", default=None)
    last_modify: Annotated[AwareDatetime, Field(description="orario dell'ultima modifica con timezone", default=None)]
    cronologia:int =  Field(description="indica la cronologia della conversazione", default=None)
    status:str = Field(description="stato del messaggio", default="unknown") 
    embedding:list[float] = Field(description="embedding del testo (parziale)", default=None)
    complete:bool= Field(description="se la commessa è completata", default=False)
    ore_ordinarie:float = Field(description="ore di lavoro ordinarie effettuate", default=None)
    ore_straordinarie:float = Field(description="ore di lavoro straordinarie effettuate", default=None)
    ore_viaggio:float = Field(description="ore di viaggio effettuate", default=None)
    find_inefficienza:bool = Field(description="inefficienze riscontrata durante il lavoro, se si trova settare a true altrimenti deve essere false", default=None)
    inefficency:float = Field(description="Ore perse dal tecnico per l'inefficienza", default=None)
    inefficency_type:str = Field(description="L'etichetta dell'inefficienza", default=None)
    inefficency_note:str = Field(description="la descrizione da parte del tecnico dell'inefficienza", default=None)
    note_commessa:str = Field(description="inserire le cause per cui non ci siano state ore di lavoro, deve essere breve e conciso senza l'utilizzo di tempi verbali", default=None)
    tool_name:str = Field(description="Il nome della funzione che l'agente utilizza", default=None)
    failed_count:int = Field(description="indica quante volte questo messaggio è andato in status failed", default=None)


    
    
    def toQuery(self, primarykey, vec_dim, conn:Connection=None, isnotnull = None, dt = "last_modify", double_precision=None):
        # prima creaimo il mapping tra python e Postgres
        if conn != None:
            try:
                info = EnumInfo.fetch(conn, "message_role")
                if info:
                    register_enum(info, conn, message_role, mapping={m: m.value for m in message_role})
            except (psycopg.errors.DuplicateObject, psycopg.errors.UniqueViolation):
                # l'enum è già stato registrato, non facciamo nulla
                pass
            except Exception as e:
                raise e
        # se double_precision è None allora passiamo una lista vuota
        double_precision = [] if double_precision is None else double_precision
        return super().toQuery(primarykey=primarykey, vec_dim=vec_dim, isnotnull=isnotnull, dt=dt, double_precision=double_precision)

   
class MessaggioFollowUp(SQLModel):
    """
    Classe per creare un messaggio della applicazione dei commerciali da inserire nella tabella Postgres
    """
    ruolo:message_role = Field(description="il ruolo dell'utente nella conversazione", default=None)
    cod_offerta:str = Field(description="il codice del offerta", frozen=True)
    cel_mittente:str = Field(description="numero di cellulare del mittente", default=None) 
    cel_destinatario:str = Field(description="numero di cellulare del destinatario", default=None)
    msg_id:str = Field(description="id del messagio inviato o ricevuto", default=None)
    data_ultimo_contatto: Annotated[AwareDatetime, Field(description="la data dell'intevento da segnlare con timezone", default=None),]
    titolo:str = Field(description="la descrizione dell offerta", default=None)
    cliente:str = Field(description="nome del cliente", default=None)
    testo:str = Field(description="Testo del messaggio", default=None)
    link:str = Field(description="link a Lyra dell offerta", default=None)
    last_modify: Annotated[AwareDatetime, Field(description="orario dell'ultima modifica con timezone", default=None)]
    cronologia:int =  Field(description="indica la cronologia della conversazione", default=None)
    status:str = Field(description="stato del messaggio", default="unknown") 
    complete:bool= Field(description="se la commessa è completata", default=False)
    prob_acquisizione:float = Field(description="probabilità di acquisto", default=None)
    data_consegna:str = Field(description="data della consegna prevista", default=None)
    prezzo_vendita:float = Field(description="prezzo di vendita previsto dell offerta", default=None)
    note:str = Field(description="tutte le informazioni riguardanti l'offerta, questo campo deve essere dettagliato e il più simile possibile a quello passato", default=None)
    referente:str = Field(description="referente della offerta", default=None)
    tool_name:str = Field(description="Il nome della funzione che l'agente utilizza", default=None)
    failed_count:int = Field(description="indica quante volte questo messaggio è andato in status failed", default=None)
    
    
    def toQuery(self, primarykey, conn:Connection=None, isnotnull = None, double_precision=None, dt = "last_modify"):
        # prima creaimo il mapping tra python e Postgres
        if conn != None:
            try:
                info = EnumInfo.fetch(conn, "message_role")
                if info:
                    register_enum(info, conn, message_role, mapping={m: m.value for m in message_role})
            except (psycopg.errors.DuplicateObject, psycopg.errors.UniqueViolation):
                # l'enum è già stato registrato, non facciamo nulla
                pass
            except Exception as e:
                raise e

        return super().toQuery(primarykey=primarykey, isnotnull=isnotnull,dt=dt,double_precision=double_precision)
    

    