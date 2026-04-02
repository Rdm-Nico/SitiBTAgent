
class Tool:
    """Classe per instanziare un tool per i modelli Ollama"""

    def to_dict(self):
        """funzione per mostrare lo schema del tool"""
        output = {}
        output['type'] = self.type
        output['function'] = {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'type': self.params_type,
                'required': self.params_required,
                'properties': self.properties
            }
        }
        return output


class ExtractorTool(Tool):
    """Classe per l'extractor Tool di Ollama"""
    type = "function"
    name = "extractor_expert"
    description = "Extract and structure work-related information from user input"
    params_type = "object"
    params_required  = ['summary']
    properties = {
                        "summary": {"type": "string", "description": "A brief summary of the work activity mentioned by the user"}
                      }
    

    def to_dict(self):
        return super().to_dict()
    

class PushTool(Tool):
    """Classe per Push Tool di Ollama"""
    type = "function"
    name = "push_data"
    description = "Push the data to a db"
    params_type = "object"
    params_required  = ['push']
    properties = {
                        "push": {"type": "boolean", "description": "A boolean that confirm the data will need to push to the db"}
                      }
    

    def to_dict(self):
        return super().to_dict()