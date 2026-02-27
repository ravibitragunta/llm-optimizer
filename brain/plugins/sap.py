from brain.plugins.base import ConversationPlugin, Entity

class SAPPlugin(ConversationPlugin):

    SAP_MODULES = ['FI','CO','MM','SD','HR','PP','WM','QM','PM','PS','BI','BW']
    SAP_CONCEPTS = [
        'ABAP','BASIS','FICO','S4HANA','S/4HANA','ECC','SAP ERP',
        'go-live','cutover','master data','transport','LSMW',
        'BAPI','BADI','user exit','enhancement spot','IMG','SPRO'
    ]
    PROJECT_PHASES = [
        'blueprint','realization','testing','go-live',
        'hypercare','UAT','SIT','migration','cutover'
    ]

    @property
    def name(self) -> str:
        return 'sap'

    def extract_entities(self, text: str) -> list:
        entities = []
        for module in self.SAP_MODULES:
            if f' {module} ' in f' {text} ':   # word boundary check
                entities.append(Entity(module, 'sap_module', 0.95))
        for concept in self.SAP_CONCEPTS:
            if concept.lower() in text.lower():
                entities.append(Entity(concept, 'sap_concept', 0.9))
        for phase in self.PROJECT_PHASES:
            if phase.lower() in text.lower():
                entities.append(Entity(phase, 'project_phase', 0.85))
        return entities

    def detect_decisions(self, text: str) -> list:
        decisions = []
        for phrase in ['we will use','decided to','confirmed that','approved','agreed to']:
            idx = text.lower().find(phrase)
            if idx >= 0:
                decisions.append(text[idx:idx+120].strip())
        return decisions

    def get_seed_domains(self) -> list:
        return ['sap_implementation']

    def get_system_prompt(self) -> str:
        return "You are an expert SAP consultant and implementation advisor."
