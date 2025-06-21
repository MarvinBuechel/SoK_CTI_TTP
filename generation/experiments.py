from database import label_split
import finetuning.cti_datasets.tram.tram_classes as tram

# This file contains all experiment definitions.




experimentTramOpen = {
    "Name": "Tram Raw Open", # Experiment Name
    "PP": False, # Pre-Processing (not covered in the paper)
    "RAG": False, # Retrieval Augmented Generation
    "FSP": False, # Few-Shot Prompting
    "JudgeAgent": False, # Judge LLM (not covered in the paper)
    "DatasetTRAM": True, # use TRAM2 dataset
    "DatasetBosch": False, # use Bosch AnnoCTR dataset
    "DocumentLevel": False, # evaluate on document-level
    "NumberOfLabel": [] # Label set to evaluate on. Empty set means accepting all valid MITRE IDs.
}

experimentT1 = {
    "Name": "Tram Raw",
    "PP": False,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": False,
    "NumberOfLabel": tram.TRAM_CLASSES
}
experimentT3 = {
    "Name": "Tram RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": False,
    "NumberOfLabel": tram.TRAM_CLASSES
}
experimentT4 = {
    "Name": "Tram RAG + FSP",
    "PP": False,
    "RAG": True,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": False,
    "NumberOfLabel": tram.TRAM_CLASSES
}

experimentT2 = {
    "Name": "Tram FSP",
    "PP": False,
    "RAG": False,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": False,
    "NumberOfLabel": tram.TRAM_CLASSES
}


experimentT6 = {
    "Name": "Tram PP",
    "PP": True,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": False,
    "NumberOfLabel": tram.TRAM_CLASSES
}

experiment_tram_impact = {
    "Name": "TRAM_IMPACT",
    "PP": False,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": False,
    "NumberOfLabel": tram.TRAM_CLASSES
}

TramDocumentLevel = {
    "Name": "TramDocumentLevel",
    "PP": False,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": True,
    "NumberOfLabel": tram.TRAM_CLASSES
}

TramDocumentLevelRAG = {
    "Name": "TramDocumentLevel + RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": True,
    "NumberOfLabel": tram.TRAM_CLASSES
}

TramDocumentLevelFSP = {
    "Name": "TramDocumentLevel + FSP",
    "PP": False,
    "RAG": False,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": True,
    "DatasetBosch": False,
    "DocumentLevel": True,
    "NumberOfLabel": tram.TRAM_CLASSES
}

experimentBosch1 = {
    "Name": "Bosch Raw",
    "PP": False,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch3 = {
    "Name": "Bosch RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch2 = {
    "Name": "Bosch FSP",
    "PP": False,
    "RAG": False,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch4 = {
    "Name": "Bosch FSP + RAG",
    "PP": False,
    "RAG": True,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}


experimentBosch1_70B = {
    "Name": "Bosch 70B Raw",
    "PP": False,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch2_70B = {
    "Name": "Bosch 70B RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch3_70B = {
    "Name": "Bosch 70B FSP",
    "PP": False,
    "RAG": False,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch4_70B = {
    "Name": "Bosch 70B FSP + RAG",
    "PP": False,
    "RAG": True,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch5 = {
    "Name": "Bosch RAG + Judge",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": True,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

experimentBosch6 = {
    "Name": "Bosch Judge",
    "PP": False,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": True,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

BoschTTPOpen = {
    "Name": "BoschOpen RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": []
}

BoschTTPAll = {
    "Name": "BoschTTPAll RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

BoschTTP1 = {
    "Name": "Bosch25 RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_25_LABELS
}

BoschTTP2 = {
    "Name": "Bosch50 RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_50_LABELS
}

BoschTTP3 = {
    "Name": "Bosch53 RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_53_LABELS
}

BoschTTP4 = {
    "Name": "Bosch10 RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": False,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_10_LABELS
}


BoschDocumentLevel = {
    "Name": "BoschDocumentLevel",
    "PP": False,
    "RAG": False,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": True,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

BoschDocumentLevelRAG = {
    "Name": "BoschDocumentLevel+RAG",
    "PP": False,
    "RAG": True,
    "FSP": False,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": True,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

BoschDocumentLevelFSP = {
    "Name": "BoschDocumentLevel+FSP",
    "PP": False,
    "RAG": False,
    "FSP": True,
    "JudgeAgent": False,
    "DatasetTRAM": False,
    "DatasetBosch": True,
    "DocumentLevel": True,
    "NumberOfLabel": label_split.BOSCH_TECHNIQUES_LABELS
}

