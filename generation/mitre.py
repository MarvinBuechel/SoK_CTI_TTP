import pandas
import pandas as pd
import random

import rag
import finetuning.cti_datasets.tram.tram_classes as tram

# Collection of the used prompts.

def create_mitre_based_prompt(report: str, few_shot=False, only_techniques=False):
    techniques = "(Sub)Technique"
    if only_techniques:
        techniques = "Technique"

    query = (
            "List all MITRE ATT&CK " + techniques + " IDs and Names that occur in the CTI report at the end, surrounded by grave accents (`).\n Do not give any assumptions or possibilities! Please provide concise, non-repetitive answers.\n"
    )

    if few_shot:
        query += "\n\nA few examples for help (these examples are independent of the sentence to analyze).\n"
        query += "-Sentence: \"TrickBot has used macros in Excel documents to download and deploy the malware on the user’s machine.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "-Sentence: \"SombRAT has the ability to use an embedded SOCKS proxy in C2 communications.\". Your Response: \"- T1090: Proxy\"\n"
        query += "-Sentence: \"Azorult can collect a list of running processes by calling CreateToolhelp32Snapshot.\". Your Response: \"- T1057: Process Discovery\"\n"
        query += "-Sentence: \"SeaDuke is capable of executing commands.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "-Sentence: \"Figure 4 shows the architecture of Crutch v4.\". Your Response: \"No technique found.\"\n"

    query += "CTI report: `" + report + "`"

    return query

def create_mitre_based_rag_prompt(report: str, closest_embeddings, mitre_table, rag_entries = 5, few_shot=False,
                                           is_bosch=False, label_to_limit = None):
    techniques = "(Sub)Technique"
    if is_bosch:
        techniques = "Technique"

    query = (
            "List all MITRE ATT&CK " + techniques + " IDs and Names that occur in the CTI report at the end, surrounded by grave accents (`). "
            "The MITRE ATT&CK descriptions below (delimited by triple quotation marks (\"\"\") may or may not contain MITRE ATT&CK " + techniques + " used in the sentence below.\n"
            "Do not give any assumptions, tips or similar! Please provide concise, non-repetitive answers.\n")

    query += "MITRE ATT&CK Techniques:\n\"\"\"\n"
    num = 0
    for i, embed in enumerate(closest_embeddings):
        id = mitre_table.at[embed["index"], "ID"]

        # Label Limits
        if label_to_limit is not None and is_bosch and id not in label_to_limit:
            continue

        if is_bosch is False and id not in tram.TRAM_CLASSES:
            continue

        if num > rag_entries:
            break
        num += 1

        query += "[\n"
        query += "MITRE ATT&CK ID: " + mitre_table.at[embed["index"], "ID"] + "\n"
        query += "MITRE ATT&CK Name: " + mitre_table.at[embed["index"], "name"] + "\n"
        query += "MITRE ATT&CK Description: " + mitre_table.at[
            embed["index"], "description"] + "\n"
        query += "],\n"
    query += "\n\"\"\"\n"

    if few_shot:
        query += "A few examples for help (these examples are independent of the sentence to analyze).\n"
        query += "For example: -Sentence: \"TrickBot has used macros in Excel documents to download and deploy the malware on the user’s machine.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "For example: -Sentence: \"SombRAT has the ability to use an embedded SOCKS proxy in C2 communications.\". Your Response: \"- T1090: Proxy\"\n"
        query += "For example: -Sentence: \"Silent Librarian has exfiltrated entire mailboxes from compromised accounts.\". Your Response: \"- T1114: Email Collection\"\n"
        query += "For example: -Sentence: \"Azorult can collect a list of running processes by calling CreateToolhelp32Snapshot.\". Your Response: \"- T1057: Process Discovery\"\n"
        query += "For example: -Sentence: \"SeaDuke is capable of executing commands.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "For example: -Sentence: \"Figure 4 shows the architecture of Crutch v4.\". Your Response: \"No technique found.\"\n"


    query += ("\nCTI report: "
              "`" + report + "`")

    return query


def judge_sentence_based(sentence: str, candidates):
    query = "Are the following MITRE ATT&CK (Sub)Sechniques included in the sentence from a CTI report at the end, surrounded by grave accents (`)?\n" \
            "MITRE ATT&CK (Sub)Techniques:"
    for candidate in candidates:
        shortened_desc = candidate[2].split("\n")[0]
        query += f"\n- {candidate[0]} - {candidate[1]}: {shortened_desc}"

    query += f"\n\nSentence to analyze: `{sentence}`\nWhich (Sub)Technique IDs are present in the content of the sentence?\nAll the techniques you mention are treated as present."

    return query


def create_mitre_sentence_based_prompt(sentence: str, few_shot=False, only_techniques=False):
    techniques = "(Sub)Technique"
    if only_techniques:
        techniques = "Technique"

    query = (
            "List all MITRE ATT&CK " + techniques + " IDs and Names that occur in the sentence from a CTI report, surrounded by grave accents (`). It is also possible that there is "
            "no MITRE ATT&CK " + techniques + " in this sentence.\n Do not give any assumptions, possibilities or reasons! Please provide concise, non-repetitive answers.\nSentence to analyze: "
            "`" + sentence + "`\n"
    )

    if few_shot:
        query += "\n\nA few examples for help (these examples are independent of the sentence to analyze).\n"
        query += "-Sentence: \"TrickBot has used macros in Excel documents to download and deploy the malware on the user’s machine.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "-Sentence: \"SombRAT has the ability to use an embedded SOCKS proxy in C2 communications.\". Your Response: \"- T1090: Proxy\"\n"
        query += "-Sentence: \"Azorult can collect a list of running processes by calling CreateToolhelp32Snapshot.\". Your Response: \"- T1057: Process Discovery\"\n"
        query += "-Sentence: \"SeaDuke is capable of executing commands.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "-Sentence: \"Figure 4 shows the architecture of Crutch v4.\". Your Response: \"No technique found.\"\n"

    return query


def create_mitre_sentence_based_rag_prompt(sentence: str, closest_embeddings, mitre_table, rag_entries = 5, few_shot=False,
                                           is_bosch=False, label_to_limit = None):
    techniques = "(Sub)Technique"
    if is_bosch is not None:
        techniques = "Technique"

    query = (
            "List all MITRE ATT&CK " + techniques + " IDs and Names that occur in the sentence from a CTI report, surrounded by grave accents (`). This " 
            "sentence is a single sentence from a whole cyber threat report. It is also possible that there is " 
            "no MITRE ATT&CK " + techniques + " in this sentence. "  # Maybe helpful The smaller the Sentence distance, the higher the probability that the sentence matches the technique.
            "The MITRE ATT&CK descriptions below (delimited by triple quotation marks (\"\"\") may or may not contain MITRE ATT&CK " + techniques + " used in the sentence below.\n"
            "Do not give any assumptions, tips or similar! Please provide concise, non-repetitive answers.\n")

    query += "MITRE ATT&CK Techniques:\n\"\"\"\n"
    num = 0
    for i, embed in enumerate(closest_embeddings):
        id = mitre_table.at[embed["index"], "ID"]

        # Label Limits
        if label_to_limit is not None and is_bosch and id not in label_to_limit:
            continue

        if is_bosch is False and id not in tram.TRAM_CLASSES:
            continue

        if num > rag_entries:
            break
        num += 1

        query += "{\n"
        query += "MITRE ATT&CK ID: " + mitre_table.at[embed["index"], "ID"] + "\n"
        query += "MITRE ATT&CK Name: " + mitre_table.at[embed["index"], "name"] + "\n"
        # query += "Sentence Distance: " + str(closest_embeddings[i]["distance"].round(2)) + "\n"
        query += "MITRE ATT&CK Description: " + mitre_table.at[
            embed["index"], "description"] + "\n"  # first paragraph: .split("\n\n")[0]
        query += "}\n"
    query += "\n\"\"\"\n"

    if few_shot:
        query += "A few examples for help (these examples are independent of the sentence to analyze).\n"
        query += "For example: -Sentence: \"TrickBot has used macros in Excel documents to download and deploy the malware on the user’s machine.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "For example: -Sentence: \"SombRAT has the ability to use an embedded SOCKS proxy in C2 communications.\". Your Response: \"- T1090: Proxy\"\n"
        query += "For example: -Sentence: \"Silent Librarian has exfiltrated entire mailboxes from compromised accounts.\". Your Response: \"- T1114: Email Collection\"\n"
        query += "For example: -Sentence: \"Azorult can collect a list of running processes by calling CreateToolhelp32Snapshot.\". Your Response: \"- T1057: Process Discovery\"\n"
        query += "For example: -Sentence: \"SeaDuke is capable of executing commands.\". Your Response: \"- T1059: Command and Scripting Interpreter\"\n"
        query += "For example: -Sentence: \"Figure 4 shows the architecture of Crutch v4.\". Your Response: \"No technique found.\"\n"


    query += ("\nSentence to analyze: "
              "`" + sentence + "`\n")

    return query


