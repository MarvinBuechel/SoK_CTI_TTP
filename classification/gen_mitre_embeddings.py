from mitreattack.stix20 import MitreAttackData
from loader import load_model_for_embedding
from const import MODEL_SENTENCE_SIM
from tqdm import tqdm
import pickle


def main():
    mitre_attack_data = MitreAttackData("datasets/enterprise-attack.json")
    techniques = mitre_attack_data.get_techniques(remove_revoked_deprecated=True)
    print(f"Retrieved {len(techniques)} ATT&CK techniques ...")
    ttps = []
    models = {}
    for t in techniques:
        technique_id = [
            e for e in t.external_references if e.source_name == "mitre-attack"
        ][0].external_id
        ttps.append({"id": technique_id, "title": t.name, "description": t.description})
    ttps_incomplete_title = sorted(ttps, key=lambda x: x["id"])

    ttps = []
    for ttp in ttps_incomplete_title:
        id_parts = ttp["id"].split(".")
        if len(id_parts) > 1:
            print("fixing title for subtechnique %s" % ".".join(id_parts))
            parent_id = id_parts[0]
            parent_title = [ttp["title"] for ttp in ttps_incomplete_title if ttp["id"] == parent_id][0]
            new_title = "%s (%s)" % (parent_title, ttp["title"])
            ttps.append({
                "id": ttp["id"],
                "title": new_title,
                "description": ttp["description"],
                "content": "%s:\n%s" % (new_title, ttp["description"])
            })
        else:
            ttps.append({
                "id": ttp["id"],
                "title": ttp["title"],
                "description": ttp["description"],
                "content": "%s:\n%s" % (ttp["title"], ttp["description"])
            })


    for mdl in MODEL_SENTENCE_SIM:
        models[mdl] = load_model_for_embedding(mdl)

    output = {}
    for mdl in list(models.keys()):
        for ttp in tqdm(ttps, desc=f"{mdl}"):
            if ttp["id"] not in output:
                output[ttp["id"]] = {}
            output[ttp["id"]][mdl] = {}
            output[ttp["id"]]["id"] = ttp["id"]
            emb = models[mdl].encode([ttp["content"]])
            output[ttp["id"]][mdl]["emb"] = emb 

    outfile = "datasets/mitre_embeddings.pickle"
    with open(outfile, "wb") as f:
        pickle.dump(output, f)
    print(f"embedding vectors written at {outfile}")


if __name__ == "__main__":
    main()
