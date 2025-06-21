import pandas as pd


def convert(input_file):
    all_sentences = []

    df_raw = pd.read_json(input_file, lines=True)
    df_report_list = df_raw.groupby("cti_report")

    i = 0
    for _, cti in df_report_list:
        i += 1
        print(i)

        for index, row in cti.iterrows():
            cti_id_label = []
            sentence = ""

            # ATT&CK
            label = row["label_link"].split("/")[-1]
            if (
                label.startswith("T")
                or label.startswith("S")
                or label.startswith("M")
                or label.startswith("G")
            ):
                cti_id_label.append(label)

            # ENTITIES
            entities = [(row["entity_type"], row["label_title"])]

            sentence = row["_context_left"] + row["mention"] + row["_context_right"]

            # sentence = cti_preprocessing.replace_iocs(sentence)

            for i, saved_sentence in enumerate(all_sentences):
                if saved_sentence["sentence"] == sentence:
                    all_sentences[i]["labels"].extend(cti_id_label)
                    if entities[0] not in all_sentences[i]["entities"]:
                        all_sentences[i]["entities"].extend(entities)
                    sentence = ""

            if sentence != "":
                all_sentences.append(
                    {
                        "sentence": sentence,
                        "labels": cti_id_label,
                        "entities": entities,
                        "document": row["document"],
                    }
                )
    return all_sentences


def convert_tram(df_file):
    df = pd.read_json(df_file)
    out = []
    for i, row in df.iterrows():
        sentences = row['cti_report']
        label_set = row['label']
        doc_title = row['doc_title']
        for sentence, labels in zip(sentences, label_set):
            out.append({
                "sentence": sentence,
                "labels": labels,
                "doc_title": doc_title,
            })
    return out

if __name__ == "__main__":
    train_file = "datasets/bosch_cti_devtrain_ds.json"
    test_file = "datasets/bosch_cti_test_ds.json"
    train_sentences = convert_tram(train_file)
    test_sentences = convert_tram(test_file)
    # train_df.to_json("datasets/bosch_train.json")
    # test_df.to_json("datasets/bosch_test.json")
    # train_file = "datasets/tram_train_aggregated.json"
    # test_file = "datasets/tram_test_aggregated.json"
    # train_sentences = convert_tram(train_file)
    # test_sentences = convert_tram(test_file)
    train_df = pd.DataFrame(train_sentences)
    test_df = pd.DataFrame(test_sentences)
    train_df.to_json("datasets/bosch_train.json")
    test_df.to_json("datasets/bosch_test.json")