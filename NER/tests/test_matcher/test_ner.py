import spacy
import tempfile
import unittest
from itertools import chain
from spacy.matcher import Matcher
from spacy_extensions.matcher import SentenceMatcher, SubphraseMatcher
from spacy_extensions.pipeline.ner import NERMatcher
from spacy_extensions.matcher.utils import sentence2pattern


class TestMatcherSentence(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = spacy.load('en_core_web_sm')


    def test_ner_matcher(self) -> None:
        # Initialise matcher
        matcher = Matcher(self.nlp.vocab)
        matcher.add("S0446", [[{"LOWER": "ryuk"}]])
        matcher.add("S0575", [[{"LOWER": "conti"}]])

        # Add custom matcher to pipeline
        self.nlp.disable_pipe('ner')
        ner : NERMatcher = self.nlp.add_pipe('ner_matcher')
        ner.add('matcher', matcher)

        # Pass text
        doc = self.nlp(
            "In one of the Ryuk attacks reconnaissance of the network was done "
            "using the tool AdFind by executing a script called “adf.bat”. This"
            " script was also used in the attack where Hive ransomware was "
            "deployed as well as in the Conti attack detailed by TheDFIRReport"
            "[6]."
        )

        # Target entities
        targets = [
            ("S0446",  4,  5), # Ryuk  in first sentence
            ("S0575", 43, 44), # Conti in final sentence
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)


    def test_ner_sentence(self) -> None:
        # Initialise matcher
        matcher = SentenceMatcher(self.nlp.vocab)
        pattern = lambda x: sentence2pattern(
            sentence = self.nlp(x),
            token_pattern = lambda x: {"LOWER": x.lower_},
        )
        matcher.add("Txxxx" , [pattern("Network Reconnaissance")])
        matcher.add("TA0011", [pattern("Command and Control")])

        # Add custom matcher to pipeline
        self.nlp.disable_pipe('ner')
        ner : NERMatcher = self.nlp.add_pipe('ner_matcher')
        ner.add('matcher', matcher)

        # Pass text
        doc = self.nlp(
            "In one of the Ryuk attacks reconnaissance of the network was done "
            "using the tool AdFind by executing a script called “adf.bat”. This"
            " script was also used in the attack where Hive ransomware was "
            "deployed as well as in the Conti attack detailed by TheDFIRReport"
            "[6]."
        )
        # Target entities
        targets = [
            ("Txxxx", 6, 10), # reconnaissance of the network
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = self.nlp(
            "The user was in control when using their program through the "
            "command line interface."
        )
        # Target entities
        targets = [
            ("TA0011", 4, 12), # control when ... the command
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = self.nlp("The command & control center was taken down.")
        # Target entities
        targets = [
            ("TA0011", 1, 4), # control when ... the command
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)


    def test_ner_subphrase(self) -> None:
        # Initialise matcher
        matcher = SubphraseMatcher(self.nlp.vocab)
        pattern = lambda x: sentence2pattern(
            sentence = self.nlp(x),
            token_pattern = lambda x: {"LOWER": x.lower_},
        )
        matcher.add("Txxxx" , [pattern("Network Reconnaissance")])
        matcher.add("TA0011", [pattern("Command and Control")])

        # Add custom matcher to pipeline
        self.nlp.disable_pipe('ner')
        ner : NERMatcher = self.nlp.add_pipe('ner_matcher')
        ner.add('matcher', matcher)

        # Pass text
        doc = self.nlp(
            "In one of the Ryuk attacks reconnaissance of the network was done "
            "using the tool AdFind by executing a script called “adf.bat”. This"
            " script was also used in the attack where Hive ransomware was "
            "deployed as well as in the Conti attack detailed by TheDFIRReport"
            "[6]."
        )
        # Target entities
        targets = [
            ("Txxxx", 6, 10), # reconnaissance of the network
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = self.nlp(
            "The user was in control when using their program through the "
            "command line interface."
        )
        # Target entities
        targets = []
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = self.nlp("The command & control center was taken down.")
        # Target entities
        targets = [
            ("TA0011", 1, 4), # control when ... the command
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)


    def test_ner_multiple(self) -> None:
        # Initialise matcher
        matcher = Matcher(self.nlp.vocab)
        matcher_sentence = SentenceMatcher(self.nlp.vocab)
        matcher_subphrase = SubphraseMatcher(self.nlp.vocab)
        pattern = lambda x: sentence2pattern(
            sentence = self.nlp(x),
            token_pattern = lambda x: {"LOWER": x.lower_},
        )
        matcher.add("S0446", [[{"LOWER": "ryuk"}]])
        matcher.add("S0575", [[{"LOWER": "conti"}]])
        matcher_sentence.add("Txxxx" , [pattern("Network Reconnaissance")])
        matcher_subphrase.add("TA0011", [pattern("Command and Control")])

        # Add custom matcher to pipeline
        self.nlp.disable_pipe('ner')
        ner : NERMatcher = self.nlp.add_pipe('ner_matcher')
        ner.add('matcher', matcher)
        ner.add('matcher_sentence', matcher_sentence)
        ner.add('matcher_subphrase', matcher_subphrase)

        # Pass text
        doc = self.nlp(
            "In one of the Ryuk attacks reconnaissance of the network was done "
            "using the tool AdFind by executing a script called “adf.bat”. This"
            " script was also used in the attack where Hive ransomware was "
            "deployed as well as in the Conti attack detailed by TheDFIRReport"
            "[6]."
        )
        # Target entities
        targets = [
            ("S0446",  4,  5), # Ryuk  in first sentence
            ("Txxxx",  6, 10), # reconnaissance of the network
            ("S0575", 43, 44), # Conti in final sentence
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = self.nlp(
            "The user was in control when using their program through the "
            "command line interface."
        )
        # Target entities
        targets = []
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = self.nlp("The command & control center was taken down.")
        # Target entities
        targets = [
            ("TA0011", 1, 4), # control when ... the command
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)


    def test_ner_io(self) -> None:
        # Initialise matcher
        matcher = Matcher(self.nlp.vocab)
        matcher_sentence = SentenceMatcher(self.nlp.vocab)
        matcher_subphrase = SubphraseMatcher(self.nlp.vocab)
        pattern = lambda x: sentence2pattern(
            sentence = self.nlp(x),
            token_pattern = lambda x: {"LOWER": x.lower_},
        )
        matcher.add("S0446", [[{"LOWER": "ryuk"}]])
        matcher.add("S0575", [[{"LOWER": "conti"}]])
        matcher_sentence.add("Txxxx" , [pattern("Network Reconnaissance")])
        matcher_subphrase.add("TA0011", [pattern("Command and Control")])

        # Add custom matcher to pipeline
        self.nlp.disable_pipe('ner')
        ner : NERMatcher = self.nlp.add_pipe('ner_matcher')
        ner.add('matcher', matcher)
        ner.add('matcher_sentence', matcher_sentence)
        ner.add('matcher_subphrase', matcher_subphrase)

        # Write pipe to disk
        with tempfile.NamedTemporaryFile('wb') as tmpfile:
            ner.to_disk(tmpfile.name)
            nlp_new = spacy.load('en_core_web_sm')
            nlp_new.disable_pipe('ner')
            ner : NERMatcher = nlp_new.add_pipe('ner_matcher')
            ner.from_disk(tmpfile.name)

        # Pass text
        doc = nlp_new(
            "In one of the Ryuk attacks reconnaissance of the network was done "
            "using the tool AdFind by executing a script called “adf.bat”. This"
            " script was also used in the attack where Hive ransomware was "
            "deployed as well as in the Conti attack detailed by TheDFIRReport"
            "[6]."
        )
        # Target entities
        targets = [
            ("S0446",  4,  5), # Ryuk  in first sentence
            ("Txxxx",  6, 10), # reconnaissance of the network
            ("S0575", 43, 44), # Conti in final sentence
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = nlp_new(
            "The user was in control when using their program through the "
            "command line interface."
        )
        # Target entities
        targets = []
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)

        # Pass text
        doc = nlp_new("The command & control center was taken down.")
        # Target entities
        targets = [
            ("TA0011", 1, 4), # control when ... the command
        ]
        # Found entities
        entities = [(ent.label_, ent.start, ent.end) for ent in doc.ents]
        # Assert equals
        self.assertEqual(entities, targets)


    def test_extensions(self) -> None:
        # Initialise matcher
        matcher = Matcher(self.nlp.vocab)
        matcher_sentence = SentenceMatcher(self.nlp.vocab)
        pattern = lambda x: sentence2pattern(
            sentence = self.nlp(x),
            token_pattern = lambda x: {"LOWER": x.lower_},
        )
        matcher.add("S0446", [[{"LOWER": "ryuk"}]])
        matcher_sentence.add("Txxxx" , [pattern("Ryuk reconnaissance")])

        # Add custom matcher to pipeline
        self.nlp.disable_pipe('ner')
        ner : NERMatcher = self.nlp.add_pipe('ner_matcher', config={
            'extension': 'test_name',
            'force': False,
        })
        ner.add('matcher', matcher)
        ner.add('matcher_sentence', matcher_sentence)

        # Pass text
        doc = self.nlp(
            "Ryuk is a malware. It performed 'Ryuk reconnaissance'."
        )
        # Target entities
        targets = [
            ("S0446", 0,  1), # Ryuk in first sentence
            ("S0446", 8,  9), # Ryuk in second sentence
            ("Txxxx", 8, 10), # Ryuk reconnaissance
        ]
        # Found entities
        entities = [
            (span.label_, span.start, span.end) for span in doc.ents
        ]
        entities_extension = [
            (span.label_, span.start, span.end) for span in
            chain.from_iterable(doc._.test_name)
        ]
        # Assert equals
        self.assertEqual(entities, [targets[0], targets[2]])
        self.assertEqual(entities_extension, targets)

        