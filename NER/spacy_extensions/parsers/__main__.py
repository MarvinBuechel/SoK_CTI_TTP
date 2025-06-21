from urllib.parse import urlparse
import argformat
import argparse
from spacy_extensions.parsers.anomali import ParserAnomali
from spacy_extensions.parsers.att import ParserAtt
from spacy_extensions.parsers.base import ParserHTML
from spacy_extensions.parsers.bitdefender import ParserBitdefender
from spacy_extensions.parsers.cybereason import ParserCybereason
from spacy_extensions.parsers.deepinstinct import ParserDeepinstinct
from spacy_extensions.parsers.dragos import ParserDragos
from spacy_extensions.parsers.eclecticlight import ParserEclecticlight
from spacy_extensions.parsers.fireeye import ParserFireeye
from spacy_extensions.parsers.forcepoint import ParserForcepoint
from spacy_extensions.parsers.github import ParserGithub
from spacy_extensions.parsers.hexacorn import ParserHexacorn
from spacy_extensions.parsers.infosecblog import ParserInfosecblog
from spacy_extensions.parsers.lolbas import ParserLolbas
from spacy_extensions.parsers.malwarebytes import ParserMalwarebytes
from spacy_extensions.parsers.mandiant import ParserMandiant
from spacy_extensions.parsers.mcafee import ParserMcafee
from spacy_extensions.parsers.medium import ParserMedium
from spacy_extensions.parsers.microsoft import ParserMicrosoft
from spacy_extensions.parsers.nist import ParserNist
from spacy_extensions.parsers.office import ParserOffice
from spacy_extensions.parsers.paloaltonetworks import ParserPaloaltonetworks
from spacy_extensions.parsers.security import ParserSecurity
from spacy_extensions.parsers.securityintelligence import ParserSecurityintelligence
from spacy_extensions.parsers.securityweek import ParserSecurityweek
from spacy_extensions.parsers.sophos import ParserSophos
from spacy_extensions.parsers.sucuri import ParserSucuri
from spacy_extensions.parsers.talosintelligence import ParserTalosintelligence
from spacy_extensions.parsers.taosecurity import ParserTaosecurity
from spacy_extensions.parsers.trendmicro import ParserTrendmicro
from spacy_extensions.parsers.twitter import ParserTwitter
from spacy_extensions.parsers.virusbulletin import ParserVirusbulletin
from spacy_extensions.parsers.webroot import ParserWebroot
from spacy_extensions.parsers.welivesecurity import ParserWelivesecurity

def parse_args():
    """Parse arguments from command line."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description     = 'Parse an HTML file to extract text',
        formatter_class = argformat.StructuredFormatter,
    )

    # Add arguments
    parser.add_argument('html'   , help='path to HTML file to parse')
    parser.add_argument('outfile', help='path to output text file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--parser', help='parser to use for parsing')
    group.add_argument('--url'   , help='original url to infer parser')

    # Parse arguments and return
    return parser.parse_args()


def get_parser(parser: str) -> ParserHTML:
    """Get parser from string. Selects the given parer """
    # Get parser as lowercase
    parser = parser.lower()

    if 'alienvault' in parser: # alienvault now redirects to AT&T
        return ParserAtt()
    elif 'anomali.com' in parser:
        return ParserAnomali()
    elif 'cybersecurity.att.com' in parser:
        return ParserAtt()
    if 'bitdefender' in parser:
        return ParserBitdefender()
    elif 'cybereason' in parser:
        return ParserCybereason()
    elif 'deepinstinct' in parser:
        return ParserDeepinstinct()
    elif 'dragos' in parser:
        return ParserDragos()
    elif 'eclecticlight' in parser:
        return ParserEclecticlight()
    elif 'fireeye' in parser:
        return ParserFireeye()
    elif 'forcepoint' in parser:
        return ParserForcepoint()
    elif 'github.com' in parser:
        return ParserGithub()
    elif 'hexacorn' in parser:
        return ParserHexacorn()
    elif 'infosecblog' in parser:
        return ParserInfosecblog()
    elif 'lolbas-project.github.io' in parser:
        return ParserLolbas()
    elif 'malwarebytes' in parser:
        return ParserMalwarebytes()
    elif 'mandiant' in parser:
        return ParserMandiant()
    elif 'mcafee' in parser:
        return ParserMcafee()
    elif 'medium' in parser:
        return ParserMedium()
    elif 'microsoft' in parser:
        return ParserMicrosoft()
    elif 'nist' in parser:
        return ParserNist()
    elif 'paloaltonetworks' in parser:
        return ParserPaloaltonetworks()
    elif '.security.com' in parser:
        return ParserSecurity()
    elif 'securityweek' in parser:
        return ParserSecurityweek()
    elif 'securityintelligence' in parser:
        return ParserSecurityintelligence()
    elif 'support.office' in parser:
        return ParserOffice()
    elif 'sophos' in parser:
        return ParserSophos()
    elif 'sucuri' in parser:
        return ParserSucuri()
    elif 'symantec' in parser: # Symantec now redirects to security.com
        return ParserSecurity()
    elif 'talosintelligence' in parser:
        return ParserTalosintelligence()
    elif 'taosecurity' in parser:
        return ParserTaosecurity()
    elif 'trendmicro' in parser:
        return ParserTrendmicro()
    elif 'twitter' in parser:
        return ParserTwitter()
    elif 'virusbulletin' in parser:
        return ParserVirusbulletin()
    elif 'webroot' in parser:
        return ParserWebroot()
    elif 'welivesecurity' in parser:
        return ParserWelivesecurity()
    else:
        raise ValueError(f"No parser supported for '{parser}'")


def main():
    # Parse arguments
    args = parse_args()

    # Load HTML file
    with open(args.html) as infile:
        html = infile.read()

    # Infer parser from URL
    if args.url:
        url = urlparse(args.url)
        args.parser = url.netloc or url.path
    
    # Set parser
    parser = get_parser(args.parser)

    # Parse html
    text = parser.parse(html)

    # Write to outfile
    with open(args.outfile, 'w') as outfile:
        outfile.write(text)


if __name__ == '__main__':
    main()