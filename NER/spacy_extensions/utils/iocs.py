"""This document specifies regexes for various indicators of compromise (IOCs)
    used by the MatcherIoc SpaCy extension.
    """

# Imports
import argformat
import argparse
import json
import os
import pathlib
import re
import requests
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Union

################################################################################
#                              Retrieve resources                              #
################################################################################


def get_file_extensions() -> List[str]:
    """Return various extensions from https://fileinfo.com/filetypes/common."""
    return [
        'doc', 'docx', 'log', 'odt', 'rtf', 'tex', 'txt', # Text files
        'bin', 'csv', 'dat', 'ppt', 'pptx', 'xml', # Data files
        'accdb', 'crypt14', 'db', 'dbf', 'mdb', 'pdf', 'sql', 'sqlite', 'sqlite3', 'sqlitedb', # Database files
        'apk', 'app', 'bat', 'bin', 'cgi', 'exe', 'jar', 'run', 'wsf', 'sh', 'so', 'mobile', 'py', # Executables
        'dll', 'sys', 'dmp', # System files
        'cfg', 'ini', # Settings
        'enc', # Encrypted files
        '7z', 'deb', 'gz', 'pak', 'pkg', 'rar', 'rpm', 'tar', 'xapk', 'zip', 'zipx' # Compressed files
    ]


def get_tlds(
        file: Optional[str] = pathlib.Path(__file__).absolute().parent.parent / 'resources' / 'tlds.txt',
        url : Optional[str] = 'https://data.iana.org/TLD/tlds-alpha-by-domain.txt',
    ) -> List[str]:
    """Returns all top level domains.
    
        Parameters
        ----------
        file : Optional[str], default = pathlib.Path(__file__).absolute().parent.parent / 'resources' / 'tlds.txt'
            If given and exists, read the top level domains from the given file.
            
        url : Optional[str], default = 'https://data.iana.org/TLD/tlds-alpha-by-domain.txt'
            If no file was given or could be found, download top level domains
            from given url.

        Returns
        -------
        tlds : List[str]
            List of top level domains.
        """
    # First try to get TLDs from file if it exists
    if os.path.isfile(file):
        with open(file) as infile:
            tlds = infile.read()
    # Otherwise, download TLDs from URL
    else:
        tlds = requests.get(url).text

    # Filter comments from top level domains
    tlds = filter(None, tlds.split('\n'))
    tlds = filter(lambda x: not x.startswith('#'), tlds)
    tlds = filter(lambda x: not '-' in x, tlds)
    # Get as lower case
    tlds = map(lambda x: x.lower(), tlds)
    # Return result
    return list(sorted(tlds, key=lambda x: len(x), reverse=True))


def get_uri_schemes(
        file: Optional[str] = pathlib.Path(__file__).absolute().parent.parent / 'resources' / 'uri-schemes.txt',
        url : Optional[str] = 'https://www.iana.org/assignments/uri-schemes/uri-schemes.txt',
    ) -> List[str]:
    """Returns all IANA defined URI schemes.
    
        Parameters
        ----------
        file : Optional[str], default = pathlib.Path(__file__).absolute().parent.parent / 'resources' / 'uri-schemes.txt'
            If given and exists, read the supported URI schemes from the given
            file.
            
        url : Optional[str], default = 'https://www.iana.org/assignments/uri-schemes/uri-schemes.txt'
            If no file was given or could be found, download supported URI =
            schemes from the given url.

        Returns
        -------
        uris : List[str]
            Set of top level domains.
        """
    # First try to get URI schemes from file if it exists
    if os.path.isfile(file):
        with open(file, encoding='utf-8') as infile:
            uris = infile.read()
    # Otherwise, download URI schemes from URL
    else:
        uris = requests.get(url).text

    # Get URI schemes
    result = list()
    for uri in uris.split('\n'):
        if len(result) or uri.strip().startswith('URI Scheme'):
            result.append(uri)
        if uri.strip().startswith('Contact Information'): break
    uris = result[2:-2]

    uris = filter(lambda x: x.strip()[0] == x[0], uris)
    uris = map   (lambda x: x.split()[0], uris)
    uris = map   (lambda x: x.lower(), uris)

    # Return uris
    return list(uris)

################################################################################
#                             Regular expressions                              #
################################################################################

# IOC helper expressions
FANGS_OPEN  = r'([\[\(\\{<])'
FANGS_CLOSE = r'([\]\)\\}>])'
IPv4SEG     = r'(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])'
IPv6SEG     = r'[0-9a-fA-F]{1,4}'
IPv6SEP     = rf'{FANGS_OPEN}*:{FANGS_CLOSE}*'
DOT_SEP     = rf'({FANGS_OPEN}*(\.|([dD][oO][tT])){FANGS_CLOSE}*)'
AT_SEP      = rf'({FANGS_OPEN}+(@|([aA][tT])){FANGS_CLOSE}+)'
EXTENSIONS  = map(lambda x: ''.join(f"[{char.lower()}{char.upper()}]" for char in x), get_file_extensions())
TLDS        = rf"({'|'.join(get_tlds())})"
URI_SCHEMES = rf"({'|'.join(get_uri_schemes())})"
HEX         = r'[0-9a-fA-F]'
BASE64      = r'[0-9a-zA-Z+/]+==?'

EXTENSIONS = rf'{DOT_SEP}({"|".join(EXTENSIONS)})'
PATH_BASE  = r'(?:[a-zA-Z0-9]+[\\\/])*[a-zA-Z0-9]+'
PATH       = rf'(([a-zA-Z]:)|\s)[\\\/]{PATH_BASE}'

# IP address expressions
IPv4   = DOT_SEP.join([IPv4SEG]*4)
PORT   = r":(6553[0-5]|655[0-2][0-9]{2}|65[0-4][0-9]{3}|6[0-4][0-9]{4}|[0-5][0-9]{4}|[1-9][0-9]{3}|[1-9][0-9]{2}|[1-9][0-9]|[0-9])"
SUBNET = r"(\/(3[0-2]|[1-2][0-9]|[0-9]))"
IPv6 = (
    rf"({IPv6SEG}{IPv6SEP}){{7,7}}{IPv6SEG}|"                          # 1:2:3:4:5:6:7:8
    rf"{IPv6SEP}(({IPv6SEP}{IPv6SEG}){{1,7}}|{IPv6SEP})|"              # ::2:3:4:5:6:7:8    ::2:3:4:5:6:7:8  ::8       ::  
    rf"{IPv6SEG}{IPv6SEP}(({IPv6SEP}{IPv6SEG}){{1,6}})|"               # 1::3:4:5:6:7:8     1::3:4:5:6:7:8   1::8    
    rf"({IPv6SEG}{IPv6SEP}){{1,2}}({IPv6SEP}{IPv6SEG}){{1,5}}|"        # 1::4:5:6:7:8       1:2::4:5:6:7:8   1:2::8
    rf"({IPv6SEG}{IPv6SEP}){{1,3}}({IPv6SEP}{IPv6SEG}){{1,4}}|"        # 1::5:6:7:8         1:2:3::5:6:7:8   1:2:3::8
    rf"({IPv6SEG}{IPv6SEP}){{1,4}}({IPv6SEP}{IPv6SEG}){{1,3}}|"        # 1::6:7:8           1:2:3:4::6:7:8   1:2:3:4::8
    rf"({IPv6SEG}{IPv6SEP}){{1,5}}({IPv6SEP}{IPv6SEG}){{1,2}}|"        # 1::7:8             1:2:3:4:5::7:8   1:2:3:4:5::8
    rf"({IPv6SEG}{IPv6SEP}){{1,6}}{IPv6SEP}{IPv6SEG}|"                 # 1::8               1:2:3:4:5:6::8   1:2:3:4:5:6::8
    rf"({IPv6SEG}{IPv6SEP}){{1,7}}{IPv6SEP}|"                          # 1::                                 1:2:3:4:5:6:7::
    rf"fe80{IPv6SEP}({IPv6SEP}{IPv6SEG}){{0,4}}%[0-9a-zA-Z]{{1,}}|"    # fe80::7:8%eth0     fe80::7:8%1  (link-local IPv6 addresses with zone index)
    rf"{IPv6SEP}(ffff({IPv6SEP}0{{1,4}}){{0,1}}{IPv6SEP}){0,1}{IPv4}|" # ::255.255.255.255  ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped IPv6 addresses and IPv4-translated addresses)
    rf"({IPv6SEG}{IPv6SEP}){{1,4}}{IPv6SEP}{IPv4}"                     # 2001:db8:3:4::192.0.2.33  64:ff9b::192.0.2.33 (IPv4-Embedded IPv6 Address)
)
IPv6_full = rf"{IPv6}({PORT})?"
IPv4_full = rf"{IPv4}({PORT}|{SUBNET})?"

URL_ENC   = r"(%[0-9a-fA-F]{2})"
URL_ENC_R = r"([0-9a-fA-F]{2}%)"
# URL_BASE = rf"([A-Za-z0-9]-_~:\/\?#\[\]@!\$&'\(\)\*\+,;=|{URL_ENC}|{DOT_SEP})+"
URL_BASE = rf"([A-Za-z0-9\-_~:\/\?#\[\]@!\$&'\(\)\*\+,;=]|{URL_ENC}|{DOT_SEP})+"

# URL_BASE = rf"([a-zA-Z0-9]|[$\-_@&+]|{DOT_SEP}|{URL_ENC})+"
URL = rf"{URI_SCHEMES}:\/\/{URL_BASE}" # http[s] address
URL = rf"{URL}({PORT})?(?<!([:\?\[\]@!'\(\),;.]))?"

class URLRegex:
    """Regular expression for quickly detecting URLs
    
        Based on the URL: ``scheme://netloc/path?query#fragment``.
        Where:
        * ``scheme://``: optional, based on any of the given schemes.
        * ``netloc``: required, based on any of the given tlds.
        * ``/path``: optional
        * ``?query``: optional
        * ``#fragment``: optional        
        """

    def __init__(
            self,
            schemes       : Iterable[str],
            tlds          : Iterable[str],
            tld_exceptions: Optional[Iterable[str]] = None,
            # base          : str = rf"([A-Za-z0-9\-_~:\/\[\]@!\$&'\(\)\*\+,;=]|{URL_ENC}|{DOT_SEP})+",
            base          : str = rf"([A-Za-z0-9\-_~:\/@!\$&'\(\)\*\+,;=]|{URL_ENC}|{DOT_SEP})+",
            netloc        : str = rf"([a-zA-Z0-9\-]|{URL_ENC}|{DOT_SEP})+",
            netloc_r      : str = rf"([a-zA-Z0-9\-]|{URL_ENC_R}|{DOT_SEP})+",
            ip            : str = rf"(({IPv4})|({IPv6}))",
            port          : str = rf"({PORT})",
            dot           : str = DOT_SEP,
            flags         : int = 0,
        ):
        """Create regex from given parameters.
        
            Parameters
            ----------
            schemes : Iterable[str]
                URL schemes to support, e.g. "http", "https", "ftp", etc.
            
            tlds : Iterable[str]
                List of top level domains to support.
                E.g., ".com", ".org", ".nl", etc.

            tld_exceptions : Optional[Iterable[str]], optional
                If given, do not return matching tld urls when only the URL
                netloc is present. E.g. ".so" is a valid TLD for Somalia, but it
                is also a file extension used for shared objects. E.g., we do
                want "https://www.somalia.so" to match, but we do not want
                "libnetpbm.so" to match.
            """
        ################################################################
        #                        Set parameters                        #
        ################################################################
        
        self.schemes        = list(schemes)
        self.schemes_r      = [scheme[::-1] for scheme in self.schemes]
        self.tlds           = list(tlds)
        self.tld_exceptions = list(tld_exceptions) if tld_exceptions else None
        self.base           = base
        self.netloc         = netloc
        self.netloc_r       = netloc_r
        self.ip             = ip
        self.port           = port
        self.dot            = dot
        self.flags          = flags

        ################################################################
        #                        Define regexes                        #
        ################################################################

        # Prepare subschemes
        schemes   = f"({'|'.join(sorted(schemes       , key=lambda x: -len(x)))})"
        schemes_r = f"({'|'.join(sorted(self.schemes_r, key=lambda x: -len(x)))})"
        tlds      = f"({'|'.join(sorted(tlds          , key=lambda x: -len(x)))})"

        # Build regular expression(s)
        self.regex_tld     = re.compile(rf"{self.dot}{tlds}", self.flags)
        self.regex_reverse = re.compile(rf"^({netloc_r})(\/\/:{schemes_r})?", self.flags)

        self.regex_scheme   = rf"{schemes}:\/\/"
        self.regex_netloc   = rf"{self.netloc}{self.dot}{tlds}(?![a-zA-Z])({self.port})?"
        self.regex_path     = rf"\/{self.base}|\/"
        self.regex_query    = rf"\?{self.base}"
        self.regex_fragment = rf"#{self.base}"
        self.regex_url = re.compile(
            f"({self.regex_scheme  })?"
            f"(({self.regex_netloc }))"
            f"({self.regex_path    })?"
            f"({self.regex_query   })?"
            f"({self.regex_fragment})?"
            f"(?<![.,;])",
            self.flags
        )
        self.regex_url_scheme = re.compile(
            f"({self.regex_scheme  })"
            f"({self.ip}({self.port})?)"
            f"({self.regex_path    })?"
            f"({self.regex_query   })?"
            f"({self.regex_fragment})?"
            f"(?<![.,;])",
            self.flags
        )
        self.regex_url_path = re.compile(
            f"({self.regex_scheme  })?"
            f"({self.regex_netloc  })"
            f"({self.regex_path    })"
            f"({self.regex_query   })?"
            f"({self.regex_fragment})?",
            self.flags
        )
        self.regex_url_www = re.compile(
            f"({self.regex_scheme  })?"
            f"([wW]{{3}}{self.dot}{self.regex_netloc})"
            f"({self.regex_path    })?"
            f"({self.regex_query   })?"
            f"({self.regex_fragment})?",
            self.flags
        )
        self.regex_url_query = re.compile(
            f"({self.regex_scheme  })?"
            f"({self.regex_netloc  })"
            f"({self.regex_path    })?"
            f"({self.regex_query   })"
            f"({self.regex_fragment})?",
            self.flags
        )
        self.regex_url_fragment = re.compile(
            f"({self.regex_scheme  })?"
            f"({self.regex_netloc  })"
            f"({self.regex_path    })?"
            f"({self.regex_query   })?"
            f"({self.regex_fragment})",
            self.flags
        )

        # Prepare exceptions subscheme if necessary
        if self.tld_exceptions:
            tld_exceptions = f"({'|'.join(sorted(tld_exceptions, key=lambda x: -len(x)))})"
            self.regex_exceptions = re.compile(rf"^{self.dot}{tld_exceptions}", self.flags)
            

    def finditer(self, text: str) -> Iterable[re.Match]:
        """Find all matching regexes quickly.
            Uses the following steps:
            1a: Identify .TLD
            1b: Identify IP
            2: find beginning of URL (either scheme:// or until valid start
            3a. If scheme://, GOTO 4
            3b. If no scheme://, filter out exception tlds; GOTO 4
            4: Apply regular URL scheme

            Parameters
            ----------
            text : str
                Text in which to find URLs.

            Yields
            ------
            match : re.Match
                Matches for URLs in text.
            """
        # Keep track of previously returned spans
        seen = set()

        # Initialise offset from start of text
        offset = 0

        # Iterate over each line in text
        for line in text.split('\n'):

            # 1a. Identify .TLD
            for tld in self.regex_tld.finditer(line):
                # Find beginning of url
                netloc = self.regex_reverse.match(line[tld.start()-1::-1])
                # Skip if we only found a TLD
                if not netloc: continue

                # Get starting position of URL
                start = tld.start() - netloc.end()

                # Skip if we found email address instead
                if start >= 1 and (line[start-1] in {'@', '.', '_'}):
                    continue
                
                if (
                    # If scheme://
                    '://' in netloc.string[netloc.start():netloc.end()][::-1] or
                    # If not scheme://, filter out tld exceptions
                    not (
                        self.tld_exceptions and
                        not self.regex_url_path    .match(text, pos=offset+start) and
                        not self.regex_url_www     .match(text, pos=offset+start) and
                        not self.regex_url_query   .match(text, pos=offset+start) and
                        not self.regex_url_fragment.match(text, pos=offset+start) and
                        self.regex_exceptions.match(tld.group(0))
                    )
                    ):
                    # Get match
                    match = self.regex_url.match(text, pos=offset+start)

                    # Check if match was returned
                    if match and match.span() not in seen:
                        # Add to seen matches
                        seen.add(match.span())
                        # Yield match
                        yield match

            # 1b, get scheme matches (includes IP addresses)
            for match in self.regex_url_scheme.finditer(line):
                # Get match
                match = self.regex_url_scheme.match(text, pos=offset+match.start())

                # Check if match was returned
                if match and match.span() not in seen:
                    # Add to seen matches
                    seen.add(match.span())
                    # Yield match
                    yield match
                           

            # Add to offset
            offset += len(line) + 1


    def __eq__(self, other):
        """Set equivalence relation."""
        return self.regex_url == other.regex_url
        

# ################################################################################
# #                             Quick regex matching                             #
# ################################################################################

# class URLRegex:

#     def __init__(self, tlds: Iterable[str], flags: Optional[Any] = None):
#         """Regular expression for finding URLs by given top-level domains.
        
#             Parameters
#             ----------
#             tlds : Iterable[str]
#                 Top-level domains for which to create regex.
#             """
#         # Store values
#         self.tlds  = list(tlds)
#         self.flags = flags

#         # Get TLDs regex
#         self.tlds_re = re.compile(f"{DOT_SEP}({'|'.join(tlds)})", self.flags)
#         self.reverse_url_base = re.compile(
#             rf"^([a-zA-Z0-9]|[$\-_&+]|{DOT_SEP}|([0-9a-fA-F]{2}%))+"
#         )
#         self.url_base = re.compile(rf"{URL_BASE}({PORT})?(?<![:\?\[\]@!'\(\),;.])")
#         self.regex_end = re.compile(r'[0-9a-zA-Z\-]', re.IGNORECASE)


#     def finditer(self, text: str) -> Iterable[re.Match]:
#         """Iterate over all matches for URLs based on top-level domains.
        
#             Parameters
#             ----------
#             text : str
#                 Text to search for matching URLs.

#             Yields
#             ------
#             match : re.Match
#                 URL matches in text.
#             """
#         # Save matches that have already been found to avoid double matches
#         found = set()

#         # Find top-level domains
#         for match in re.finditer(self.tlds_re, text):
#             # Check character immediately after domaind
#             if (
#                 match.end() < len(text) and
#                 self.regex_end.match(text[match.end()])
#                 ):
#                 continue

#             # Get string until TLD
#             rstring = text[match.start()-1::-1]

#             # Find beginning of URL
#             for rmatch in re.finditer(self.reverse_url_base, rstring):
#                 # Find URL from given starting position
#                 result = self.url_base.search(
#                     text, pos=match.start()-rmatch.end()
#                 )

#                 # Skip if we already found the result before
#                 if result.span() in found: continue

#                 # Skip if we actually found a file extension
#                 skip = False
#                 for ext in get_file_extensions():
#                     ext = re.compile(rf'{DOT_SEP}{ext}', re.IGNORECASE)
#                     if ext.search(result.group(0)):
#                         skip = True
#                         break
#                 if skip: continue

#                 # Add result
#                 found.add(result.span())
#                 # Yield result
#                 yield result

    
#     def __eq__(self, other):
#         """Set equivalence relation."""
#         return self.tlds == other.tlds and self.flags == other.flags

EMAIL =(
    r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:" + DOT_SEP + r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|" + '"'
    r"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*" + '"'
    r")" + AT_SEP +
    r"(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?" + DOT_SEP + r")+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)" + DOT_SEP + r"){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
)

# Hashes
MD5    = rf'{HEX}{{32}}'
SHA1   = rf'{HEX}{{40}}'
SHA256 = rf'{HEX}{{64}}'
SHA512 = rf'{HEX}{{128}}'

# MITRE frameworks
CVE = r"[cC][vV][eE]-\d{4}-\d{4,6}"
CWE = r"[cC][wW][eE]-\d{1,4}"
CAPEC = r"[cC][aA][pP][eE][cC]-\d{1,4}"
ATTACK = r"(TA|T|DS|S|G|M)\d{4}(\.\d{3})?"

# Windows paths and registry
FILE = rf'{PATH}{EXTENSIONS}|{PATH_BASE}{EXTENSIONS}'
REGISTRY = r"(HKEY_LOCAL_MACHINE\\|HKLM\\)([a-zA-Z0-9_@\-\^!#.\:\/\$%&+={}\[\]\\*])+"

################################################################################
#                            Dictionary of all IOCs                            #
################################################################################

IOCs = OrderedDict([
    ('IPv4'    , re.compile(IPv4_full, re.IGNORECASE)),
    ('IPv6'    , re.compile(IPv6_full, re.IGNORECASE)),
    ('EMAIL'   , re.compile(EMAIL    , re.IGNORECASE)),
    ('SHA512'  , re.compile(SHA512   , re.IGNORECASE)),
    ('SHA256'  , re.compile(SHA256   , re.IGNORECASE)),
    ('SHA1'    , re.compile(SHA1     , re.IGNORECASE)),
    ('MD5'     , re.compile(MD5      , re.IGNORECASE)),
    ('CVE'     , re.compile(CVE      , re.IGNORECASE)),
    ('CWE'     , re.compile(CWE      , re.IGNORECASE)),
    ('CAPEC'   , re.compile(CAPEC    , re.IGNORECASE)),
    ('ATTACK'  , re.compile(ATTACK   , re.IGNORECASE)),
    ('FILE'    , re.compile(FILE     , re.IGNORECASE)),
    ('PATH'    , re.compile(PATH     , re.IGNORECASE)),
    ('REGISTRY', re.compile(REGISTRY , re.IGNORECASE)),
    # ('BASE64'  , re.compile(BASE64   , re.IGNORECASE)),
    # ('URL'     , re.compile(URL     , re.IGNORECASE)),
    ('URL'     , URLRegex(
        schemes        = get_uri_schemes(),
        tlds           = get_tlds(),
        tld_exceptions = get_file_extensions(),
        flags          = re.IGNORECASE
    )),
])

################################################################################
#                                 I/O methods                                  #
################################################################################

def iocs2json(iocs: Dict[str, re.Pattern]) -> str:
    """Transform IOC regex dictionary to json string.
        Inverse of json2iocs.
    
        Parameters
        ----------
        iocs : Dict[str, re.Pattern]
            IOCs to transform into json string.
        
        Returns
        -------
        json : str
            JSON string representing iocs.
        """
    # Initialise result
    result = OrderedDict()

    # Loop over all iocs
    for key, regex in iocs.items():
        # Transform to json-compatible dictionary
        if isinstance(regex, re.Pattern):
            result[key] = {
                'pattern': regex.pattern,
                'flags'  : regex.flags,
                'type'   : 'regex',
            }
        elif isinstance(regex, URLRegex):
            result[key] = {
                'type'  : 'URLRegex',
                'params': {
                    "schemes"       : regex.schemes,
                    "tlds"          : regex.tlds,
                    "tld_exceptions": regex.tld_exceptions,
                    "base"          : regex.base,
                    "netloc"        : regex.netloc,
                    "netloc_r"      : regex.netloc_r,
                    "dot"           : regex.dot,
                    "flags"         : regex.flags,
                },
            }
        else:
            raise ValueError(f"Unsupported regex type: '{type(regex)}'")

    # Return as json string
    return json.dumps(result, indent='    ')


def json2iocs(iocs_json: str) -> Dict[str, re.Pattern]:
    """Transform json string to IOC regex dictionary.
        Inverse of iocs2json.
    
        Parameters
        ----------
        iocs_json : str
            JSON string representing iocs.
            
        Returns
        -------
        iocs : Dict[str, re.Pattern]
            IOCs extracted from json string.
        """
    # Initialise result
    result = OrderedDict()

    # Loop over all IOCs
    for key, value in json.loads(iocs_json).items():
        if value.get('type') == 'regex':
            result[key] = re.compile(value['pattern'], value['flags'])
        elif value.get('type') == 'URLRegex':
            result[key] = URLRegex(**value.get('params'))
        else:
            raise ValueError(f"Unsupported regex type: '{value.get('type')}'")

    # Return result
    return result

def jsonfile2iocs(path: Union[str, pathlib.Path]) -> Dict[str, re.Pattern]:
    """Load IOC regex dictionary from json file.
        Opens file and uses json2ioc to convert file to IOC dictionary.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path from which to load IOC dictionary.

        Returns
        -------
        iocs : Dict[str, re.Pattern]
            IOCs extracted from json file.
        """
    # Open file
    with open(path) as infile:
        # Read file and return as IOC dictionary
        return json2iocs(infile.read())

################################################################################
#                                     Main                                     #
################################################################################

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = 'Store IOCs in json file',
        formatter_class = argformat.StructuredFormatter,
    )

    # Optional arguments
    parser.add_argument('json', help='json file to store IOCs')

    # Parse arguments
    args = parser.parse_args()

    # Write IOCs to json file
    with open(args.json, 'w') as outfile:
        outfile.write(iocs2json(IOCs))


if __name__ == '__main__':
    main()
