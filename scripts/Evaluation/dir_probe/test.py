from fuzzywuzzy import fuzz
import unidecode


fuzz.ratio(unidecode.unidecode(str(returned_title)).lower(), unidecode.unidecode(str(title)).lower())