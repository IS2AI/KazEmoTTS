""" from https://github.com/keithito/tacotron """

import re
from unidecode import unidecode


_whitespace_re = re.compile(r'\s+')

_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def replace_english_words(text):
    text = text.replace("bluetooth не usb", "блютуз не юэсби").replace("mega silk way", "мега силк уэй")
    return text

def kazakh_cleaners(text):
#    text = convert_to_ascii(text)
    text = lowercase(text)
#    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = replace_english_words(text)
    text = collapse_whitespace(text)
    return text.replace("c", "с").strip()
