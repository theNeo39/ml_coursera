from PorterStemmer import PorterStemmer
from getVocabList import getVocabList
import re

def processEmail(email_contents):
    """preprocesses a the body of an email and
    returns a list of word_indices
    word_indices = PROCESSEMAIL(email_contents) preprocesses
    the body of an email and returns a list of indices of the
    words contained in the email.
    """

# Load Vocabulary
    vocabList = getVocabList()

# Init return value
    word_indices = []

# ========================== Preprocess Email ===========================

# Find the Headers ( \n\n and remove )
# Uncomment the following lines if you are working with raw emails with the
# full headers

# hdrstart = strfind(email_contents, ([chr(10) chr(10)]))
# email_contents = email_contents(hdrstart(1):end)

# Lower case
    email_contents = str.lower(email_contents)

# Strip all HTML
# Looks for any expression that starts with < and ends with > and replace
# and does not have any < or > in the tag it with a space
    rx = re.compile('<[^<>]+>')
    email_contents = rx.sub(' ', email_contents)
# Handle Numbers
# Look for one or more characters between 0-9
    rx = re.compile('[0-9]+')
    email_contents = rx.sub('number ', email_contents)

# Handle URLS
# Look for strings starting with http:// or https://
    rx = re.compile('(http|https)://[^\s]*')
    email_contents = rx.sub('httpaddr ', email_contents)

# Handle Email Addresses
# Look for strings with @ in the middle
    rx = re.compile('[^\s]+@[^\s]+')
    email_contents = rx.sub('emailaddr ', email_contents)

# Handle $ sign
    rx = re.compile('[$]+')
    email_contents = rx.sub('dollar ', email_contents)
    
    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]
    
    # Stem the email contents word by word
    stemmer = PorterStemmer()
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue
        if word in vocabList:
            word_indices.append(vocabList.index(word)+1)
    return word_indices

