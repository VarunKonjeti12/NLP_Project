import yake # For Keyword Extraction
from transformers import BartTokenizer, BartForConditionalGeneration # For Text Summarization
#For aspect based sentimental analysis
import nltk
from nltk.corpus import stopwords
import stanza
# For Keyword Extraction
def extract_keywords(text, language="en", max_keywords=10):
    # Create a YAKE keyword extractor
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_keywords, dedupLim=0.7, dedupFunc='seqm')

    # Extract keywords
    keywords = kw_extractor.extract_keywords(text)

    # Deduplicate keywords using a set and combining similar keywords
    unique_keywords = set()
    deduplicated_keywords = []
    for keyword, score in keywords:
        # Check for similar keywords already in the set
        similar_keywords = [kw for kw in unique_keywords if kw in keyword or keyword in kw]
        
        if not similar_keywords:
            deduplicated_keywords.append(keyword)
            unique_keywords.add(keyword)

    # Return the list of deduplicated keywords
    return deduplicated_keywords
#Text Summarization 
# Loading pre-trained model and tokenizer

def text_summarization(product_review):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenizing and generating summary
    inputs = tokenizer.encode("summarize: " + product_review, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=50, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
  # Here instead of narrowsing the sentiment of aspects into positive or negative,
#we are returning the sentiment word and leaving the choice to the user itself for judging the sentiment of the aspect.
#( Clearly shown in the output)  
def aspect_sentiment_analysis(txt, stop_words, nlp):
    
    txt = txt.lower()  # Lowercasing the given Text
    sentList = nltk.sent_tokenize(txt)  # Splitting the text into sentences

    fcluster = []
    dic = {}

    for line in sentList:
        txt_list = nltk.word_tokenize(line)  # Splitting up into words
        taggedList = nltk.pos_tag(txt_list)  # Doing Part-of-Speech Tagging to each word

        doc = nlp(line)  # Object of Stanza NLP Pipeline
        
        # Getting the dependency relations between the words
        dep_node = []
        for dep_edge in doc.sentences[0].dependencies:
            dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])

        # Converting it into an appropriate format
        for i in range(0, len(dep_node)):
            if (int(dep_node[i][1]) != 0):
                dep_node[i][1] = txt_list[(int(dep_node[i][1]) - 1)]
        featureList = []
        for i in taggedList:
            if i[1].startswith('JJ'):  # Filter adjectives
                featureList.append(i[0])

        for i in featureList:
            filist = []
            for j in dep_node:
                if ((j[0] == i or j[1] == i) and j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"]):
                    if j[0] == i:
                        filist.append(j[1])
                    else:
                        filist.append(j[0])
            fcluster.append([i, filist])

    for i in fcluster:
        aspect = i[0]
        related_adjectives = ' '.join(i[1]).replace(' ', '')  # Combine words and remove spaces

        # Check for negation and adjust sentiment
        if "not" in i[1]:
            aspect_tokens = txt.split()
            not_index = i[1].index("not")
            if not_index < len(aspect_tokens) - 1:
                next_word = aspect_tokens[not_index + 1]
                aspect = "not_" + aspect if next_word in stop_words else "not " + aspect
            related_adjectives = related_adjectives.replace("not", "")

        if aspect not in dic:
            dic[aspect] = related_adjectives
        else:
            dic[aspect] += ' ' + related_adjectives
            
    finalcluster = [[value, [key]] for key, value in dic.items()]
    return finalcluster

nlp = stanza.Pipeline()
stop_words = set(stopwords.words('english'))
